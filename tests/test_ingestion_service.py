import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_profile_matching.services.ingestion_service import IngestionService


class TestIngestionService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.temp_dir.name)

        # Create dummy candidate files
        self.valid_file = self.test_dir_path / "resume_jane_doe.txt"
        self.valid_file.write_text(
            "Jane Doe\nEXPERIENCE\n5 years of experience in Python, AWS, Docker.\nEDUCATION\nB.S. Computer Science\n"
        )

        self.empty_file = self.test_dir_path / "empty.txt"
        self.empty_file.write_text("   \n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ingest_file_nonexistent(self):
        """Test error handling when file does not exist."""
        service = IngestionService(pipeline=MagicMock())
        result = service.ingest_file(str(self.test_dir_path / "nonexistent.txt"))
        self.assertFalse(result["success"])
        self.assertIn("File not found", result["error"])

    def test_ingest_file_empty(self):
        """Test error handling when file content is empty."""
        service = IngestionService(pipeline=MagicMock())
        result = service.ingest_file(str(self.empty_file))
        self.assertFalse(result["success"])
        self.assertIn("Empty content", result["error"])

    def test_ingest_file_success(self):
        """Test successful single file ingestion with mock pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        service = IngestionService(pipeline=mock_pipeline)
        result = service.ingest_file(str(self.valid_file))

        self.assertTrue(result["success"])
        self.assertEqual(result["filename"], "resume_jane_doe.txt")
        self.assertEqual(result["candidate_name"], "Jane Doe")
        self.assertGreater(result["chunks_ingested"], 0)

        # Verify store.upsert was called on vector store (idempotent write)
        self.assertTrue(mock_pipeline.store.upsert.called)

    def test_ingest_directory_success(self):
        """Test ingesting an entire directory of resume files."""
        mock_pipeline = MagicMock()
        mock_pipeline.embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        service = IngestionService(pipeline=mock_pipeline)
        result = service.ingest_directory(str(self.test_dir_path))

        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 2)  # valid_file + empty_file
        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed"], 1)

    def test_ingest_directory_nonexistent(self):
        """Test directory ingestion error handling for missing dir."""
        service = IngestionService(pipeline=MagicMock())
        result = service.ingest_directory(str(self.test_dir_path / "missing_dir"))
        self.assertFalse(result["success"])
        self.assertIn("Directory not found", result["error"])

    def test_ingest_stream_success(self):
        """Test in-memory zero-disk stream ingestion (ADR-016)."""
        mock_pipeline = MagicMock()
        mock_pipeline.embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        service = IngestionService(pipeline=mock_pipeline)
        raw_text = (
            "Alice Walker\nEXPERIENCE\n6 years in Java, Python, and AWS Cloud.\nEDUCATION\nM.S. Software Engineering\n"
        )
        stream_bytes = raw_text.encode("utf-8")

        result = service.ingest_stream("alice_walker_resume.txt", stream_bytes)
        self.assertTrue(result["success"])
        self.assertEqual(result["candidate_name"], "Alice Walker")
        self.assertTrue(result["filepath"].startswith("stream://alice_walker_resume.txt"))
        self.assertIn("hash=", result["filepath"])
        self.assertIn("raw_text", result)
        self.assertGreater(result["chunks_ingested"], 0)
        self.assertTrue(mock_pipeline.store.delete.called)
        self.assertTrue(mock_pipeline.store.upsert.called)

    def test_ingest_stream_empty(self):
        """Test empty stream buffer handling."""
        service = IngestionService(pipeline=MagicMock())
        result = service.ingest_stream("empty.txt", b"")
        self.assertFalse(result["success"])
        self.assertIn("Empty stream", result["error"])


if __name__ == "__main__":
    unittest.main()
