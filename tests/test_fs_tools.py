import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from agentic_profile_matching.fs_tools import (
    read_file,
    list_files,
    write_file,
    search_in_file,
)


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


def test_write_and_read_txt_file(temp_dir):
    file_path = temp_dir / "test.txt"
    content = "Hello, this is a test file for fs_tools."

    # Test write_file
    res_write = write_file(str(file_path), content)
    assert res_write["success"] is True
    assert res_write["bytes_written"] == len(content.encode("utf-8"))

    # Test read_file
    res_read = read_file(str(file_path))
    assert res_read["success"] is True
    assert res_read["content"] == content
    assert res_read["metadata"]["filename"] == "test.txt"
    assert res_read["metadata"]["extension"] == ".txt"


def test_read_file_not_found():
    res = read_file("/nonexistent/file/path.txt")
    assert res["success"] is False
    assert "File not found" in res["error"]


def test_read_unsupported_file(temp_dir):
    file_path = temp_dir / "test.xyz"
    file_path.write_text("dummy")
    res = read_file(str(file_path))
    assert res["success"] is False
    assert "Unsupported file type" in res["error"]


@patch("agentic_profile_matching.fs_tools.PdfReader")
def test_read_pdf_file_pypdf_fallback(mock_pdf_reader, temp_dir):
    file_path = temp_dir / "test.pdf"
    file_path.write_text("dummy")

    # Mocking PdfReader structure for pypdf fallback
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF page content"
    mock_pdf_reader.return_value.pages = [mock_page]

    with patch.dict("sys.modules", {"pymupdf": None, "fitz": None}):
        res = read_file(str(file_path))
        assert res["success"] is True
        assert res["content"] == "PDF page content"
        assert res["metadata"]["extension"] == ".pdf"


def test_read_pdf_file_pymupdf(temp_dir):
    file_path = temp_dir / "test_pymupdf.pdf"
    file_path.write_text("dummy")

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "PyMuPDF sorted layout text"
    mock_doc.__iter__.return_value = [mock_page]

    with patch("pymupdf.open", return_value=mock_doc, create=True):
        with patch("fitz.open", return_value=mock_doc, create=True):
            res = read_file(str(file_path))
            assert res["success"] is True
            assert res["content"] == "PyMuPDF sorted layout text"


def test_read_pdf_unstructured_option(temp_dir, monkeypatch):
    monkeypatch.setattr("agentic_profile_matching.config.USE_UNSTRUCTURED", True)
    file_path = temp_dir / "test_unstructured.pdf"
    file_path.write_text("dummy")

    mock_el = MagicMock()
    mock_el.__str__.return_value = "Unstructured element text"

    mock_partition = MagicMock(return_value=[mock_el])
    with patch.dict("sys.modules", {"unstructured.partition.pdf": MagicMock(partition_pdf=mock_partition)}):
        res = read_file(str(file_path))
        assert res["success"] is True
        assert res["content"] == "Unstructured element text"


@patch("agentic_profile_matching.fs_tools.Document")
def test_read_docx_file(mock_docx, temp_dir):
    file_path = temp_dir / "test.docx"
    file_path.write_text("dummy")

    # Mocking Document structure
    mock_para = MagicMock()
    mock_para.text = "DOCX paragraph content"
    mock_docx.return_value.paragraphs = [mock_para]

    res = read_file(str(file_path))
    assert res["success"] is True
    assert res["content"] == "DOCX paragraph content"
    assert res["metadata"]["extension"] == ".docx"


def test_list_files(temp_dir):
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.pdf").write_text("content2")
    (temp_dir / "sub_dir").mkdir()
    (temp_dir / "sub_dir" / "file3.txt").write_text("content3")

    # Test listing all files
    files = list_files(str(temp_dir))
    assert len(files) == 3

    # Test listing with extension filter
    txt_files = list_files(str(temp_dir), extension=".txt")
    assert len(txt_files) == 2
    assert all(f["name"].endswith(".txt") for f in txt_files)


def test_search_in_file(temp_dir):
    file_path = temp_dir / "test_search.txt"
    content = "The quick brown fox jumps over the lazy dog. Python is great. Another python reference."
    write_file(str(file_path), content)

    res = search_in_file(str(file_path), "python")
    assert res["success"] is True
    assert res["total_matches"] == 2
    assert len(res["matches"]) == 2
    assert "Python is great" in res["matches"][0]["context"]
