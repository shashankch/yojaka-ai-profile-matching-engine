import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from agentic_profile_matching.resume_rag import (
    ResumeRAGPipeline,
    MetadataExtractor,
    ResumeChunker,
)
from agentic_profile_matching.fs_tools import (
    read_file as direct_read_file,
    list_files as direct_list_files,
)

from agentic_profile_matching.stores import BaseVectorStore

logger = logging.getLogger("ingestion_service")


class IngestionService:
    """
    Business logic service for candidate resume ingestion into the vector store.
    Decouples protocol handlers (e.g., FastMCP filesystem server) from RAG ingestion mechanics.
    """

    def __init__(
        self,
        store: Optional[BaseVectorStore] = None,
        pipeline: Optional[ResumeRAGPipeline] = None,
    ):
        self._store = store
        self._pipeline = pipeline
        if self._pipeline is None and self._store is not None:
            self._pipeline = ResumeRAGPipeline(store=self._store)

    @property
    def pipeline(self) -> ResumeRAGPipeline:
        """Lazily initialize the underlying vector store RAG pipeline if not injected."""
        if self._pipeline is None:
            self._pipeline = ResumeRAGPipeline(store=self._store)
        return self._pipeline

    def ingest_file(self, filepath: str) -> Dict[str, Any]:
        """
        Ingest a single candidate resume file (PDF, TXT, DOCX) into the vector database.

        Args:
            filepath: Absolute or relative path to the resume file.

        Returns:
            Dict containing ingestion status, metadata, and ingested chunk count.
        """
        path = Path(filepath).resolve()
        if not path.exists() or not path.is_file():
            logger.warning(f"File not found for ingestion: {filepath}")
            return {
                "success": False,
                "filepath": str(path),
                "error": f"File not found: {filepath}",
            }

        data = direct_read_file(str(path))
        if not data.get("success"):
            logger.error(f"Failed to read file for ingestion: {filepath}")
            return {
                "success": False,
                "filepath": str(path),
                "error": data.get("error", "Read failed"),
            }

        text = data.get("content", "")
        if not text or not text.strip():
            logger.warning(f"Empty content in file for ingestion: {filepath}")
            return {"success": False, "filepath": str(path), "error": "Empty content"}

        extractor = MetadataExtractor()
        chunker = ResumeChunker()

        filename = path.name
        meta = extractor.extract(filename, text)
        chunks = chunker.chunk(text)

        logger.info(
            f"Ingesting file '{filename}' - Candidate: {meta['candidate_name']}, "
            f"Exp: {meta['experience_years']} yrs, Skills: {len(meta['skills'])}, Chunks: {len(chunks)}"
        )

        added_chunks = 0
        pipeline = self.pipeline

        for idx, ch in enumerate(chunks):
            emb = pipeline.embedder.encode(ch["content"]).tolist()
            section_clean = ch["section"].lower().replace(" ", "_")
            chunk_id = f"{filename}_{section_clean}_{idx}"
            chunk_meta = {
                "candidate_name": meta["candidate_name"],
                "skills": ", ".join(meta["skills"]),
                "experience_years": int(meta["experience_years"]),
                "education": meta["education"],
                "resume_path": str(path),
                "filename": filename,
                "section": ch["section"],
            }

            pipeline.store.upsert(
                ids=[chunk_id],
                documents=[ch["content"]],
                embeddings=[emb],
                metadatas=[chunk_meta],
            )
            added_chunks += 1

        return {
            "success": True,
            "filepath": str(path),
            "filename": filename,
            "candidate_name": meta["candidate_name"],
            "chunks_ingested": added_chunks,
            "metadata": meta,
        }

    def ingest_directory(self, directory: str) -> Dict[str, Any]:
        """
        Ingest all supported resume files from a directory.

        Args:
            directory: Absolute or relative path to directory.

        Returns:
            Dict containing overall summary of directory ingestion results.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Directory not found for ingestion: {directory}")
            return {
                "success": False,
                "directory": str(dir_path),
                "error": f"Directory not found: {directory}",
            }

        files = direct_list_files(str(dir_path))
        results = []
        successful = 0
        failed = 0

        for f in files:
            res = self.ingest_file(f["path"])
            results.append(res)
            if res.get("success"):
                successful += 1
            else:
                failed += 1

        return {
            "success": True,
            "directory": str(dir_path),
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    def ingest_stream(self, filename: str, stream_bytes: bytes) -> Dict[str, Any]:
        """
        Ingest a resume directly from an in-memory byte buffer (zero-disk persistence).
        Implements ADR-016 architecture for secure, serverless resume upload.

        Args:
            filename: Name of the uploaded file (e.g. 'Jane_Doe_Resume.pdf').
            stream_bytes: Raw binary bytes of the resume.

        Returns:
            Dict containing ingestion status, extracted metadata, and ingested chunks count.
        """
        if not stream_bytes:
            return {"success": False, "filename": filename, "error": "Empty stream buffer"}

        suffix = Path(filename).suffix.lower()
        content = ""

        try:
            if suffix == ".txt":
                content = stream_bytes.decode("utf-8", errors="ignore")
            elif suffix == ".pdf":
                try:
                    import pymupdf

                    doc = pymupdf.open(stream=stream_bytes, filetype="pdf")
                    pages_text = []
                    for page in doc:
                        txt = page.get_text("text", sort=True)
                        if txt and txt.strip():
                            pages_text.append(txt)
                    content = "\n\n".join(pages_text)
                    doc.close()
                except Exception:
                    import io
                    from pypdf import PdfReader

                    reader = PdfReader(io.BytesIO(stream_bytes))
                    content = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif suffix == ".docx":
                import io
                from docx import Document

                doc = Document(io.BytesIO(stream_bytes))
                content = "\n".join(p.text for p in doc.paragraphs if p.text)
            else:
                return {"success": False, "filename": filename, "error": f"Unsupported file type: {suffix}"}
        except Exception as e:
            logger.error(f"Failed to parse in-memory stream for {filename}: {e}")
            return {"success": False, "filename": filename, "error": f"Parse error: {str(e)}"}

        if not content.strip():
            return {"success": False, "filename": filename, "error": "Extracted content is empty"}

        extractor = MetadataExtractor()
        chunker = ResumeChunker()

        meta = extractor.extract(filename, content)
        chunks = chunker.chunk(content)

        logger.info(
            f"Ingesting in-memory stream '{filename}' - Candidate: {meta['candidate_name']}, "
            f"Exp: {meta['experience_years']} yrs, Skills: {len(meta['skills'])}, Chunks: {len(chunks)}"
        )

        pipeline = self.pipeline
        added_chunks = 0
        content_hash = hashlib.sha256(stream_bytes).hexdigest()[:12]
        resume_ref = f"stream://{filename}?hash={content_hash}"

        # Clean up any prior chunks for this exact filename in the store before re-indexing
        try:
            if hasattr(pipeline.store, "delete"):
                pipeline.store.delete(where={"filename": filename})
        except Exception as e:
            logger.debug(f"Could not prune prior chunks for {filename}: {e}")

        for idx, ch in enumerate(chunks):
            emb = pipeline.embedder.encode(ch["content"]).tolist()
            section_clean = ch["section"].lower().replace(" ", "_")
            chunk_id = f"stream_{filename}_{content_hash}_{section_clean}_{idx}"
            chunk_meta = {
                "candidate_name": meta["candidate_name"],
                "skills": ", ".join(meta["skills"]),
                "experience_years": int(meta["experience_years"]),
                "education": meta["education"],
                "resume_path": resume_ref,
                "filename": filename,
                "content_hash": content_hash,
                "section": ch["section"],
            }

            pipeline.store.upsert(
                ids=[chunk_id],
                documents=[ch["content"]],
                embeddings=[emb],
                metadatas=[chunk_meta],
            )
            added_chunks += 1

        return {
            "success": True,
            "filepath": resume_ref,
            "filename": filename,
            "candidate_name": meta["candidate_name"],
            "chunks_ingested": added_chunks,
            "metadata": meta,
            "content_hash": content_hash,
            "raw_text": content,
        }

    def prune_stale_records(self) -> int:
        """
        Prunes orphan records from the vector store whose underlying files no longer exist on disk.
        Skips in-memory stream:// uploads.

        Returns:
            Count of deleted stale chunk IDs.
        """
        store = self.pipeline.store
        all_items = store.get_all()
        ids_to_delete = []
        for doc_id, meta in zip(all_items.get("ids", []), all_items.get("metadatas", [])):
            path_str = meta.get("resume_path")
            if path_str and not path_str.startswith("stream://"):
                if not Path(path_str).exists():
                    ids_to_delete.append(doc_id)

        if ids_to_delete:
            logger.info(f"Pruning {len(ids_to_delete)} orphan chunks from vector store.")
            try:
                if hasattr(store, "collection"):
                    store.collection.delete(ids=ids_to_delete)
                elif hasattr(store, "delete"):
                    store.delete(ids=ids_to_delete)
            except Exception as e:
                logger.warning(f"Failed to prune orphan chunks: {e}")
        return len(ids_to_delete)
