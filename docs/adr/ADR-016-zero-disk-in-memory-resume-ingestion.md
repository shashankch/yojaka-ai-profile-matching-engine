# ADR-016: Zero-Disk In-Memory Resume Ingestion & Ephemeral PII Isolation

## Status
Implemented (Released in `v1.2.0`, Phase 15; Hardened in Phase 15.1)

## Context
Standard enterprise ingestion pipelines typically stage uploaded candidate resumes on server-side disk storage (e.g., `/tmp/resumes/`) before invoking text extractors and embedding models, and persist chunks into a shared global vector database. In containerized cloud environments (such as Streamlit Community Cloud, AWS ECS, or Kubernetes pods) and privacy-sensitive HR recruitment contexts, writing candidate resumes to local server storage presents severe drawbacks:
1. **Data Leakage & Privacy**: Leaving unencrypted candidate PII on shared host filesystems or shared database collections violates GDPR, SOC2, and data retention guidelines.
2. **Multi-Tenant Isolation**: Concurrent recruiter sessions sharing a persistent vector store risk leaking candidate PII across sessions.
3. **Ephemeral Cloud Failures**: Container restarts or horizontal autoscaling instances do not share ephemeral `/tmp` storage.
4. **Disk Quota Exhaustion**: High-volume recruiter uploads can rapidly exhaust ephemeral container storage limits.

## Decision
1. **In-Memory Stream Processing (`io.BytesIO`)**: Implement `IngestionService.ingest_stream(filename, stream_bytes)` to process candidate resumes entirely in memory, without performing any filesystem writes.
2. **Stream-Compatible Document Parsing**: Leverage PyMuPDF `pymupdf.open(stream=stream_bytes, filetype="pdf")` with single-pass page extraction for PDFs, `docx.Document(io.BytesIO(stream_bytes))` for Word documents, and direct UTF-8 decode for plain text.
3. **CompositeVectorStore & Ephemeral Session Isolation**:
   - Introduce `CompositeVectorStore(base_store, ephemeral_store)` conforming to `BaseVectorStore`.
   - The shared persistent store (`vector_store`) holds only read-only pre-indexed baseline candidate profiles from disk (`data/resumes`).
   - Each recruiter session allocates a session-scoped in-memory collection (`uploads_{session_id}`) via `ChromaVectorStore(ephemeral=True)` powered by `chromadb.EphemeralClient()`.
   - User-uploaded resumes are vectorized strictly into RAM; zero uploaded candidate PII is ever written to disk at `VECTOR_DB_PATH`.
   - Chunks and collections are automatically garbage collected when the session closes.
4. **Chunk Provenance, Deduplication & Pruning**:
   - Each uploaded file is identified by content digest: `f"{filename}_{size}_{sha256[:12]}"`.
   - Chunk IDs follow the scheme `stream_{filename}_{content_hash}_{section_clean}_{idx}` with reference `stream://{filename}?hash={content_hash}`.
   - Re-uploading an updated version of a candidate automatically prunes prior chunks for that filename via `store.delete(where={"filename": filename})`.
5. **Anti-XSS HTML Sanitization**:
   - All candidate-derived tokens and fields (`name`, `skills`, `education`, `experience_years`, `score`, `rel_path`) are strictly escaped via `html.escape(..., quote=True)` before HTML interpolation in candidate cards.
6. **Upload Limits Guardrail**:
   - A strict 10MB maximum file size limit is enforced on `uf.size` at the UI boundary before `uf.getvalue()` is materialized into process memory.
7. **Dedicated Ingestion Workspace**:
   - Provide a dedicated workspace in **Tab 2: Resume Ingestion & Talent Pool** featuring live chunk metrics, drag-and-drop ingestion, and an interactive candidate pool inventory table reflecting both disk and in-memory session uploads.

## Consequences
- **Positive**: Zero residual candidate files or PII on disk; multi-tenant session isolation preventing cross-session candidate data leaks; complete compatibility with read-only and ephemeral container environments; zero disk I/O latency; satisfies strict enterprise PII data-at-rest policies; high UI/UX visibility for recruiters.
- **Negative**: Server RAM temporarily buffers file streams and in-memory embeddings during ingestion (safely bounded by a 10MB file-size limit enforced at the UI boundary and process memory ceilings).

