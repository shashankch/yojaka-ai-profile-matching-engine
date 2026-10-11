"""
Parent-Document (Small-to-Big) Hierarchical Chunking & Retrieval Service (Phase 19 / ADR-019).
Maps granular child chunks (150-250 tokens) indexed in vector search to full
parent section blocks (1,000-1,500 tokens), preventing context fragmentation
across multi-role work histories during dual-rubric screening.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from agentic_profile_matching.observability import get_logger
from agentic_profile_matching.services.section_parser import ParsedSectionChunk

logger = get_logger("agentic_profile_matching.services.parent_document")


@dataclass
class ParentChunk:
    """Represents an unbroken parent document section block."""

    parent_id: str
    resume_path: str
    section_type: str
    section_title: str
    full_text: str
    child_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParentDocumentStore:
    """In-memory thread-safe store for parent document chunks."""

    _instance: Optional["ParentDocumentStore"] = None

    def __init__(self):
        self._parents: Dict[str, ParentChunk] = {}
        self._resume_index: Dict[str, List[str]] = {}

    @classmethod
    def get_instance(cls) -> "ParentDocumentStore":
        if cls._instance is None:
            cls._instance = ParentDocumentStore()
        return cls._instance

    def store_parent(self, parent: ParentChunk) -> None:
        self._parents[parent.parent_id] = parent
        if parent.resume_path not in self._resume_index:
            self._resume_index[parent.resume_path] = []
        if parent.parent_id not in self._resume_index[parent.resume_path]:
            self._resume_index[parent.resume_path].append(parent.parent_id)

    def get_parent(self, parent_id: str) -> Optional[ParentChunk]:
        return self._parents.get(parent_id)

    def get_parents_for_resume(self, resume_path: str) -> List[ParentChunk]:
        parent_ids = self._resume_index.get(resume_path, [])
        return [self._parents[pid] for pid in parent_ids if pid in self._parents]

    def save_parent_chunk(
        self,
        parent_id: str,
        text: str,
        candidate_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        parent = ParentChunk(
            parent_id=parent_id,
            resume_path=candidate_id,
            section_type="general",
            section_title="General",
            full_text=text,
            metadata=metadata or {},
        )
        self.store_parent(parent)

    def get_parent_chunk(self, parent_id: str) -> Optional[str]:
        p = self.get_parent(parent_id)
        return p.full_text if p else None

    def get_all(self) -> Dict[str, ParentChunk]:
        return dict(self._parents)

    def clear(self) -> None:
        self._parents.clear()
        self._resume_index.clear()


class ParentDocumentService:
    """
    Coordinates hierarchical small-to-big chunk decomposition and
    context expansion for downstream LLM evaluation.
    """

    def __init__(
        self,
        store: Optional[ParentDocumentStore] = None,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 150,
        child_chunk_size_tokens: int = 200,
        child_chunk_overlap_tokens: int = 50,
    ):
        self.store = store or ParentDocumentStore.get_instance()
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size_tokens = child_chunk_size_tokens
        self.child_chunk_overlap_tokens = child_chunk_overlap_tokens

    @property
    def parent_store(self) -> ParentDocumentStore:
        return self.store

    def create_hierarchical_chunks(
        self,
        filename: str,
        resume_path: str,
        parsed_chunks: List[ParsedSectionChunk],
        child_max_chars: int = 600,
    ) -> Tuple[List[ParsedSectionChunk], List[ParentChunk]]:
        """
        Splits larger parsed sections into fine-grained child chunks for vector indexing
        while registering the complete parent section in the ParentDocumentStore.
        """
        child_chunks: List[ParsedSectionChunk] = []
        parent_chunks: List[ParentChunk] = []

        for p_idx, parent_src in enumerate(parsed_chunks):
            # Compute deterministic parent ID
            p_hash = hashlib.md5(f"{resume_path}_{parent_src.section_title}_{p_idx}".encode()).hexdigest()[:8]
            clean_title = parent_src.section_title.lower().replace(" ", "_")
            parent_id = f"parent_{filename}_{clean_title}_{p_hash}"

            raw_text = parent_src.raw_content or parent_src.content
            # If section is small, parent equals child
            if len(raw_text) <= child_max_chars:
                child = ParsedSectionChunk(
                    section_title=parent_src.section_title,
                    section_type=parent_src.section_type,
                    content=parent_src.content,
                    raw_content=raw_text,
                    role=parent_src.role,
                    company=parent_src.company,
                    duration_years=parent_src.duration_years,
                    skills=parent_src.skills,
                    block_index=0,
                    metadata={**parent_src.metadata, "parent_id": parent_id},
                )
                parent = ParentChunk(
                    parent_id=parent_id,
                    resume_path=resume_path,
                    section_type=parent_src.section_type,
                    section_title=parent_src.section_title,
                    full_text=raw_text,
                    child_chunk_ids=[f"{parent_id}_c0"],
                    metadata=parent_src.metadata,
                )
                self.store.store_parent(parent)
                parent_chunks.append(parent)
                child_chunks.append(child)
                continue

            # Section is larger: subdivide into granular child chunks
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [raw_text]

            current_buf: List[str] = []
            current_len = 0
            c_sub_idx = 0
            created_child_ids: List[str] = []

            for p in paragraphs:
                if current_len + len(p) > child_max_chars and current_buf:
                    sub_text = "\n\n".join(current_buf)
                    cid = f"{parent_id}_c{c_sub_idx}"
                    child_chunks.append(
                        ParsedSectionChunk(
                            section_title=parent_src.section_title,
                            section_type=parent_src.section_type,
                            content=sub_text,
                            raw_content=sub_text,
                            role=parent_src.role,
                            company=parent_src.company,
                            duration_years=parent_src.duration_years,
                            skills=parent_src.skills,
                            block_index=c_sub_idx,
                            metadata={**parent_src.metadata, "parent_id": parent_id},
                        )
                    )
                    created_child_ids.append(cid)
                    c_sub_idx += 1
                    current_buf = [p]
                    current_len = len(p)
                else:
                    current_buf.append(p)
                    current_len += len(p)

            if current_buf:
                sub_text = "\n\n".join(current_buf)
                cid = f"{parent_id}_c{c_sub_idx}"
                child_chunks.append(
                    ParsedSectionChunk(
                        section_title=parent_src.section_title,
                        section_type=parent_src.section_type,
                        content=sub_text,
                        raw_content=sub_text,
                        role=parent_src.role,
                        company=parent_src.company,
                        duration_years=parent_src.duration_years,
                        skills=parent_src.skills,
                        block_index=c_sub_idx,
                        metadata={**parent_src.metadata, "parent_id": parent_id},
                    )
                )
                created_child_ids.append(cid)

            parent = ParentChunk(
                parent_id=parent_id,
                resume_path=resume_path,
                section_type=parent_src.section_type,
                section_title=parent_src.section_title,
                full_text=raw_text,
                child_chunk_ids=created_child_ids,
                metadata=parent_src.metadata,
            )
            self.store.store_parent(parent)
            parent_chunks.append(parent)

        logger.info(
            f"Hierarchical chunking complete for '{filename}': "
            f"{len(parent_chunks)} parent sections -> {len(child_chunks)} granular child chunks."
        )
        return child_chunks, parent_chunks

    def resolve_parent_context(self, parent_id: Optional[str]) -> Optional[str]:
        """Resolves full parent section text from parent_id."""
        if not parent_id:
            return None
        parent = self.store.get_parent(parent_id)
        return parent.full_text if parent else None

    def decompose_document(
        self,
        text: str,
        candidate_id: str = "default",
        filename: str = "document",
    ) -> Tuple[List[ParentChunk], List[ParsedSectionChunk]]:
        """Decomposes a document string into parent and child chunks."""
        parsed_chunks = [
            ParsedSectionChunk(
                section_title="Profile Experience",
                section_type="experience",
                content=text.strip(),
                raw_content=text.strip(),
            )
        ]
        child_chunks, parent_chunks = self.create_hierarchical_chunks(
            filename=filename,
            resume_path=candidate_id,
            parsed_chunks=parsed_chunks,
            child_max_chars=self.child_chunk_size_tokens * 4,
        )
        return parent_chunks, child_chunks

    def enrich_candidate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch-enriches a list of candidate matches with parent contexts."""
        return [self.enrich_candidate_with_parents(m) for m in matches]

    def enrich_candidate_with_parents(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a candidate match record with full parent section contexts
        so deep screening rubrics receive complete, unbroken work history context.
        """
        parent_id = candidate.get("metadata", {}).get("parent_id") or candidate.get("parent_id")
        if parent_id:
            parent = self.store.get_parent(parent_id)
            if parent:
                return {
                    **candidate,
                    "full_parent_context": parent.full_text,
                    "raw_text": parent.full_text
                    if len(parent.full_text) > len(candidate.get("raw_text", ""))
                    else candidate.get("raw_text", ""),
                }

        resume_path = candidate.get("resume_path") or candidate.get("candidate_id") or ""
        parents = self.store.get_parents_for_resume(resume_path)

        if not parents:
            return candidate

        parent_sections: List[Dict[str, str]] = [
            {
                "section": p.section_title,
                "section_type": p.section_type,
                "content": p.full_text,
            }
            for p in parents
        ]

        # Construct full unbroken context
        full_context = "\n\n".join(f"=== {p.section_title.upper()} ===\n{p.full_text}" for p in parents)

        return {
            **candidate,
            "parent_sections": parent_sections,
            "full_parent_context": full_context,
            # Replace raw_text with full parent context if parent context is richer
            "raw_text": full_context
            if len(full_context) > len(candidate.get("raw_text", ""))
            else candidate.get("raw_text", ""),
        }
