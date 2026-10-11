"""Services package for agentic profile matching engine."""

from agentic_profile_matching.services.ingestion_service import IngestionService
from agentic_profile_matching.services.section_parser import (
    SectionParser,
    ParsedSectionChunk,
    CANONICAL_SECTION_MAP,
)
from agentic_profile_matching.services.contextual_retrieval import (
    ContextualEnricher,
    ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE,
)
from agentic_profile_matching.services.reranker import (
    CrossEncoderReranker,
    compute_rrf_scores,
)
from agentic_profile_matching.services.hyde_service import (
    HyDEService,
    HYDE_SYSTEM_PROMPT,
)
from agentic_profile_matching.services.parent_document_service import (
    ParentDocumentService,
    ParentDocumentStore,
    ParentChunk,
)
from agentic_profile_matching.services.local_inference import (
    LocalInferenceService,
)
from agentic_profile_matching.services.faceted_filter import (
    FacetedFilter,
)

__all__ = [
    "IngestionService",
    "SectionParser",
    "ParsedSectionChunk",
    "CANONICAL_SECTION_MAP",
    "ContextualEnricher",
    "ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE",
    "CrossEncoderReranker",
    "compute_rrf_scores",
    "HyDEService",
    "HYDE_SYSTEM_PROMPT",
    "ParentDocumentService",
    "ParentDocumentStore",
    "ParentChunk",
    "LocalInferenceService",
    "FacetedFilter",
]
