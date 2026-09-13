# ADR-009: LLM-Driven Intent Routing, Dynamic Anchors & Semantic Caching

## Status
Accepted (Implemented in `v1.1.0`, Re-architected to LLM-Driven & Dynamic Anchors in `v1.2.0`)

## Context
Using rigid substring keyword matching (e.g. `if "search" in text`) or hardcoded static string example lists in conversational state machine routers is brittle, fails on natural language variations, typos, domain slang, and negations, and requires endless maintenance of handcrafted example dictionaries. Furthermore, in real-world recruitment workflows, recruiters alternate between candidate sourcing, constraint adjustments, general technology inquiries, and external internet search queries (e.g. *"can you tell me about graph engineering in 2026"*, *"search google for python 3.14 features"*).

If an initial state check naively assumes any message with an unpopulated requirements dictionary is a Job Description or candidate sourcing request, innocent conversational and technical inquiries are misrouted into `extract_requirements_node`. This triggers full candidate screening, hybrid retrieval, ranking, and deep screening cascades, polluting state with rejected candidate cards and creating severe UX confusion.

## Decision
Implement a **Production LLM-Driven Intent Router** with dynamic anchor synthesis and in-memory semantic caching in `src/agentic_profile_matching/agent/routers.py`:

1. **Zero-Token Structural Fast-Paths**:
   - Empty input or empty messages immediately route to `extract_requirements`.
   - Multi-line raw Job Description pastes (> 3 lines with standard structural headers like `Job Description:`, `Responsibilities:`, `Requirements:`) route to `extract_requirements` without burning LLM tokens on raw document pastes.
2. **In-Memory Query Routing Cache (`_ROUTING_CACHE`)**:
   - Resolves repeated queries within the same conversational context (`(query.lower(), has_requirements, has_shortlist)`) in **0.00ms with zero LLM token consumption**.
3. **Primary Authority — LLM Structured Intent Classifier (`_classify_via_llm`)**:
   - The LLM acts as the primary, authoritative router, evaluating incoming user prompts via `with_structured_output(RouteDecision)` (`Literal["extract_requirements", "adjust_requirements", "conversational_query"]` + reasoning).
   - Seamlessly generalizes across any natural language prompt, multilingual input, domain jargon, and conversational context without relying on rigid keyword lists.
   - Provider credentials are resolved statelessly from `RunnableConfig` or environment variables (preserving ADR-011 isolation).
4. **Secondary Tier — Dynamic LLM Anchor Synthesis (`generate_dynamic_intent_anchors`)**:
   - Eliminates hardcoded static string example lists. Intent anchors can be dynamically synthesized by the LLM on startup or demand (`generate_dynamic_intent_anchors(llm)`).
   - For offline/local initialization, utilizes high-dimensional **Rich Semantic Intent Signatures** representing the core conceptual capability of each branch.
   - Local vector router computes cosine similarity against these dynamic prototypes using `SentenceTransformer('all-MiniLM-L6-v2')` as a fast local fallback.
5. **Tertiary Tier — State-Aware Resilient Fallback**:
   - Clean state-aware heuristics ensure graceful pipeline continuity if external LLM APIs are unreachable.
6. **Conversational Search Tool Integration**:
   - `conversational_query_node` is equipped with `search_web_tool` and `fetch_candidate_notes_tool` with dual-mode in-process execution fallback (`USE_MCP=False`), answering tech and web queries directly without evaluating the candidate talent pool.

## Consequences
- **Positive**:
  - Eliminates all hardcoded keyword arrays (`sourcing_patterns`, `conversational_prefixes`) and static phrase dictionaries (`SEMANTIC_INTENT_ANCHORS`) from source code.
  - Generates dynamic intent anchors via LLM, adapting to evolving recruitment and technology terminology.
  - Zero-cost, 0ms repeat query execution via in-memory query routing cache.
  - Completely separates candidate resume screening from general technical inquiries and Google/web searches.
- **Negative**:
  - LLM classification adds a minor 80–150ms network hop on first-turn cold queries (mitigated by ultra-fast inference and routing caching).
