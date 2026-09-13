# ADR-013: Dynamic Generative LLM Skill Expansion & Semantic Equivalence Matching

## Status
Implemented (Released in `v1.2.0`, Phase 15; Supersedes static manual YAML taxonomy)

## Context
Hardcoding static skill whitelist arrays directly in source code classes (`MetadataExtractor.SKILLS = [...]`) causes silent false negatives for unlisted technologies (e.g., `ArgoCD`, `Pulumi`, `dbt`, `LlamaIndex`), fails on casing variations, and rejects candidates who possess specific cloud or framework implementations (e.g., a candidate with "AWS" and "GCP" failing a query for "Cloud").

While externalizing into a static `skills_taxonomy.yaml` was initially considered, static manual taxonomies in 2026 SaaS AI platforms create a never-ending maintenance burden. In modern engineering recruitment, tech stacks and open-source frameworks evolve too rapidly for static dictionaries to remain accurate.

## Decision
1. **Generative LLM Skill Expansion**: Shift from static manual taxonomies to LLM-driven structured query expansion. During job requirement extraction, the LLM emits a dynamic `skill_expansions: Dict[str, List[str]]` mapping parent competencies to specific technologies (e.g., `{"Cloud": ["AWS", "GCP", "Azure", "Kubernetes", "Docker"]}`, `{"DevOps": ["CI/CD", "Terraform", "GitHub Actions"]}`).
2. **Hierarchical Semantic Equivalence**: In `JobMatcher`, `_skill_matches_candidate()` checks both direct skill identity and candidate-held specialized sub-skills against the extracted expansions.
3. **Open-Ended Resume Section Parsing**: In `resume_rag.py`, complement the base whitelist with open-ended regex pattern matching for `SKILLS:` and `TECHNICAL SKILLS:` resume sections, preserving niche and emerging candidate skills directly in the vector metadata.

## Consequences
- **Positive**: Eliminates manual taxonomy YAML curation; adapts dynamically to novel tools and tech stacks; matches candidates based on deep conceptual equivalence (e.g. AWS satisfies Cloud).
- **Negative**: Adds structured output generation fields during requirement extraction; relies on LLM domain competence for query expansion.
