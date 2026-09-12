# ADR-012: Functional State Immutability & LangGraph Node Invariance

## Status
Implemented (Released in `v1.2.0`, Phase 15)

## Context
In LangGraph, node functions must behave as pure, idempotent transitions receiving an immutable state snapshot and returning partial state update dictionaries. In-place mutation of nested dictionary references (e.g. iterating over `state["shortlist"]` and directly mutating `c["strengths"] = ...` or `c["screening_status"] = ...`) breaks checkpoint idempotency. If a node fails or is interrupted mid-execution, re-running the node operates on corrupted, partially-mutated state.

## Decision
1. **Pure Functional Updates**: Mandate that all workflow nodes (`deep_screen_node`, `recommendation_node`, etc.) treat incoming candidate dicts as immutable data structures.
2. **Explicit Profile Copying**: Construct and return new candidate dictionaries (`{**c, ...}`) when updating evaluation metadata, returning clean replacement lists (`{"shortlist": screened + unscreened}`).
3. **Idempotent Retry Safety**: Guarantee that node re-execution from any checkpoint produces bit-identical results without side effects.

## Consequences
- **Positive**: Guarantees deterministic state checkpointing; eliminates race conditions and partial-update corruption during node retries or human-in-the-loop interrupts.
- **Negative**: Creates minor ephemeral dictionary allocations during screening loops.
