# ADR-011: Stateless Credential Isolation & Checkpoint Security

## Status
Implemented (Released in `v1.2.0`, Phase 15)

## Context
In agentic graph workflows governed by persistence layers (such as LangGraph's `MemorySaver` or Redis checkpointers), state dictionaries (`AgentState`) are serialized and snapshotted at every node boundary. Storing sensitive parameters (such as `api_key`, `api_url`, or provider credentials) directly within `AgentState` violates **CWE-312 (Cleartext Storage of Sensitive Information)**. Any state export, audit log, debugging trace, or distributed state synchronization will leak active API credentials.

## Decision
1. **Purge State Credentials**: Eliminate `api_key`, `llm_provider`, `llm_model`, and `api_url` from serializable `AgentState` / `TypedDict`.
2. **Context-Driven Runtime Injection**: Resolve and instantiate LLM instances at the invocation boundary and pass pre-instantiated clients exclusively via `RunnableConfig` (`config["configurable"]["llm"]`) or an in-memory thread-safe `CredentialStore` keyed by `thread_id`.
3. **Structured Log Sanitization**: Introduce a standard logging filter that automatically masks API keys, bearer tokens, and credential headers across all stdout/stderr streams.

## Consequences
- **Positive**: Guarantees zero credential leakage in state checkpoints, database exports, and distributed traces; complies with enterprise security and SOC2 audit requirements.
- **Negative**: Requires passing instantiated LLM objects or thread-keyed config maps when invoking the compiled graph.
