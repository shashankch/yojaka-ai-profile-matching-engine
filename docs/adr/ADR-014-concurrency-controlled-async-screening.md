# ADR-014: Concurrency-Controlled Asynchronous Candidate Screening

## Status
Implemented (Released in `v1.2.0`, Phase 15)

## Context
Sequential execution of multi-candidate deep text screening in `deep_screen_node` (5 candidates $\times$ ~15s LLM audit) introduces 60–75 seconds of blocking latency, freezing web UI event loops. Unbounded parallel execution, however, triggers immediate HTTP 429 Rate Limit exceptions (RPM/TPM exhaustion) from cloud inference providers.

## Decision
1. **Bounded Concurrency Pool**: Implement concurrent candidate evaluation using `concurrent.futures.ThreadPoolExecutor` paired with a concurrency `threading.Semaphore(max_concurrent=2)`.
2. **Adaptive Jittered Throttling**: Apply randomized exponential backoff and inter-call pacing within the worker pool to stay strictly within provider rate-limit envelopes.
3. **Real-Time Event Streaming**: Emit per-candidate screening completion events to allow responsive UI progress updates.

## Consequences
- **Positive**: Cuts Round 2 screening wall-clock time from ~75s down to **~15–20s** (3.5x throughput gain) without exceeding provider RPM thresholds.
- **Negative**: Increases peak memory usage and concurrent connection count.
