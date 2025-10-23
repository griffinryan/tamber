# ADR-002 – HTTP Polling Transport for Phase 0

## Status
Accepted – Phase 0

## Context
The CLI needs to talk to the worker with minimal friction while we iterate on
the API. Candidates included gRPC streams (typed, bidirectional), WebSockets
(lightweight streaming), and plain HTTP polling. Early milestones prioritise
developer velocity over perfect efficiency, and our Ratatui prototype already
integrates `reqwest`.

## Decision
Expose REST endpoints from the worker and poll from the CLI:

- `/generate` submits a job and returns the initial status payload.
- `/status/{job_id}` surfaces progress updates (queued/running/etc.).
- `/artifact/{job_id}` returns metadata and filesystem paths once complete.
- `/health` reports baseline worker readiness/configuration.

## Consequences
- Polling is less efficient than streaming updates, but easiest to implement
  and debug. Latency remains acceptable for minute-scale inference.
- The API shapes the JSON schemas mirrored in Rust and Python, preventing
  type drift.
- Future milestones can add a streaming endpoint in parallel without breaking
  the existing CLI, because the status payload already contains enough context
  to feature-detect richer transports.

