# ADR-001 – Adopt Hybrid Rust CLI and Python Worker

## Status
Accepted – Phase 0

## Context
We need a responsive terminal experience for musicians while iterating quickly on
text-to-audio research. Rust, paired with Ratatui, offers excellent terminal
ergonomics and async orchestration. Audio-model experimentation, however, moves
fast inside the Python ecosystem (`torch`, `diffusers`, `torchaudio`), and
staying in Python avoids constant FFI or re-implementation work each time we try
a new backend.

## Decision
Split responsibilities between a Rust CLI and a Python worker:

- **Rust (cli/)** – Owns the event loop, prompt capture, job queue UI, artifact
  persistence, and local playback.
- **Python (worker/)** – Exposes HTTP APIs around audio inference pipelines,
  normalisation, and artifact management.
- Communicate over HTTP/JSON initially, keeping the protocol simple and easily
  debuggable.

## Consequences
- Two toolchains raise onboarding cost; `make setup` + `uv` scripts mitigate it.
- Clear seams enable swapping either side independently (future web front-ends
  or native Rust inference).
- We accept some cross-language friction for now, but gain velocity when trying
  new audio models or diffusers releases.

