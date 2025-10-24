# Timbre Next-Branch Goal

## Branch Strategy
- **Start from clean branch**: `git checkout -b feature/tui-rust-orchestrator` once current Python CLI milestone is merged into `main`.
- Preserve Python CLI as functional reference; new work occurs in the feature branch to avoid regression risk.
- Gradually introduce Rust orchestrator components while keeping parity tests and session logs consistent between languages.

## Core Objectives
1. **Expressive TUI**
   - Rebuild the chat interface using Rust + Ratatui for fluid animations and low-latency rendering.
   - Reproduce the “flow state” gradient aesthetic (Codex-style shimmer) and animated thinking loop.
   - Support multi-panel layout: chat messages, parameter controls, progress timeline, output browser.
   - Integrate keyboard shortcuts (regenerate, tweak params) and mouse-friendly selection where feasible.
2. **Hybrid Runtime**
   - Rust orchestrator coordinates prompt handling, job queue, and process orchestration.
   - Retain Python backend adapters initially via FFI or IPC (e.g., JSON over stdio); plan future native Rust inference wrappers.
   - Package shared protobuf/JSON schema for generation requests/results to maintain compatibility.
3. **Backend Innovation**
   - Optimize MusicGen pipeline with caching, faster sample streaming, seed management, and optional LoRA patches.
   - Improve Riffusion spectral inversion (multi-pass Griffin-Lim, potential VAE decoder).
   - Explore additional backends (e.g., MusicGen streaming, Dance Diffusion) hidden behind feature flags.
4. **Audio Post-Processing**
   - Add mastering presets (loudness normalization, gentle compression, stereo widening).
   - Optional stem export using source separation (e.g., Demucs) invoked from Rust orchestrator.
   - Embed rich metadata (prompt, seed, backend version) into WAV/FLAC headers.
5. **Developer Experience**
   - Rust workspace using `cargo` with modules for `tui`, `orchestrator`, and `bridge` (Python integration).
   - Provide `make`/`just` targets for building Rust binaries and launching with Python workers.
   - Maintain parity CI (Rust fmt/clippy, Python lint/tests) and end-to-end smoke tests.

## Milestone Outline (Feature Branch)
1. **Foundations**
   - Scaffold Rust project, implement core event loop, connect to Python CLI for backend calls.
   - Define shared request/result schema.
2. **TUI MVP**
   - Ratatui interface replicating current CLI functionality: prompt input, progress animation, output listing.
   - Implement hot reload for theme/gradient experiments.
3. **Backend Optimization**
   - Profile generation pipeline, add streaming updates, integrate caching for model load.
   - Prototype additional backends or improve Riffusion output quality.
4. **Post-Processing & Extras**
   - Implement audio mastering toggles, metadata writing, optional playback controls.
   - Introduce webhook/DAW export automation.

## Success Criteria
- Rust TUI matches or exceeds Python CLI usability with sub-50 ms UI latency.
- Backend calls remain interchangeable (Python adapters still working while Rust experiment runs).
- Enhanced audio quality and reproducibility via metadata/stem exports.
- Documentation updated with dual-runtime workflow and branching instructions.

