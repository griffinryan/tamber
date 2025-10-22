# Phase 0 Execution Plan – Discovery & Foundations

## Objectives
- Establish codebase scaffolding for a hybrid Rust (Ratatui) TUI and Python-based audio generation service centered on Riffusion.
- Validate local inference of Riffusion in a controllable environment to guarantee at least one end-to-end text → audio path.
- Define clear communication contracts and directory conventions so Phase 1 integration work can begin without rework.
- Deliver developer tooling and documentation that make onboarding trivial (`uv` + `cargo` setup, project structure tour).

## Success Criteria
- Repositories initialized (`cli/`, `worker/`, `docs/`) with lint/test tooling wired into CI.
- Riffusion backend runs locally with deterministic prompt, producing a WAV artifact saved to the designated artifact tree.
- Interface schema (`GenerationRequest`, `GenerationStatus`, `GenerationArtifact`) captured in shared specification (OpenAPI/proto/JSON schema).
- Initial Ratatui shell renders placeholder layout and can issue mock requests (no real networking yet).
- Developer setup guide verified on macOS (target platform) with documented resource requirements.

## Workstreams & Tasks

### 1. Repository & Tooling Scaffold
- **Task**: Initialize mono-repo structure
  - Create `cli/` (Rust crate), `worker/` (Python package), `docs/` directories.
  - Configure top-level `.gitignore`, `README.md` stub referencing ROADMAP.
- **Task**: Configure Python environment with `uv`
  - Add `pyproject.toml` (`worker/`) listing `uv`, `fastapi`, `uvicorn[standard]`, `numpy`, `torch`, `diffusers`, `torchaudio`, `soundfile`.
  - Create `uv.lock` via `uv sync` (document GPU/CPU notes).
  - Provide `Makefile` or `Justfile` commands: `setup`, `lint`, `test`, `run-worker`.
- **Task**: Configure Rust workspace
  - Set up `Cargo.toml` workspace with members `cli`.
  - Add baseline dependencies: `ratatui`, `crossterm`, `tokio`, `serde`, `serde_json`, `reqwest` (feature placeholder).
  - Set `rustfmt.toml`, `clippy` CI rules.
- **Task**: CI bootstrap (GitHub Actions)
  - Workflow running `cargo fmt --check`, `cargo clippy`, `cargo test`.
  - Python job running `uv pip compile` (if needed), `pytest`, `ruff`, `mypy`.

### 2. Riffusion Backend Validation
- **Task**: Research & pin Riffusion implementation path
  - Evaluate `riffusion/riffusion-model-v1` via `diffusers` with `StableDiffusionPipeline`.
  - Decide CPU-only vs. GPU baseline; note dependencies (`torch==2.x`, `accelerate`, `safetensors`).
- **Task**: Prototype generation script
  - Implement `scripts/riffusion_smoke.py` generating 5-second clip from prompt, storing WAV in `./artifacts/tmp`.
  - Log seed, prompt, inference time; ensure reproducibility by setting RNG seeds.
- **Task**: Package worker skeleton
  - Create `worker/app/models.py` with Pydantic schemas.
  - Implement `worker/app/services/riffusion.py` encapsulating pipeline load and generation.
  - Provide CLI entrypoint `python -m worker.generate --prompt "..."` for manual testing.
- **Task**: Document resource requirements
  - Capture memory/VRAM usage for CPU and GPU runs.
  - Note optional ONNX conversion path (future) and caching directories (`~/.cache/huggingface`).

### 3. Interface Contract & Process Communication
- **Task**: Draft schema specification
  - Choose transport (REST + WebSocket vs. gRPC). For Phase 0, document both and recommend REST for simplicity.
  - Define JSON schema for `GenerationRequest` (prompt, seed, model, duration), `GenerationStatus`, `GenerationArtifact`.
- **Task**: Create shared types
  - In Rust: `cli/src/types.rs` with placeholder structs mirrored from JSON.
  - In Python: reuse Pydantic models; ensure serialization compatibility.
- **Task**: Mock transport layer
  - Provide stub `worker/app/routes.py` with `/health` and `/generate` endpoints returning mock data.
  - Implement Rust client stub hitting `http://localhost:8000/health` to validate connectivity once worker online.

### 4. Ratatui Prototype
- **Task**: Establish UI skeleton
  - Implement layout with panes: conversation, prompt editor, status sidebar.
  - Add key handling for prompt submission, navigation, quitting.
- **Task**: Integrate mock data
  - Hard-code sample generation history and progress indicators to validate theme.
  - Set up state management struct (`AppState`) with placeholder job queue.
- **Task**: Logging & diagnostics
  - Integrate `tracing` + `tracing-subscriber` for logs.
  - Provide developer shortcut (`?`) to open help overlay (even if static).

### 5. Documentation & Developer Experience
- **Task**: Author `docs/setup.md`
  - Hardware assumptions, Python/Rust version requirements, environment variables, known issues.
  - Include instructions for downloading Riffusion weights (via Hugging Face auth if needed).
- **Task**: Create ADRs
  - ADR-001: Hybrid architecture decision (Rust UI + Python worker).
  - ADR-002: Transport protocol choice for MVP.
- **Task**: Update ROADMAP references
  - Cross-link Phase 0 deliverables with roadmap sections for traceability.

## Sequencing & Dependencies
1. Repository & tooling scaffold (Workstream 1) — unlocks parallel development.
2. Riffusion backend validation (Workstream 2) — ensures technical feasibility before UI integration.
3. Interface contract (Workstream 3) — depends on Workstream 2 discoveries for payload fields.
4. Ratatui prototype (Workstream 4) — can begin once Rust workspace exists; final integration deferred until backend stable.
5. Documentation (Workstream 5) — ongoing, but finalised after major tasks to capture accurate instructions.

## Resolved Inputs
- **Target hardware**: macOS ARM64 (Apple Silicon). Prioritise Metal acceleration via PyTorch MPS; ensure CPU fallback documented.
- **Acceleration assumptions**: Expect access to M3 Pro GPU during development; design pipelines that leverage MPS with CPU offload when needed.
- **Transport preference**: Start with HTTP polling for simplicity; leave WebSocket hooks in design notes for later scalability.
- **Model distribution**: Rely on Hugging Face downloads; document cache directories and authentication in a dedicated architecture doc.
- **Team focus**: Single cross-functional team for Phase 0 to land a functional prototype before branching into TUI/backend squads.

## Immediate Action Items
1. Circulate Phase 0 plan for sign-off, highlighting decisions above.
2. Assign temporary leads for TUI and backend domains (shared engineer acceptable) to coordinate Workstreams 1–4.
3. Draft `docs/architecture.md` capturing hybrid design, Hugging Face dependency flow, and transport rationale.
4. Schedule checkpoint at end of Week 1 to validate repository scaffold and Riffusion smoke test progress.

## Completion Notes
- `scripts/riffusion_smoke.py` and `python -m timbre_worker.generate` cover the backend smoke tests and CLI entrypoint planned for Workstream 2.
- JSON schemas, shared types, and HTTP endpoints are live; CLI/worker integration now exchanges real artifacts with placeholder fallback.
- Documentation updated (`docs/architecture.md`, `docs/setup.md`, `docs/testing/e2e.md`) and ADR-001/ADR-002 recorded the hybrid architecture and transport decisions.
- CI uses `uv` to mirror contributor workflows, ensuring lint/test parity with `make lint` / `make test`.
