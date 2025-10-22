# Timbre Architecture Overview

## 1. System Goals
- Provide a terminal-first creative workspace capable of generating `.wav`/`.mp3` clips from textual prompts with low friction.
- Leverage a Ratatui-based Rust frontend for responsive interaction, while delegating ML inference to a Python backend for ecosystem velocity.
- Support multiple audio generation backends starting with Riffusion, ensuring the architecture can expand to other models (MusicGen, Stable Audio, etc.).
- Maintain reproducibility, offline resilience, and straightforward path toward future desktop/web clients.

## 2. High-Level Topology

```
┌──────────────────────────┐        HTTP/JSON (polling)        ┌──────────────────────────┐
│   Ratatui CLI (Rust)     │  <──────────────────────────────> │  Python Worker (FastAPI) │
│ ──────────────────────── │                                   │ ──────────────────────── │
│ - tokio event loop       │  Submit GenerationRequest         │ - Riffusion pipeline     │
│ - ratatui components     │  Poll GenerationStatus            │ - Torch + MPS acceleration│
│ - rodio playback         │  Fetch GenerationArtifact         │ - Post-processing        │
│ - Local session state    │                                   │ - Artifact catalog       │
└──────────────────────────┘                                   └──────────────────────────┘
             │                                                              │
             │ Writes metadata & audio files                                │
             ▼                                                              ▼
    ~/.config/timbre/ (state)                              ~/Music/Timbre/ (artifacts)
```

- **Transport**: Phase 0 uses HTTP polling for simplicity; the worker exposes REST endpoints (`/generate`, `/status/{job_id}`, `/artifact/{job_id}`).
- **Process model**: CLI and worker run as separate processes. The CLI ships as a Rust binary; the worker runs via `uv`-managed Python environment.
- **Artifact flow**: Worker writes audio files to the artifact tree and returns metadata; CLI refreshes its library view, loads playback via `rodio`, and logs session history.

## 3. Key Components

### 3.1 Ratatui CLI (`cli/`)
- **Runtime**: `tokio` handles async networking and background tasks.
- **UI**: Ratatui defines layout panes (chat timeline, prompt editor, status sidebar, library view). `crossterm` manages terminal IO.
- **State Management**: `AppState` struct tracks active prompt, queued jobs, history, and configuration.
- **Networking**: `reqwest` client submits generation requests and polls status endpoints. Job updates drive UI notifications.
- **Playback**: `rodio` loads generated WAVs. For MP3 playback (future), integrate `symphonia`.
- **Logging**: `tracing` + `tracing-subscriber` for structured logs. Configurable verbosity via CLI flags.

### 3.2 Python Worker (`worker/`)
- **Framework**: `FastAPI` for HTTP APIs; `uvicorn` ASGI server. WebSocket endpoints reserved for future.
- **Inference**: `diffusers`-based Riffusion pipeline with PyTorch (`torch>=2.x`) using MPS backend (`mps` device) on Apple Silicon; CPU fallback using `float32` pipeline.
- **Job handling**: In-memory queue tracks job state (queued → running → complete/failed). For Phase 0, synchronous inference executes in request thread; future phases will introduce background tasks/workers.
- **Artifacts**: `soundfile` or `torchaudio` writes WAV to `~/Music/Timbre/<session>/<job_id>.wav`. Metadata (prompt, seed, duration) stored in JSON alongside audio.
- **Config**: `pydantic` models define requests/responses, aligning with Rust `serde` structs.
- **Logging**: `loguru` or standard logging with structured output, optionally mirrored to file for diagnostics.

### 3.3 Shared Schemas
- **GenerationRequest**: `prompt: str`, `seed: Optional[int]`, `duration_seconds: int`, `model_id: str`, `cfg_scale: Optional[float]`, `scheduler: Optional[str]`.
- **GenerationStatus**: `job_id: str`, `state: Literal["queued","running","succeeded","failed"]`, `progress: float`, `message: Optional[str]`.
- **GenerationArtifact**: `job_id: str`, `artifact_path: str`, `preview_url: Optional[str]`, `metadata: dict`.
- Store schema definitions in `docs/schemas/` (JSON schema) and reflect in Rust/Python code to avoid drift.

## 4. Platform Considerations
- **Target hardware**: macOS ARM64 (Apple Silicon, e.g., M3 Pro). Utilize Metal Performance Shaders (MPS) backend in PyTorch.
- **Dependencies**:
  - Install PyTorch nightly for optimal MPS support if stable release lags.
  - Ensure `accelerate`, `safetensors`, `diffusers`, `soundfile`, and `torchaudio` compiled for ARM64.
  - Document optional `brew install ffmpeg` for future MP3 exports and playback fallback.
- **Performance**: Monitor GPU memory usage; default to 5–10 s clip lengths during Phase 0. Provide configuration for downscaling resolution or using cached embeddings for speed.

## 5. Data & File Layout
- **Config directory** (`~/.config/timbre/`):
  - `config.toml` — user settings (default model, theme, output dir).
  - `history.jsonl` — append-only log of prompts and results.
  - `sessions/<session_id>.json` — optional detailed session state.
- **Artifact directory** (`~/Music/Timbre/`):
  - `YYYY/MM/DD/<session_id>/take-<n>.wav`
  - `take-<n>.json` containing metadata (prompt, seed, inference time, backend).
- **Cache**: Rely on Hugging Face default cache (`~/.cache/huggingface/hub`), note cleanup procedure and environment variables for relocation.

## 6. Future-Proofing Hooks
- **WebSocket upgrade path**: Design REST responses to include `status_endpoint` and `stream_endpoint` fields so the CLI can seamlessly switch to streaming when backend adds it.
- **Backend plugins**: Structure worker with adapter registry (`Backend` interface). Each backend reports capabilities (supports melody conditioning, requires GPU, etc.).
- **Rust-native inference**: Keep CLI logic modular so a future `backend_local` module can host ONNX/Candle inference without Python dependency.
- **Telemetry**: Reserve `telemetry_id` fields in requests to correlate events when analytics layer is added (opt-in).
- **Testing**: Provide integration harness `tests/integration/test_cli_worker.py` spinning up worker via `subprocess` and verifying CLI command triggers generation.

## 7. Developer Tooling
- **Python**: Use `uv` for dependency management. Provide `uv sync`, `uv run` commands for worker.
- **Rust**: Configure `cargo fmt`, `cargo clippy`, `cargo test`. Optionally use `just` recipes for cross-language tasks.
- **CI**: GitHub Actions runs lint/test on both languages. Document manual steps for enabling Metal backend (arm64 self-hosted runner required for GPU tests).
- **Docs**: Maintain ADRs under `docs/adrs/` capturing decisions (hybrid architecture, transport strategy, backend prioritization). Link relevant sections of this document.

## 8. Sequence for Phase 0
1. Implement repository scaffold and tooling (ensure `cli/` and `worker/` compile/run with placeholder logic).
2. Validate Riffusion inference pipeline on macOS M3 Pro via `scripts/riffusion_smoke.py`.
3. Solidify JSON schemas and shared models; create stub endpoints.
4. Build Ratatui mock UI with static data.
5. Iterate on dev docs once smoke test passes.

This document should guide future contributors on the architectural intent and integration points while Phase 0 evolves toward an end-to-end prototype.
