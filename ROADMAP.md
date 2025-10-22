# Timbre CLI/TUI Roadmap

## 1. Vision & Design Tenets
- Deliver a terminal-first creative workspace where musicians iterate on prompts, audition generations, and export high-quality `.wav`/`.mp3` assets without leaving the shell.
- Anchor user experience in a Ratatui-powered interface that feels elegant, fast, and discoverable, with a conversational loop as the primary interaction pattern.
- Keep generation backends and UI orchestration loosely coupled so we can swap models, expand to web/desktop clients, or move inference between local and hosted environments with minimal refactoring.
- Favour reproducibility and offline resilience: deterministic seeds, cached model checkpoints, and clear metadata trails from prompt → render.
- Default to Rust for the TUI and orchestration layer; evaluate Rust vs. Python vs. hybrid strategies for inference logic to balance developer velocity with runtime performance.

## 2. Research Highlights

### 2.1 Text-to-Music & Audio Generation Landscape
- **MusicGen (Meta Audiocraft)**: Strong open-source baseline, supports text-only and melody conditioning. `facebook/musicgen-small` (≈1.5 GB) is feasible on modern laptops with CPU fallback; `-medium`/`-large` (≈3–7 GB) require >12 GB VRAM. Provides 32 kHz stereo output up to ~30 s. Hugging Face `transformers` integration already stable.
- **MusicGen-Lite & Moûsai**: Community finetunes with reduced parameter counts; good candidates for fast previews. Track Hugging Face for `musicgen-songstarter` variants.
- **Stable Audio**: Commercial API with high fidelity, streaming roadmap, and parameterized controls (duration, BPM, style). Requires REST key management and handles generation asynchronously.
- **Riffusion v1/v1.1**: Diffusion over spectrograms, CPU-friendly, ambient/texture oriented. Works with ONNX runtime and `diffusers`. Suitable for low-resource mode or background texture generation.
- **AudioLDM 2 / LatentComposer**: Offers prompt-based and reference-based generation, supports style transfer, more expressive but requires GPU. Integrates via `diffusers` pipelines.
- **Suno (Bark)**: Primarily text-to-speech with musicality; not ideal for full tracks but useful for vocals/vox layers when combined with instrumental backends.
- **Jukebox / MusicLM derivatives**: Research-grade, heavy, long render times; keep on watchlist but not roadmap-critical.
- **Hugging Face Inference Endpoints / Replicate / Together AI**: Hosted model marketplaces for rapid backend swaps, though cost and latency must be tracked.

### 2.2 Serving & Inference Architecture Options
- **Python-first service**: `FastAPI` or `Litestar` with `uvicorn`, wrapping `transformers`, `torchaudio`, and optional `audiocraft` modules. Easiest path to integrate new checkpoints, quantization (`bitsandbytes`), and LoRA adapters. Can expose REST + WebSocket for progress updates.
- **Rust-native inference**: `candle` (Hugging Face) and `burn` offer Rust inference for ONNX/Safetensors; however, MusicGen support is experimental and requires manual graph export. High effort but provides single-language story.
- **Hybrid microservice** (recommended short term): Ratatui client in Rust communicates with a Python worker over gRPC (`tonic` + `grpcio`), HTTP (`reqwest` + `FastAPI`), or message bus (ZeroMQ, NATS). Python retains model flexibility; Rust remains slim orchestrator.
- **Hosted API connectors**: Implement adapters for Stability Audio, OpenAI Audio, or third-party endpoints. Requires async job polling and usage tracking.
- **Model management**: Use Hugging Face Hub caching, optional local mirror via `huggingface_hub` CLI, and include tooling to convert checkpoints to ONNX for Rust-native experiments.

### 2.3 Terminal UI Stack & Patterns
- **Ratatui** (fork of tui-rs): Flexible layout, theme customization, and component ecosystem. Pair with `crossterm` for terminal control and `ratatui-async` or `ratatui-logger` as needed.
- **Event loop**: Use `tokio` runtime for async tasks, enabling background generation requests, progress streams, and audio playback updates.
- **Component ideas**: Chat timeline, prompt editor with syntax highlighting (via `syntect`), model parameter sidebar, job queue inspector, waveform/FFT rendering (basic ASCII sparkline, ability to preview amplitude).
- **Input UX**: Provide modal workflows (prompt composition vs. parameter editing), slash-commands for power users, and `Command Palette` for quick actions.
- **Accessibility**: Offer high-contrast themes, configurable keybindings, and minimal reliance on mouse.

### 2.4 Audio Processing & Playback Tooling
- **Rust**: `rodio` (simple playback), `cpal` (lower-level audio I/O), `hound` (WAV serialization), `symphonia` (decoding for future imports), `kira` (audio engine). Playback pipeline likely uses `rodio` for simplicity.
- **Python**: `torchaudio`, `librosa`, `audiomentations`, `pyloudnorm` for normalization, `soundfile` for writing FLAC/WAV. Use `demucs` for optional stem separation, `ffmpeg` for MP3 export.
- **Cross-communication**: When backend returns raw PCM or WAV path, Rust client can trigger playback via `rodio`, show waveform stats, and manage file organization.
- **Metadata**: Use `mutagen` (Python) or `lofty-rs` (Rust) to embed prompt, seed, model info into audio exports.

### 2.5 Persistence, Configuration, and Reproducibility
- Session history stored under `~/.config/timbre/` (`history.jsonl` or SQLite via `sqlx`). Include prompts, parameters, model version, seed, and file pointers.
- Artifact storage in `~/Music/Timbre/` with deterministic folder naming (`YYYY/MM/DD/<session-id>/take-<n>.wav`).
- Config management via `figment` or `serde`-based loader in Rust; mirror defaults in Python worker using `pydantic`.
- Provide declarative presets (`.toml` files) for genres, instrument profiles, mastering chains.

### 2.6 Observability & QA
- Structured logging with `tracing` (Rust) and `structlog`/`loguru` (Python) forwarded to files for debugging.
- Telemetry-ready: wrap in feature flag; support anonymized timing metrics (time-to-first-byte, render duration, failure rate).
- Testing strategy: unit tests for prompt parsing, CLI command routing, backend adapters; integration harness that spins up test worker and performs 2 s MusicGen Tiny render.

## 3. Architecture Options & Recommendation

| Option | Description | Pros | Cons | Recommended Use |
| --- | --- | --- | --- | --- |
| **A. All-in Rust** | Ratatui UI + inference using `candle`/ONNX inside the same binary. | Single deployment artifact, strong type safety, high performance. | Heavy upfront R&D to port models, limited ecosystem support, GPU story evolving. | Longer-term exploration once ONNX export stable. |
| **B. Hybrid (Rust UI + Python Service)** | Ratatui client orchestrates; Python worker handles model inference and audio post-processing. Communication via gRPC/WebSocket. | Rapid backend iteration, access to full Python ML stack, clean separation. | Requires process supervision, IPC complexity, packaging two runtimes. | **Primary roadmap path for alpha/beta releases.** |
| **C. Hosted API Connectors** | Rust client talks directly to SaaS APIs (Stability, OpenAI). | Minimal local setup, predictable latency on managed infra. | Cost, dependency on external uptime, limited offline capability. | Complementary path for users without GPU or quick trials. |

**Recommended Stack for MVP**: Option B with an architecture composed of:
1. **Rust TUI binary** (`timbre-cli`):
   - Ratatui + `tokio` runtime, `reqwest`/`tonic` client, `rodio` playback, `serde` config.
   - Maintains session state, handles user input, renders progress, and surfaces completed takes.
2. **Python inference service** (`timbre-worker`):
   - `FastAPI` + WebSocket for streaming status updates.
   - `musicgen` (Audiocraft) pipeline with optional quantization and scheduler controls.
   - Post-processing (normalization, metadata), file storage, and event notifications back to UI.
3. **Shared artifact & config layer**:
   - Local directories for assets, JSON/SQLite for logs.
   - Consistent schema definitions shared via protobuf/OpenAPI.
4. **Adapter layer** to plug in hosted APIs. Each adapter returns `GenerationJob` objects so the UI can treat local and remote jobs uniformly.

## 4. Phased Roadmap

### Phase 0 – Discovery & Foundations (Week 1)
- Confirm target developer hardware (CPU/GPU mix), storage expectations, and offline mode requirements.
- Decide packaging toolchain: `uv` for Python env management, `cargo` workspaces for Rust, `just` or `make` for orchestrating tasks.
- Draft high-level architecture diagram and interface contracts (`GenerationRequest`, `GenerationStatus`, `GenerationArtifact`).
- Bootstrap repos: `cli/` (Rust), `worker/` (Python) with shared docs folder. Establish lint/test CI (GitHub Actions CPU runner).

### Phase 1 – Backend Prototype (Weeks 2–3)
- Implement Python worker skeleton with `FastAPI` REST + WebSocket, `pydantic` models mirroring Rust structs.
- Integrate `musicgen-small` via `audiocraft` or Hugging Face `transformers`; add prompt, duration, tempo, style controls.
- Add asynchronous job queue (in-memory first, `redis` optional later) and progress events (queued → generating → mastering → ready).
- Implement WAV export, normalization, metadata embedding, and artifact catalog.
- Build CLI script to exercise worker independently; capture benchmarks (render time, CPU/GPU utilization).

### Phase 2 – Ratatui Shell & Orchestration (Weeks 3–4)
- Scaffold `timbre-cli` with `ratatui`, `crossterm`, and `tokio`. Implement core event loop and state management.
- Design UI components: chat timeline, prompt composer, status pane, audio library view. Establish styling/theme palette.
- Implement HTTP/gRPC client to submit jobs, poll status, and stream logs. Provide fallback spinner and progress indicators.
- Add local session logging, command palette (`:` commands), and auto-saving of prompts.
- Integrate `rodio` playback for completed takes; display waveform summary using ASCII sparkline.

### Phase 3 – End-to-End Alpha Loop (Weeks 5–6)
- Wire prompt submissions from Ratatui to worker, persist responses, and surface download/export options.
- Implement undo/redo history and preset management (genre templates, seed locking).
- Add error handling, retry logic, toast notifications for failures, and offline-friendly messaging.
- Provide CLI hooks for headless usage (`timbre-cli --generate "prompt"`).
- Ship initial documentation (`README`, `SETUP.md`, architecture overview) and quickstart demo.

### Phase 4 – Model Extensibility & Advanced Controls (Weeks 7–8)
- Add adapter interface in worker for additional backends (Riffusion ONNX, Stable Audio API). Expose model selection within TUI.
- Implement parameter panels for temperature, top-k, CFG scale (where applicable), and melody upload for conditioned generation.
- Introduce light mastering presets and stem separation (optional, via `demucs`).
- Optimize caching of model weights, support quantized checkpoints, and add warnings if resources insufficient.
- Extend testing harness to cover multi-backend runs and concurrency scenarios.

### Phase 5 – Packaging, Distribution & Telemetry (Weeks 9–10)
- Bundle Rust CLI via `cargo dist`, Python worker via `uv` self-contained environment or `pipx`.
- Provide installer scripts for macOS/Linux, document GPU setup, and optional Docker compose for full stack.
- Implement opt-in anonymous telemetry module capturing usage stats (generation count, duration).
- Harden logging, crash reporting, and config migrations. Run cross-platform QA.
- Prepare alpha release notes, tutorial walkthrough, and marketing checklist.

### Phase 6 – Post-Alpha Enhancements (Weeks 11+)
- Explore real-time streaming previews (chunked audio via WebSocket).
- Support collaborative sessions (shared state via `supabase`/`sqlite` sync or peer-to-peer).
- Investigate Rust-native inference path (Candle ONNX, direct CUDA bindings).
- Build plugin API for DAW integrations, e.g., Ableton project folder sync or Reaper extension.

## 5. API & Integration Strategy
- **Local Worker**: Default path. Employ plugin registry within Python worker to register new models with capability metadata (supports `melody`, `lyrics`, etc.). Provide CLI command to list/install additional checkpoints.
- **Hosted APIs**: Implement connectors as Rust traits (`AudioBackend`) with adapters for Stability Audio, OpenAI Audio, and Replicate. Handle job polling with exponential backoff, cost tracking, and caching of returned WAVs.
- **Scheduling**: For longer renders, use background job runner (Celery/Arq in Python or NATS JetStream). For local alpha, keep in-process queue but design interface for remote execution later.
- **Security**: Store API keys via OS keychain (macOS Keychain, `keyring` crate). Ensure logs redact secrets.

## 6. Risk Assessment & Mitigations
- **Model footprint**: Large checkpoints stress local storage/RAM. Mitigate with configurable model tiers, optional remote inference, and documentation on cleanup.
- **Latency & UX drift**: Slow generations can frustrate users. Provide progress feedback, estimated remaining time, and ability to queue multiple jobs.
- **Packaging complexity**: Two-language stack complicates setup. Address with scripted bootstrap (`uv sync`, `cargo install`), asdf instructions, and Docker fallback.
- **Audio playback parity**: Cross-platform playback may fail; implement diagnostics and offer `ffplay` fallback command.
- **Licensing & compliance**: Track model licenses in docs, embed metadata into exports, provide disclaimers regarding commercial use.
- **Maintenance load**: Keep architecture modular; adopt ADR (Architecture Decision Record) log and consistent code ownership boundaries.

## 7. Metrics & Validation
- **Generation throughput**: Time from submit → playable output (<90 s for 10 s clip on target hardware).
- **User interaction**: Ratio of prompt tweaks per session, adoption of parameter panels, number of saved presets.
- **Stability**: Successful generations vs. failures, mean time between worker crashes.
- **Quality feedback**: In-TUI thumbs up/down, optional text feedback, audio similarity scoring for prompt iterations.
- **Resource monitoring**: GPU/CPU usage snapshots logged per job to inform optimization.

## 8. Immediate Next Steps
1. Finalize architecture choice (Hybrid) and document interface contracts between Rust CLI and Python worker.
2. Stand up `uv`-managed Python environment, confirm MusicGen small inference on target hardware, record baseline metrics.
3. Scaffold Ratatui prototype illustrating layout, event loop, and mock job states to validate UX and theming.
4. Draft ADRs for backend selection, job protocol, and artifact storage path conventions.
5. Begin dependency audit (licenses, OS support) and prepare developer onboarding guide.

