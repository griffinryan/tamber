# TIMBRE ROADMAP

## Vision
- Deliver a beautiful terminal-first chat experience that guides musicians and hobbyists through text-to-music experimentation.
- Enable iterative prompt engineering, model parameter tweaking, and direct `.wav` exports from a conversational loop.
- Ship a maintainable codebase with clear abstractions for model backends, chat state, audio post-processing, and TUI rendering.

## Guiding Principles
- **User-centric loops**: Every milestone should shorten the gap between a text idea and an audible preview.
- **Model agnosticism**: Wrap generative backends behind a consistent interface so we can swap between local checkpoints (e.g., MusicGen) and hosted APIs (e.g., Stability, OpenAI) with minimal friction.
- **Deterministic reproducibility**: Log prompts, seeds, and sampler settings to re-run or evolve previous generations.
- **Offline-first**: Assume limited network access; favor locally cached models and pre/post-processing pipelines we can run on developer hardware.
- **Composable architecture**: Separate CLI orchestration (Rust or Python) from ML inference processes so future GPU/cluster support is straightforward.

## Technical Pillars & Research Notes

### Generative Audio Backends
- **Meta Audiocraft / MusicGen**: Open-source, strong text-to-music baseline; supports 320 kbps stereo, controllable duration, style transfer (melody-conditioning). Requires GPU with ~12 GB VRAM for real-time usage; CPU inference viable but slow (~2-4x duration). Use `transformers` integration to load `facebook/musicgen-small`, `-medium`, `-large`. Quantize with `bitsandbytes` 4-bit for smaller VRAM.
- **Stability Audio (if API access)**: Text-to-music with prompt-based generation. Provides high-quality outputs but depends on external API cost and latency.
- **Riffusion v1/v1.1**: Diffusion on spectrograms; easier to run on CPU, but quality lower. Works well for ambient textures.
- **Moûsai / MusicLM derivatives**: Emerging models; keep watch but expect heavier resources.
- **OpenAI Audio API (if available)**: Potential for general audio generation, though access may be limited.
- **Pipeline strategy**: Implement a plugin architecture with shared `GenerateAudioRequest` and `GenerateAudioResult`. Start with MusicGen through `torchaudio` & `transformers`, store outputs as 32-bit float PCM before normalizing to 16-bit WAV.

### Audio Post-Processing
- Use `torchaudio` or `librosa` for resampling, loudness normalization (EBU R128 approximations), and trimming silence.
- Provide optional mastering chain (limiter + EQ preset) via `pyloudnorm` or `audiomentations`.
- Metadata embedding (`mutagen`) to store prompt info inside the WAV.

### Chat & Prompt Management
- Maintain conversation history with structured prompts (system/assistant/user), and attach generation metadata per message.
- Support prompt templates, parameter presets (BPM, instrumentation, mood tags).
- Provide undo/redo with an append-only event log. Persist state in `~/.config/timbre/history.jsonl`.

### CLI/TUI Architecture
- **Language choice**: Python for ML prototyping speed, plus `textual` (Rich TUI) for a polished chat UI. Consider Rust + `ratatui` once inference APIs stabilize, but prioritize Python to stay close to the ML ecosystem.
- **Process separation**: Run heavy generation in worker threads or subprocesses to keep the TUI responsive. Use `asyncio` with `textual`'s `work()` helper or `trio`.
- **Playback & preview**: Integrate simple audio playback (e.g., `sounddevice`, `ffplay` fallback) while keeping WAV export reliable.

### Tooling & Infrastructure
- Package management via `uv` or `poetry`. Target Python 3.10+.
- Provide Makefile tasks for `make setup`, `make lint`, `make test`, `make demo`.
- Testing: unit tests for prompt parsing, backend adapters; integration test that runs a short 2-second generation using a tiny checkpoint (`musicgen_tiny`).
- CI: GitHub Actions with CPU-only workflow; for GPU tests, integrate self-hosted runner later.

### Risk Register
- **Model weight size**: MusicGen large (~7 GB) may exceed developer resources. Mitigate by defaulting to `musicgen-small` and providing documentation for remote inference.
- **Latency**: Generation can take minutes on CPU. Plan for asynchronous UI updates and progress indicators.
- **Licensing**: Clarify data usage rights; embed metadata recording model + license.
- **Audio quality variance**: Provide guidance on prompt engineering and parameter adjustments.
- **Cross-platform audio playback**: Differences between macOS, Linux, Windows. Start with macOS/Linux support.

## Milestone Plan

### Milestone 0 – Project Foundation (Week 1)
**Goals**
- Repository scaffolding, tooling, and baseline documentation.
- Ensure contributors can run setup without manual steps.
**Key Deliverables**
- Project structure with `src/`, `timbre/` package, `tests/`, `scripts/`.
- `pyproject.toml` (Poetry/uv) with core dependencies (`transformers`, `torchaudio`, `textual`).
- `README.md` describing project goals and setup instructions.
- `CONTRIBUTING.md` + `CODE_OF_CONDUCT.md`.
- Basic `make` targets and pre-commit hooks (format, lint).
**Risks & Mitigations**
- Package conflicts: pin versions, use `uv lock`.
- Torch install complexity: provide platform-specific instructions and fallback CPU build.

### Milestone 1 – Research & Prototype Validation (Weeks 2-3)
**Goals**
- Validate candidate audio generators locally; document performance/quality trade-offs.
- Establish data structures for prompts and generation metadata.
**Key Tasks**
- Download & benchmark `musicgen-small` inference on target hardware; record RAM/VRAM usage, generation latency for 10s clip.
- Prototype CLI script (`scripts/musicgen_sample.py`) that takes text prompt and exports `out.wav`.
- Evaluate Riffusion or Stability Audio API (if credentials) as fallback; create comparative matrix (quality, speed, licensing).
- Define `GenerationRequest` schema (dataclass/pydantic) capturing prompt, seed, duration, model, temperature, top-k/p.
- Draft prompt engineering guide with best practices.
**Deliverables**
- `docs/research/audio_backends.md` summarizing findings.
- Prototype WAVs stored under `samples/`.
- Benchmarks logged in `docs/research/benchmarks.md`.

### Milestone 2 – Core CLI Chat Loop (Weeks 4-5)
**Goals**
- Implement minimal chat experience in terminal (non-TUI) that orchestrates prompts and generations asynchronously.
- Persist conversation history and outputs.
**Key Tasks**
- CLI entry point `timbre.chat` using `argparse` or `typer`.
- Conversation state manager with append-only JSONL log.
- Generation queue using `asyncio` + background worker; supports `generate`, `list`, `play`, `export`.
- WAV export pipeline encapsulated in `timbre/audio/exporter.py`.
- Add tests for state manager, serialization, and generation request building.
**Deliverables**
- Usable CLI where user can enter prompt, receive status updates, and see path to generated WAV (`./outputs/<timestamp>.wav`).
- Logging that captures prompt, seed, model, duration.
- Docs updates: quickstart for CLI usage.

### Milestone 3 – TUI Chat Experience (Weeks 6-7)
**Goals**
- Replace/basic CLI with rich TUI featuring chat bubbles, parameter panel, progress bars.
- Maintain responsive UI during long generations.
**Key Tasks**
- Build `textual` app with layout: left prompt history, right parameter inspector, bottom input panel.
- Implement progress widget tied to generation worker (time remaining estimate).
- Add inline notifications for success/failure, clickable items to trigger playback or open output folder.
- Keyboard shortcuts (e.g., `Ctrl+R` regenerate, `Ctrl+E` edit last prompt).
**Deliverables**
- Polished TUI app `python -m timbre.app`.
- Screenshots/gifs in `docs/ui/`.
- Snapshot tests for layout (Textual pilot).
- Usability feedback round with internal testers (document in `docs/research/usability.md`).

### Milestone 4 – Advanced Generation Controls (Weeks 8-9)
**Goals**
- Enhance creative control and iteration workflows.
- Introduce parameter presets, stem export, and optional conditioning.
**Key Tasks**
- Parameter presets: tempo, instrumentation, genre; allow quick toggles from TUI.
- Implement seed locking and variation mode (perturb prompt, hold seed).
- Add melody/harmonic conditioning (if using MusicGen multi-band); allow user to upload/reference MIDI or WAV for guidance.
- Post-processing chain toggles (normalization, mastering preset).
- Extend metadata embedding; allow exporting `.json` sidecar with full session info.
**Deliverables**
- Updated generation backend supporting conditioning inputs.
- Documentation for new controls and creative tips.
- Automated tests covering presets and metadata serialization.

### Milestone 5 – Polish, Packaging, and Release (Weeks 10-11)
**Goals**
- Stabilize for public alpha; provide installers and usage docs.
- Add telemetry/analytics opt-in to understand usage (if desired).
**Key Tasks**
- Error handling and retry logic (graceful fallback if model unavailable).
- Cross-platform testing (macOS, Linux). Document Windows caveats or support.
- Package distribution: `pipx install timbre-cli`, Homebrew tap or similar.
- Create tutorial video/script and embed in docs.
- Add optional plugin to integrate with DAWs (e.g., export stems to Ableton project folder).
**Deliverables**
- Versioned release notes (`CHANGELOG.md`).
- `docs/getting_started.md` with screenshots, sample prompts, troubleshooting.
- Announcement plan for alpha (blog post outline).

### Future Milestones (Post-alpha)
- Multi-user collaboration (shared sessions via WebRTC/SignalR).
- Cloud GPU inference option with queueing.
- Model fine-tuning tooling (LoRA adapters) on user-provided datasets.
- Real-time jam mode with streaming audio output.

## Workstreams & Ownership
- **ML Backend**: Researcher/ML engineer drives model evaluation, optimization, LoRA experiments.
- **Core CLI/TUI**: Systems engineer focuses on chat orchestration, UI responsiveness, packaging.
- **Audio Engineering**: Specialist handles mastering presets, normalization, metadata.
- **Documentation & DX**: Developer advocate maintains docs, tutorials, and community feedback loop.

## Metrics & Validation
- Time-to-audio (<90 seconds for 10s clip on target hardware).
- User satisfaction scores collected via post-generation prompt (thumbs up/down).
- Stability (mean time between errors > 20 generations).
- Engagement: percentage of sessions using advanced controls after Milestone 4.

## Open Questions
- GPU availability across contributors? Need shared CUDA server?
- Licensing for generated outputs—are they royalty-free? Provide legal review.
- Should we support voice/lyrics generation or strictly instrumentals?
- Offline playback dependencies—bundle `ffmpeg` or rely on system install?

## Immediate Next Steps
1. Confirm hardware targets and decide default model tier.
2. Choose package manager (`uv` vs `poetry`) and initialize repo per Milestone 0.
3. Download `musicgen-small` checkpoint and start benchmark notebook.
4. Draft architecture diagram for CLI ↔ generation worker ↔ output storage.

