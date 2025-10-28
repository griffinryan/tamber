# Timbre Architecture

This document describes how the current Timbre stack fits together: the Ratatui CLI, the Python worker, and the composition pipeline that sits between them. It reflects planner **v3** and orchestrator updates from October 2025.

---

## 1. Objectives

1. Provide a terminal-first workflow for exploring prompt → music ideas in seconds.
2. Keep planning, rendering, and asset management deterministic so sessions are reproducible.
3. Support “light” operation (placeholder audio, no GPU) as well as full inference pipelines (MusicGen, Riffusion) on Apple Silicon.
4. Leave space for future clients (desktop UI, web) by keeping the API boundary clean.

---

## 2. System Topology

```
┌──────────────────────────┐        HTTP/JSON polling         ┌──────────────────────────┐
│   Ratatui CLI (Rust)     │  <──────────────────────────────> │  FastAPI Worker (Python) │
│ ──────────────────────── │                                   │ ──────────────────────── │
│ tokio + reqwest client   │ Submit GenerationRequest          │ FastAPI/uvicorn          │
│ planner mirror (Rust)    │ Poll GenerationStatus             │ Composition planner v3   │
│ status + chat UI         │ Download GenerationArtifact       │ Composer orchestrator    │
│ rodio playback           │                                   │ MusicGen + Riffusion     │
└──────────────────────────┘                                   └──────────────────────────┘
             │                                                             │
             │ Local filesystem copies                                     │
             ▼                                                             ▼
    ~/.config/timbre/ (state, config)                     ~/Music/Timbre/<job_id>/ (audio + metadata)
```

Key facts:

- Transport remains simple HTTP polling (`/generate`, `/status/{job_id}`, `/artifact/{job_id}`) with JSON payloads. Schemas live in `docs/schemas/`.
- The CLI is a pure consumer; all planning happens server-side. The CLI keeps a planner mirror in Rust purely for previews.
- Worker responses always embed the `CompositionPlan` that was rendered, allowing offline inspection and deterministic replays.

---

## 3. Planner (Python `worker/services/planner.py`)

Planner v3 contains two pathways:

| Request Duration | Template Set | Characteristics |
| --- | --- | --- |
| ≥ 90 s | Long-form | Intro → Motif → Chorus → Outro (+ Bridge for ≥150 s). Tempo quantised 68–128 BPM, minimum 16 s per section, arrangement sentences generated per section. |
| < 90 s | Short-form | Compact templates (intro → motif → resolve). Keeps tests/snippets quick while still emitting orchestration metadata. |

Common behaviour:

- **ThemeDescriptor**: motif phrase, instrumentation keywords, rhythm tag, texture, dynamic curve. Derived from prompt tokens and template energies.
- **Orchestration layers**: per-section lists for rhythm, bass, harmony, lead, textures, vocals. These inform MusicGen prompts and show up in CLI UI extras.
- **Seed strategy**: base seed derived from prompt hash unless explicitly provided. Section offsets are deterministic (motif seed = base, chorus = base+1, etc.).
- **Plan versioning**: `CompositionPlan.version = "v3"`. Downstream consumers should guard on this before assuming certain fields exist.

Outputs feed both the worker orchestrator and the CLI status view through `plan.sections[*]`.

---

## 4. Orchestrator (Python `worker/services/orchestrator.py`)

The orchestrator coordinates renders and assembles the final mix:

1. **Warmup**: inspects plan sections to decide which backends to load (`musicgen`, `riffusion`). Missing backends annotate status but don’t abort.
2. **Section rendering**: for each section
   - Builds MusicGen/Riffusion prompts using the planner data and previous motif tail.
   - Passes section-specific render hints (`target_seconds`, padding driven by tempo).
   - Stores render extras (backend, conditioning info, sampling params) for CLI tooling.
3. **Motif capture**: first “motif seed” section is exported as a standalone WAV with spectral metadata.
4. **Preparation**: `_shape_to_target_length` normalises duration, trims or pads with short fades, and records per-section RMS.
5. **Mix assembly**: sections are concatenated with either butt joins (micro fades) or longer crossfades when conditioning is missing or placeholders appear. Metadata captures duration, transition mode, and conditioning flags.
6. **Mastering**: normalise to target RMS (0.2), apply a gentle high tilt, run a soft limiter, resample to `Settings.export_sample_rate` (default 48 kHz), and write PCM WAV (`export_bit_depth`).

Artifacts land in `Settings.artifact_root` (defaults to `~/Music/Timbre`). Metadata extras include:

- `mix.section_rms`, `mix.crossfades`, `mix.target_rms`, `mix.sample_rate`.
- Section extras with orchestration, prompt parameters, backend device, placeholder flags.
- `motif_seed` describing extracted motif stems.

---

## 5. Backends

### MusicGen (`worker/services/musicgen.py`)

- Uses Hugging Face transformers + `AutoProcessor`. The service lazily loads models keyed by `model_id` (`musicgen-stereo-medium` default).
- Renders up to ~29.5 s per call; long-form compositions are built from sequential renders conditioned on motif audio.
- Prompt construction combines planner template text, arrangement sentence (“Elevate the arrangement with …”), and theme hints.
- Extras include sampling hyperparameters (`top_k`, `top_p`, `temperature`, `cfg_coef`, `two_step_cfg`), arrangement text, orchestration payload, and conditioning info.
- When dependencies are missing a deterministic sine/noise placeholder is produced with matching metadata.

### Riffusion (`worker/services/riffusion.py`)

- Loads the spectrogram diffusion pipeline when allowed, otherwise returns placeholders.
- `_fallback_plan` still emits a single-section motif plan so metadata remains consistent.
- Spectrogram decoder prefers stereo and float32 on MPS to avoid artefacts.

---

## 6. CLI (Rust `cli/`)

- `AppState` holds config (`cli/src/config.rs`), active jobs, and planner previews.
- Slash commands modify `GenerationConfig` and are applied to subsequent jobs.
- Planner mirror (`cli/src/planner.rs`) mirrors Python logic so previews match worker output.
- `cli/src/ui/mod.rs` renders chat history, job list, and per-section status. Section extras from metadata (“MusicGen · render 42.1s · arrangement …”) are displayed once a job finishes.
- Artifacts are copied into the user artifact directory and enriched with CLI-only extras (`local_path`).

Error handling:

- Worker health failures raise status toasts and keep prompts in draft.
- Job failure states show detailed messages from the worker’s `GenerationStatus.message`.

---

## 7. Metadata & Schemas

| Schema | Purpose | Location |
| --- | --- | --- |
| `generation_request.json` | CLI → worker request payload (prompt, duration, model, overrides). | `docs/schemas/` |
| `generation_status.json` | Worker status poll responses. | `docs/schemas/` |
| `generation_artifact.json` | Artifact metadata contract (prompt, duration, extras, plan). | `docs/schemas/` |

Extras worth noting:

- `extras.sections[*]` – backend-level metadata (conditioning, orchestration, sampling settings).
- `extras.mix.crossfades[*]` – records transition durations and fallback modes (`butt`, `crossfade`).
- `extras.motif_seed` – describes extracted motif stems and spectral info.

Keep schemas aligned with Rust `cli/src/types.rs` and Python `worker/app/models.py`. Update both when fields change.

---

## 8. Configuration & Environment Variables

See also `docs/setup.md`.

| Setting | Defined In | Notes |
| --- | --- | --- |
| `Settings.default_duration_seconds` | Worker `app/settings.py` | Defaults to 120 but clamps between 90–180 for long-form; short-form requests allowed via API. |
| `TIMBRE_DEFAULT_DURATION` | CLI env override; values < 90 get clamped to 90. |
| `TIMBRE_RIFFUSION_ALLOW_INFERENCE` | Disables Riffusion pipeline when set to `0`. |
| `TIMBRE_INFERENCE_DEVICE` | Force `cpu`/`mps`/`cuda`. |
| `TIMBRE_EXPORT_SAMPLE_RATE` / `BIT_DEPTH` / `FORMAT` | Mastering export settings. |
| `TIMBRE_ARTIFACT_DIR` | CLI copy destination (defaults to `~/Music/Timbre`). |

The CLI caches config in `~/.config/timbre/config.toml`. The worker ensures directories exist at startup.

---

## 9. Development Flow

1. `make setup` 👉 baseline dependencies.
2. `make worker-serve` and `make cli-run` 👉 manual testing.
3. `make lint` / `make test` 👉 pre-commit checks.
4. Update docs + schemas alongside code.

Testing shortcuts:

- `cargo test` (Rust planner/UI). Expects orchestrator extras to remain stable.
- `uv run --project worker pytest` (Python planner/orchestrator/backends). Install `--extra dev` for linting, `--extra inference` for real renders.
- `scripts/riffusion_smoke.py` for quick backend validation (respects env overrides).

---

## 10. Future Considerations

- **Real-time feedback**: Graph mix RMS and crossfade data in the CLI to inform regeneration decisions.
- **Plan editing endpoints**: Expose beat-level mutations so the CLI can trim or swap sections without full replans.
- **Additional backends**: The orchestrator already tokenises model IDs; adding another backend primarily means wiring a new service with `render_section` semantics.
- **Streaming transport**: HTTP polling is fine for Phase 0. WebSocket subscriptions are the likely upgrade path once we need richer progress data or streaming audio.

Keep this document fresh whenever planner versions change, new metadata appears, or transports/backends evolve.

---

## 11. Related Documentation

For deeper technical understanding, see:

### Core Technical Guides
- **[MUSICGEN.md](MUSICGEN.md)** – Comprehensive MusicGen integration guide
  - Model architecture, prompt engineering, audio conditioning mechanics
  - Sampling parameters, 29.5s chunking strategy, placeholder fallback
- **[CONDITIONING.md](CONDITIONING.md)** – Audio conditioning deep dive
  - Why conditioning matters, motif seed + previous tail strategy
  - Impact on crossfade decisions, metadata tracking
- **[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md)** – Mixing & mastering reference
  - Section preparation, crossfade decision matrix, mastering chain
  - RMS normalization, HF tilt, soft limiter, resampling & dithering
- **[PLANNER_SYNC.md](PLANNER_SYNC.md)** – Rust/Python planner synchronization
  - Why dual planners exist, synchronization invariants
  - Testing strategy, change protocol, common pitfalls
- **[RIFFUSION.md](RIFFUSION.md)** – Riffusion backend guide
  - Spectrogram diffusion overview, Griffin-Lim phase reconstruction
  - Comparison with MusicGen, troubleshooting
- **[TUI_GUIDE.md](TUI_GUIDE.md)** – Rust TUI implementation guide
  - Architecture, event loop, rendering, HTTP client, audio playback

### Architecture Decisions
- **[ADR-001](adrs/ADR-001-hybrid-architecture.md)** – Hybrid Rust CLI + Python Worker
- **[ADR-002](adrs/ADR-002-transport-strategy.md)** – HTTP Polling Transport
- **[ADR-003](adrs/ADR-003-audio-conditioning.md)** – Audio Conditioning Strategy
- **[ADR-004](adrs/ADR-004-planner-mirroring.md)** – Dual Planner Implementation

### Support Documentation
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** – Common issues & solutions
- **[CONTRIBUTING.md](CONTRIBUTING.md)** – Developer contribution guide
- **[setup.md](setup.md)** – Detailed setup instructions
