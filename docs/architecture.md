# Timbre Architecture

This document describes how the current Timbre stack fits together: the Ratatui CLI, the Python worker, and the composition pipeline that sits between them. It reflects planner **v3** and orchestrator updates from October 2025.

---

## 1. Objectives

1. Provide a terminal-first workflow for exploring prompt → music ideas in seconds.
2. Keep planning, rendering, and asset management deterministic so sessions are reproducible.
3. Support “light” operation (placeholder audio, no GPU) as well as full MusicGen inference on Apple Silicon.
4. Leave space for future clients (desktop UI, web) by keeping the API boundary clean.

---

## 2. System Topology

```
┌──────────────────────────┐        HTTP/JSON polling         ┌──────────────────────────┐
│   Ratatui CLI (Rust)     │  <──────────────────────────────> │  FastAPI Worker (Python) │
│ ──────────────────────── │                                   │ ──────────────────────── │
│ tokio + reqwest client   │ Submit GenerationRequest          │ FastAPI/uvicorn          │
│ planner mirror (Rust)    │ Poll GenerationStatus             │ Composition planner v3   │
│ session view + status UI │ Download GenerationArtifact       │ Composer orchestrator    │
│ rodio playback           │                                   │ MusicGen backend         │
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
- **Plan versioning**: Long-form/short-form tracks remain `"v3"`. Clip-oriented requests (via `/clip` or session APIs) emit `"v4"` single-section plans tuned for loop playback.
- **Clip planning**: `CompositionPlanner.build_clip_plan` constrains tempo/key to the session seed, focuses orchestration on the requested layer, and emits bar-perfect loop durations for the CLI Session View.

Outputs feed both the worker orchestrator and the CLI status view through `plan.sections[*]`.

---

## 4. Orchestrator (Python `worker/services/orchestrator.py`)

The orchestrator coordinates renders and assembles the final mix:

1. **Warmup**: primes the MusicGen backend based on the plan/model hint so renders can start without on-demand weight loads.
2. **Section rendering**: for each section
   - Builds MusicGen prompts using the planner data and previous motif tail.
   - Passes section-specific render hints (`target_seconds`, padding driven by tempo).
   - Stores render extras (backend, conditioning info, sampling params) for CLI tooling.
3. **Motif capture**: first “motif seed” section is exported as a standalone WAV with spectral metadata.
4. **Preparation**: `_shape_to_target_length` normalises duration, trims or pads with short fades, and records per-section RMS.
5. **Mix assembly**: sections are concatenated with either butt joins (micro fades) or longer crossfades when conditioning is missing or placeholders appear. Metadata captures duration, transition mode, and conditioning flags.
6. **Mastering**: normalise to target RMS (0.2), apply a gentle high tilt, run a soft limiter, resample to `Settings.export_sample_rate` (default 48 kHz), and write PCM WAV (`export_bit_depth`).
7. **Clip loops**: when handling `GenerationMode::CLIP`, the orchestrator trims/fades the render to an exact bar length and tags `extras.clip` with loop metadata for the Session View.

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

## 6. CLI (Rust `cli/`)

- `AppState` holds config (`cli/src/config.rs`), active jobs, planner previews, and the Session View grid. Session layouts are snapshotted to `~/.config/timbre/session.json` on exit and restored at start-up.
- Slash commands modify `GenerationConfig` and session state. In addition to `/duration`, `/model`, `/cfg`, `/seed`, the CLI exposes `/session start`, `/session status`, `/clip <layer> [prompt]`, and `/scene rename` for Ableton-style clip launching across three fixed scene columns.
- Planner mirror (`cli/src/planner.rs`) mirrors Python logic so previews match worker output (long-form v3 plans for full tracks, `build_clip_plan` v4 for session clips).
- `cli/src/ui/mod.rs` renders the Session View grid, job list, and status sidebar. Clip cells reflect layer/scene status (queued, rendering, ready, failed) and playback state; section extras from metadata populate the status panel once a job finishes.
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

---

## 10. Future Considerations

- **Real-time feedback**: Graph mix RMS and crossfade data in the CLI to inform regeneration decisions.
- **Plan editing endpoints**: Expose beat-level mutations so the CLI can trim or swap sections without full replans.
- **Additional backends**: The orchestrator already tokenises model IDs; adding another backend primarily means wiring a new service with `render_section` semantics.
- **Streaming transport**: HTTP polling is fine for Phase 0. WebSocket subscriptions are the likely upgrade path once we need richer progress data or streaming audio.

Keep this document fresh whenever planner versions change, new metadata appears, or transports/backends evolve.
