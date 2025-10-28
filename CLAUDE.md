# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Timbre is a hybrid text-to-music playground consisting of:
- **Ratatui CLI (Rust)**: Terminal UI client with planner mirror, HTTP client, rodio playback
- **FastAPI Worker (Python)**: Composition planner v3, orchestrator, MusicGen + Riffusion backends
- **Target platform**: Apple Silicon for Phase 0, designed for extensibility

## Essential Commands

### Setup & Dependencies
```bash
make setup                            # Bootstrap: cargo fetch + uv sync with dev + inference extras
uv python pin 3.11                    # Pin Python 3.11 if uv chooses unsupported interpreter
uv sync --project worker --extra dev  # Install dev tools (pytest, ruff, mypy)
```

### Development Workflow
```bash
# Terminal 1 - Start worker with auto-reload
make worker-serve                     # Or: make worker-serve-reload for hot reload

# Terminal 2 - Run CLI
make cli-run                          # Connects to http://localhost:8000 by default
TIMBRE_WORKER_URL="http://HOST:8000" make cli-run  # Connect to remote worker

# Validate backend in isolation
uv run --project worker python scripts/riffusion_smoke.py --prompt "dreamy lo-fi piano"
```

### Testing
```bash
make test                             # Run both cargo test + pytest
cargo test                            # Rust unit tests only
uv run --project worker pytest        # Python tests only (needs --extra dev)
```

### Linting & Formatting
```bash
make lint                             # cargo fmt --check, clippy, ruff, mypy
make fmt                              # cargo fmt (auto-format Rust)
```

### Running Single Tests
```bash
# Rust: use standard cargo filter syntax
cargo test test_name                  # Run specific test
cargo test planner::                  # Run all tests in planner module

# Python: use pytest filter syntax
uv run --project worker pytest tests/test_planner.py::test_long_form_plan
uv run --project worker pytest -k "planner"
```

## Architecture Essentials

### Dual Planner System
The composition planner exists in **both** Rust (`cli/src/planner.rs`) and Python (`worker/src/timbre_worker/services/planner.py`). These mirrors MUST remain synchronized:

- **Constants**: `PLAN_VERSION`, tempo ranges, thresholds, role priorities, keyword mappings
- **Templates**: Section role sequences, bar allocations, energy curves
- **Logic**: Theme extraction, orchestration layer assignment, seed offsets

When modifying planner behavior:
1. Update Python planner first (`worker/services/planner.py`)
2. Mirror changes in Rust (`cli/src/planner.rs`)
3. Update tests in both: `worker/tests/test_planner.py` and Rust module tests at bottom of `planner.rs`
4. Update `docs/COMPOSITION.md` if templates or metadata changes

### Plan Versions & Thresholds
- **Current version**: `v3` (embedded in all `CompositionPlan` outputs)
- **Long-form threshold**: ≥ 90s → (Intro → Motif → Chorus → Outro + optional Bridge)
- **Short-form**: < 90s → (Intro → Motif → Resolve) for fast experiments/tests
- **Section minimums**: Long-form ≥ 16s per section, short-form ≥ 2s
- **Tempo clamping**: 68–128 BPM

### HTTP API Contract
Client-worker communication via JSON:
- `POST /generate` with `GenerationRequest` (prompt, duration, model, overrides)
- `GET /status/{job_id}` for `GenerationStatus` polling
- `GET /artifact/{job_id}` for `GenerationArtifact` download

Schemas in `docs/schemas/` must stay aligned with:
- Rust types: `cli/src/types.rs`
- Python models: `worker/app/models.py`

### Orchestration & Metadata
The orchestrator (`worker/services/orchestrator.py`) produces rich metadata:
- `plan.sections[*].orchestration`: Per-section layer assignments (rhythm, bass, harmony, lead, textures, vocals)
- `extras.sections[*]`: Backend render details, arrangement text, conditioning flags, placeholder info
- `extras.mix.crossfades[*]`: Transition modes (`butt` vs `crossfade`)
- `extras.motif_seed`: Extracted motif stems with spectral data

CLI status panel (`cli/src/ui/mod.rs`) consumes this metadata to display real-time rendering progress and post-completion section details.

## Key Configuration

Environment variables control both CLI and worker behavior:

| Variable | Scope | Purpose |
|----------|-------|---------|
| `TIMBRE_WORKER_URL` | CLI | Override worker URL (default: `http://localhost:8000`) |
| `TIMBRE_DEFAULT_MODEL` | CLI | Initial model selection (default: `musicgen-stereo-medium`) |
| `TIMBRE_DEFAULT_DURATION` | CLI | Initial duration in seconds (clamped to 90–180 for long-form) |
| `TIMBRE_ARTIFACT_DIR` | CLI | Local copy destination (default: `~/Music/Timbre`) |
| `TIMBRE_RIFFUSION_ALLOW_INFERENCE` | Worker | Set `0` to force placeholder renders |
| `TIMBRE_INFERENCE_DEVICE` | Worker | Force `cpu`, `mps`, or `cuda` |
| `TIMBRE_EXPORT_SAMPLE_RATE` | Worker | Mastering sample rate (default: 48000) |
| `TIMBRE_EXPORT_BIT_DEPTH` | Worker | PCM bit depth (default: 24) |

See `worker/src/timbre_worker/app/settings.py` and `cli/src/config.rs` for complete settings.

## Backend Services

### MusicGen (`worker/services/musicgen.py`)
- Primary backend using Hugging Face transformers
- Renders ~29.5s per call, chains longer pieces via conditioning on motif audio tails
- Prompt = planner template + arrangement sentence + theme descriptor
- Falls back to deterministic sine/noise placeholder when dependencies missing

### Riffusion (`worker/services/riffusion.py`)
- Spectrogram diffusion pipeline (optional)
- Uses `_fallback_plan` for consistent metadata when inference unavailable
- Controlled by `TIMBRE_RIFFUSION_ALLOW_INFERENCE`

## Testing Philosophy

- **Placeholder mode**: Services produce deterministic placeholders when ML dependencies are missing, keeping CLI flow testable without GPU
- **Live audio tests**: Full inference requires `--extra inference` and appropriate device
- **Metadata stability**: Tests validate plan structure, extras contracts, schema alignment
- **Orchestrator invariants**: Mix behavior (crossfades, RMS normalization, section joins) is heavily tested in `worker/tests/test_orchestrator.py`

## Common Development Patterns

### Modifying Planner Bar Allocation
1. Edit `_allocate_bars` / `_build_long_form_plan` in Python
2. Mirror in Rust `cli/src/planner.rs`
3. Run `cargo test` and `uv run --project worker pytest tests/test_planner.py`
4. Update `docs/COMPOSITION.md` table

### Adding Backend Parameters
1. Extend `SectionRender.extras` in backend service
2. Update `cli/src/types.rs` structs (`SectionExtras`, etc.)
3. Update JSON schema in `docs/schemas/`
4. Modify `extract_section_extras` in `cli/src/ui/mod.rs` for display

### Changing Mix Behavior
1. Touch `_shape_to_target_length` / `_butt_join` / `_crossfade_seconds` in orchestrator
2. Update tests in `worker/tests/test_orchestrator.py`
3. Document in `docs/COMPOSITION.md`

## Documentation Structure

- `docs/architecture.md` – System topology, transport, planner/orchestrator internals
- `docs/COMPOSITION.md` – Detailed composition pipeline reference (planner v3)
- `docs/setup.md` – Prerequisites, dependency layers, GPU considerations
- `docs/testing/e2e.md` – Manual validation checklist
- `docs/adrs/` – Architectural decision records
- `docs/schemas/` – JSON schema definitions

Keep documentation synchronized with code changes, especially when:
- Planner versions increment
- Mix behavior changes
- Slash commands are added/modified
- Configuration surface expands

## Repository Context

- **Python version**: 3.11 (strictly, not 3.12+)
- **Rust edition**: 2021
- **Workspace structure**: Cargo workspace at root, `uv` project in `worker/`
- **Branch strategy**: Main branch is `main`, use conventional commit prefixes (`cli:`, `worker:`, `docs:`)
- **License**: Proprietary R&D (review upstream model licenses before distribution)
