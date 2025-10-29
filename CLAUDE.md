# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Overview

Timbre is a hybrid text-to-music system with a Ratatui terminal client (Rust) and FastAPI worker (Python). The stack uses HTTP/JSON transport and targets Apple Silicon for Phase 0. Key architectural constraints include Python 3.11 pinning and deterministic planner v3 behavior across both languages.

---

## Essential Commands

```bash
# Initial setup
make setup              # uv sync + cargo fetch (runtime deps only)
make setup-musicgen     # Full inference stack (torch, transformers, etc.)

# Running services
make worker-serve       # FastAPI at localhost:8000 (auto-reload)
make cli-run            # Ratatui TUI client

# Development workflow
make lint               # cargo fmt --check, clippy, ruff, mypy
make test               # cargo test + pytest
make fmt                # cargo fmt only

# Smoke testing
make smoke              # scripts/riffusion_smoke.py

# Running individual test suites
cargo test              # Rust unit tests
uv run --project worker pytest                    # All Python tests
uv run --project worker pytest tests/test_*.py    # Single test file

# Manual Python service testing
uv run --project worker python -m timbre_worker.generate
```

**Note**: If `uv` selects an unsupported Python version, pin first:
```bash
uv python pin 3.11
```

---

## Core Architecture Patterns

### 1. Planner Mirror Pattern

The composition planner exists in **both** Python and Rust and must stay synchronized:

- **Python**: `worker/src/timbre_worker/services/planner.py`
- **Rust**: `cli/src/planner.rs`
- **Shared data**: `planner/lexicon.json` (genre profiles, templates, keywords)

**Critical invariants:**
- Constants must match: `PLAN_VERSION = "v3"`, tempo bounds (68–128), thresholds (90s long-form vs short-form), section minimums
- Template definitions must mirror exactly (long-form: Intro→Motif→Chorus→Outro with Bridge ≥150s; short-form: Intro→Motif→Resolve)
- Theme extraction and orchestration layer logic should produce identical results
- Both planners read from the same `lexicon.json` via different parsers

**When modifying the planner:**
1. Update both Python and Rust implementations together
2. Run tests on both sides: `cargo test` + `uv run --project worker pytest tests/test_planner.py`
3. Verify lexicon schema changes work with both parsers
4. Update `docs/COMPOSITION.md` if behavior changes

### 2. Type Contract Synchronization

Request/response contracts appear in three places and must stay aligned:

- **JSON Schemas**: `docs/schemas/*.json` (generation_request, generation_status, generation_artifact)
- **Rust types**: `cli/src/types.rs` (GenerationRequest, GenerationStatus, GenerationArtifact, CompositionPlan, etc.)
- **Python models**: `worker/src/timbre_worker/app/models.py` (pydantic models)

**When adding fields:**
1. Update the JSON schema first
2. Add field to Python pydantic model with proper type hints
3. Add field to Rust struct with matching serde attributes
4. Add round-trip test (serialize in Python, deserialize in Rust, verify equality)
5. Update `docs/architecture.md` if metadata contract changes

**Common gotchas:**
- Enum naming: Rust uses PascalCase variants with `#[serde(rename_all = "snake_case")]`, Python uses lowercase strings
- Optional fields need `#[serde(default, skip_serializing_if = "Option::is_none")]` in Rust
- DateTime handling: Use chrono in Rust, pydantic's `datetime` in Python, ISO8601 strings over wire

### 3. Orchestrator Pipeline Flow

The worker orchestrator (`worker/src/timbre_worker/services/orchestrator.py`) follows a strict sequence:

```
warmup() → render_composition()
  ├─ _warmup_backends()
  ├─ _render_section() (sequential, per section)
  │   ├─ backend.render() (MusicGen/Riffusion)
  │   ├─ _capture_motif_stem() (first motif section only)
  │   └─ _shape_to_target_length()
  ├─ _assemble_mix()
  │   └─ _butt_join() or _crossfade() (tempo-driven)
  └─ _master()
      ├─ RMS normalize
      ├─ high-shelf tilt
      ├─ soft limiter
      └─ resample + write PCM WAV
```

**Key decisions:**
- Section order is sequential (not parallel) to enable audio conditioning from previous section tails
- Motif capture happens during section rendering, not afterward
- Mix mode selection (`butt` vs `crossfade`) depends on conditioning presence and placeholder flags
- Mastering applies to the entire assembled mix, not individual sections

**When modifying the orchestrator:**
- Ensure section extras (`metadata.extras.sections[*]`) capture all new render parameters
- Update `docs/COMPOSITION.md` with new mix behavior
- Test both placeholder mode and full inference paths
- Verify CLI status rendering still works (`cli/src/ui/mod.rs` parses section extras)

### 4. Backend Service Pattern

Both MusicGen and Riffusion follow a consistent service interface:

```python
class Backend:
    async def warmup() -> BackendStatus
    async def render(prompt, duration, seed, **params) -> SectionRender
```

**Placeholder fallback strategy:**
- Services detect missing dependencies at `warmup()` and return `ready=False`
- `render()` calls emit deterministic sine/noise placeholders when dependencies are missing
- Metadata always includes `placeholder=True` and `placeholder_reason` when fallback is active
- CLI checks `TIMBRE_RIFFUSION_ALLOW_INFERENCE` and `TIMBRE_INFERENCE_DEVICE` env vars

**When adding a new backend:**
1. Implement the service interface in `worker/src/timbre_worker/services/`
2. Add to orchestrator's `_warmup_backends()` and `_render_section()` dispatch
3. Define placeholder behavior for missing dependencies
4. Add section extras fields to capture backend-specific parameters
5. Update Rust types if new metadata fields are required

---

## Testing Strategy

### Rust (cli/)

- Unit tests live in `#[cfg(test)]` modules at end of source files
- Integration tests under `cli/tests/` if needed
- Test naming: `handles_*`, `returns_*`, `parses_*`
- Run with `cargo test` or `cargo test <test_name>`

**Critical areas to test:**
- Planner output matches Python planner (template selection, bar allocation, orchestration)
- Slash command parsing (`/duration`, `/model`, `/cfg`, `/seed`, `/reset`)
- Type deserialization from worker JSON responses

### Python (worker/)

- Tests under `worker/tests/test_*.py`
- Use pytest fixtures, `httpx.AsyncClient` for HTTP flows
- Run with `uv run --project worker pytest`

**Critical areas to test:**
- Planner determinism (same prompt+seed → same plan)
- Orchestrator section sequencing and mix assembly
- Backend warmup and placeholder fallback paths
- Request/response validation against schemas

**Schema validation tests:**
- When touching contracts, add round-trip tests:
  1. Create Python model instance
  2. Serialize to JSON
  3. Parse with Rust serde
  4. Assert fields match expectations

---

## Configuration & Environment

### CLI Configuration

- Config file: `~/.config/timbre/config.toml`
- Override with `TIMBRE_CONFIG_PATH`
- Key env vars:
  - `TIMBRE_WORKER_URL` (default: `http://localhost:8000`)
  - `TIMBRE_DEFAULT_MODEL` (default: `musicgen-stereo-medium`)
  - `TIMBRE_DEFAULT_DURATION` (default: 120, clamped to 90–180 in UI)
  - `TIMBRE_ARTIFACT_DIR` (default: `~/Music/Timbre`)

### Worker Configuration

- Settings via `worker/src/timbre_worker/app/settings.py` (pydantic-settings)
- Key env vars:
  - `TIMBRE_RIFFUSION_ALLOW_INFERENCE` (set `0` to force placeholder mode)
  - `TIMBRE_INFERENCE_DEVICE` (`cpu`, `mps`, or `cuda`)
  - `TIMBRE_EXPORT_SAMPLE_RATE` (default: 48000)
  - `TIMBRE_EXPORT_BIT_DEPTH` (default: `pcm24`)
  - `TIMBRE_EXPORT_FORMAT` (default: `wav`)

**Important**: Python version is pinned to 3.11 in `.python-version` and `worker/pyproject.toml` (`requires-python = ">=3.11,<3.12"`). Do not upgrade Python version without testing full inference stack compatibility.

---

## Critical Implementation Details

### Planner v3 Contract

- All plans carry `version = "v3"`
- Long-form (≥90s): Intro → Motif → Chorus → Outro (+ Bridge if ≥150s)
- Short-form (<90s): Intro → Motif → Resolve
- Tempo: 68–128 BPM, quantized
- Minimum section duration: 16s (long-form), 2s (short-form)
- Every section includes `orchestration` dict with `{rhythm, bass, harmony, lead, textures, vocals}` lists

### CLI Slash Commands

Commands parsed in `cli/src/app.rs`, applied to subsequent jobs:

```
/duration <90-180>      Set target duration (clamped by UI)
/model <model_id>       Set model (musicgen-stereo-medium, etc.)
/cfg <float>            Set CFG scale (or "off" for None)
/seed <int>             Set seed (deterministic generation)
/reset                  Reset all config to defaults
```

### Metadata Extras Structure

The `metadata.extras` field captures:

```json
{
  "sections": [
    {
      "backend": "musicgen",
      "arrangement_text": "Elevate the arrangement with...",
      "orchestration": {"rhythm": [...], "bass": [...]},
      "phrase": {"seconds": 42.1},
      "prompt": "full MusicGen prompt text",
      "sampling": {"top_k": 250, "cfg_coef": 3.0},
      "placeholder": false,
      "audio_conditioning_applied": true
    }
  ],
  "mix": {
    "section_rms": [0.18, 0.22, ...],
    "crossfades": [{"from_idx": 0, "to_idx": 1, "mode": "butt", "seconds": 0.01}],
    "target_rms": 0.2,
    "sample_rate": 48000
  },
  "motif_seed": {
    "captured": true,
    "path": "path/to/motif.wav",
    "spectral_centroid_hz": 2400.5,
    "chroma_vector": [...]
  }
}
```

### Audio Conditioning

- First section (usually Intro) renders from scratch
- Subsequent sections receive the last ~4 beats of previous section audio as conditioning
- MusicGen uses this tail to maintain continuity
- When conditioning is missing or placeholder is active, mix uses longer crossfades instead of butt joins

---

## Common Development Workflows

### Adding a New Slash Command

1. Add command variant to `cli/src/app.rs` parsing logic
2. Update `GenerationConfig` or `AppState` to store the setting
3. Add field to `GenerationRequest` in `cli/src/types.rs`
4. Add matching field to Python `GenerationRequest` in `worker/app/models.py`
5. Update JSON schema in `docs/schemas/generation_request.json`
6. Handle in worker `routes.py` or `jobs.py`
7. Test round-trip: CLI sends, worker receives, plan reflects change

### Modifying Planner Templates

1. Edit `planner/lexicon.json` template definitions
2. Update Python `planner.py` if template selection logic changes
3. Mirror changes in Rust `planner.rs`
4. Run both test suites to verify consistency
5. Update `docs/COMPOSITION.md` with new template behavior
6. Bump `PLAN_VERSION` if contract breaks (requires CLI update)

### Adding New Metadata Fields

1. Define in Python `SectionRender` or `OrchestrationMetadata` dataclasses (`services/types.py`)
2. Populate during render or orchestration
3. Add to Rust `SectionExtras` struct (`cli/src/types.rs`)
4. Update JSON schema
5. Extend CLI UI rendering (`cli/src/ui/mod.rs`) to display new field
6. Add test to verify serialization/deserialization

### Debugging Audio Issues

1. Check `metadata.extras.sections[*].placeholder` flags
2. Inspect `mix.crossfades[*].mode` to see join strategy
3. Verify `audio_conditioning_applied` flags match expectations
4. Review section RMS values in `mix.section_rms`
5. Test with `scripts/riffusion_smoke.py` to isolate worker behavior
6. Use `TIMBRE_INFERENCE_DEVICE=cpu` to bypass MPS if GPU suspected

---

## File Structure Reference

```
cli/src/
  app.rs              AppState, slash command parsing, polling loop
  planner.rs          Rust planner mirror (must match Python)
  types.rs            Request/response/plan types (must match schemas)
  config.rs           Settings, TOML loading
  ui/mod.rs           Ratatui rendering, status panel, section extras display
  api.rs              HTTP client wrapper

worker/src/timbre_worker/
  app/
    main.py           FastAPI app factory, warmup event handler
    routes.py         /generate, /status, /artifact endpoints
    jobs.py           JobManager, async task orchestration
    settings.py       Pydantic settings, env var binding
    models.py         Pydantic models (must match types.rs)
  services/
    planner.py        Composition planner v3 (must match cli/planner.rs)
    orchestrator.py   Pipeline coordinator (warmup → render → mix → master)
    musicgen.py       MusicGen backend
    riffusion.py      Riffusion backend
    audio_utils.py    DSP utilities (RMS, crossfade, mastering)
    types.py          Internal dataclasses (SectionRender, etc.)

docs/
  architecture.md     System topology, contracts, flow
  COMPOSITION.md      Planner v3, orchestration, mix behavior
  schemas/            JSON schemas (generation_*, composition_plan)
  setup.md            Environment, dependencies
  testing/e2e.md      Manual validation checklist

planner/
  lexicon.json        Templates, genre profiles, keyword mappings (shared by Rust + Python)
```

---

## Documentation Maintenance

When making changes, update these files:

- **Planner changes**: `docs/COMPOSITION.md`, test files, both planner implementations
- **API contract changes**: `docs/schemas/*.json`, `docs/architecture.md`, Rust types, Python models
- **Mix/orchestration changes**: `docs/COMPOSITION.md`, orchestrator tests
- **Slash command changes**: README.md (if user-facing), AGENTS.md (if workflow changes)
- **Significant architecture shifts**: Add ADR in `docs/adrs/`

Commit messages should use scoped prefixes (`cli:`, `worker:`, `docs:`) matching existing history.
