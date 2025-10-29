# Timbre Worker

FastAPI service that plans, renders, and mixes multi-section audio for the Timbre CLI. Planner v3 handles both long-form (intro → motif → chorus → outro) and short-form requests; the orchestrator stitches MusicGen renders, normalises loudness, and exports mastered WAVs with rich metadata. See `../docs/architecture.md` and `../docs/COMPOSITION.md` for full details.

---

## Environment

From the repository root:

```bash
# Base runtime (placeholder audio when models are missing)
make setup

# Optional: tooling & tests
uv sync --project worker --extra dev

# Optional: full inference stack (MusicGen checkpoints)
uv sync --project worker --extra inference   # same as `make setup-musicgen`
```

Key env vars:

| Variable | Purpose |
| --- | --- |
| `TIMBRE_INFERENCE_DEVICE=cpu|mps|cuda` | Override device selection. |
| `TIMBRE_EXPORT_SAMPLE_RATE`, `TIMBRE_EXPORT_BIT_DEPTH` | Mastering overrides (default 48 kHz / pcm24). |

---

## Running

```bash
# from repo root
make worker-serve              # uvicorn with auto-reload (default localhost:8000)

# standalone generation (no HTTP layer)
uv run --project worker python -m timbre_worker.generate \
    --prompt "dreamy piano over rain" --duration 120
```

HTTP surface:

- `POST /generate` – enqueue a `GenerationRequest`
- `GET /status/{job_id}` – poll `GenerationStatus`
- `GET /artifact/{job_id}` – download `GenerationArtifact`
- `GET /health` – backend readiness + device metadata

Schemas live in `../docs/schemas/` and are mirrored by Rust `cli/src/types.rs`.

---

## Testing & Linting

```bash
uv run --project worker pytest         # unit/integration tests
uv run --project worker ruff check     # lint
uv run --project worker mypy           # type checking

# repo-wide convenience
make lint
make test
```

Short-form requests (duration < 90 s) keep tests fast while planner/orchestrator logic is shared with long-form paths.

---

## Project Layout

```
worker/
├─ src/timbre_worker/
│  ├─ app/            # FastAPI routers, Pydantic models, settings
│  ├─ services/
│  │  ├─ planner.py   # CompositionPlanner (v3 templates)
│  │  ├─ orchestrator.py
│  │  ├─ musicgen.py  # Hugging Face service wrapper (with placeholders)
│  │  └─ audio_utils.py
│  └─ generate.py     # CLI entrypoint for direct renders
├─ tests/             # pytest suites mirroring service modules
└─ pyproject.toml     # tooling config (ruff, mypy, extras)
```

---

For architectural and composition specifics consult `../docs/architecture.md` and `../docs/COMPOSITION.md`. Update those docs alongside code changes.
