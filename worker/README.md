# Timbre Worker

Python-based audio generation service backing the Timbre Ratatui CLI. Runs FastAPI endpoints that wrap Riffusion and future audio backends. See `../docs/architecture.md` for architectural context.

## Development Setup

From the `worker/` directory:

```bash
# install base + dev dependencies
uv sync
# install inference extras when ready to run Riffusion locally
uv pip install '.[inference]'

# quick smoke test (prints JSON payload)
uv run python ../scripts/riffusion_smoke.py
uv run uvicorn timbre_worker.app.main:app --reload
```

Generate a single clip without the HTTP layer:

```bash
uv run python -m timbre_worker.generate --prompt "intimate jazz trio in a small club"
```

## Status
- Phase 0 worker with real Riffusion inference when dependencies are installed; deterministic placeholder fallback otherwise.
