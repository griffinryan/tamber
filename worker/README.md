# Timbre Worker

Python-based audio generation service backing the Timbre Ratatui CLI. Runs FastAPI endpoints that wrap Riffusion and future audio backends. See `../docs/architecture.md` for architectural context.

## Development Setup

From the `worker/` directory:

```bash
# install base + dev dependencies
uv sync
# install inference extras when ready to run Riffusion locally
uv pip install '.[inference]'
uv run uvicorn timbre_worker.app.main:app --reload
```

## Status
- Phase 0 placeholder with mock endpoints.
