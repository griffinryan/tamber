# Timbre

Hybrid Ratatui (Rust) and FastAPI (Python) system for text-to-music experimentation.

## Requirements
- macOS on Apple Silicon (Phase 0 target) with Rust 1.75+ and Python 3.11+ installed.
- [`uv`](https://github.com/astral-sh/uv) for Python dependency management.
- Optional: Homebrew packages `ffmpeg`, `pkg-config`, `libsndfile` to support future audio workflows.

## Initial Setup

1. Review the direction in `ROADMAP.md`, `PHASE0_PLAN.md`, and `docs/architecture.md`.
2. Install dependencies via `make setup` (runs `uv sync --project worker` and `cargo fetch`). If `uv` selects Python 3.14, pin to a supported runtime first:

   ```bash
   uv python pin 3.12
   ```

   ```bash
   make setup
   ```

3. (Optional, large downloads) Install development + inference extras once you are ready to run the real pipelines locally:

   ```bash
   # from repo root
   uv sync --project worker --extra dev --extra inference
   # verify the backend with a one-off smoke run
   uv run --project worker python ../scripts/riffusion_smoke.py
   ```

   > MusicGen support (Audiocraft) still targets Pydantic 1 and is therefore not part of the default extras. Until we land a compatible integration, the worker will return placeholder sections for the MusicGen backend.

## Running the Prototype

Start the worker and CLI in separate terminals from the repository root:

```bash
# Terminal 1 – FastAPI worker with auto-reload
make worker-serve

# Terminal 2 – Ratatui shell
make cli-run
```

With the inference extras installed the worker runs the real Riffusion pipeline; without them it falls back to deterministic placeholder audio so the flow stays testable.

Generate a standalone clip without the HTTP API via:

```bash
uv run --project worker python -m timbre_worker.generate --prompt "dreamy piano over rain"
```

The CLI looks for the worker URL via `TIMBRE_WORKER_URL` (defaults to `http://localhost:8000`). Override when targeting remote workers:

```bash
TIMBRE_WORKER_URL="http://192.168.1.20:8000" make cli-run
```

## Documentation
- `PHASE0_PLAN.md` – execution plan for current milestone.
- `docs/setup.md` – developer environment details.
- `docs/architecture.md` – system topology and shared schemas.
- `docs/adrs/` – architecture decision records (coming soon).

## Licensing
Project status: private R&D. Review third-party model licenses before redistribution.
