# Development Setup

## Prerequisites
- macOS 13+ on Apple Silicon (validated on M3 Pro).
- Rust toolchain (`rustup` recommended) targeting stable 1.75+.
- Python 3.11+ with [`uv`](https://github.com/astral-sh/uv) installed.
- Homebrew packages (optional but recommended):
  - `ffmpeg` for future audio conversions.
  - `pkg-config` and `libsndfile` to support `soundfile`.

## Repository Bootstrap

```bash
git clone <repo-url>
cd tamber
make setup
```

`make setup` installs Rust dependencies and synchronises the Python worker’s environment using `uv`.

## Running the Services

Open two terminals from the repository root:

```bash
# Terminal 1 – start FastAPI worker with hot reload
make worker-serve

# Terminal 2 – run Ratatui CLI prototype
make cli-run
```

With inference dependencies installed the worker loads the real Riffusion pipeline and the CLI polls for live job updates. If dependencies are missing, the worker emits deterministic placeholder audio so the flow remains testable.

To target a worker running on a different host/port, set `TIMBRE_WORKER_URL` before launching the CLI:

```bash
TIMBRE_WORKER_URL="http://localhost:9000" make cli-run
```

## Installing Inference Dependencies

When ready to use Riffusion locally, install the optional inference extras (large downloads):

```bash
cd worker
uv pip install '.[inference]'
# quick health check (prints metadata + placeholder info if deps are missing)
uv run python ../scripts/riffusion_smoke.py
```

Ensure PyTorch detects the Metal (MPS) backend where available:

```python
>>> import torch
>>> torch.backends.mps.is_available()
True
```

## Testing & Linting

```bash
# Rust checks
cargo fmt --check
cargo clippy -- -D warnings

# Python checks
cd worker
uv run ruff check src tests
uv run mypy
uv run pytest

Run a one-off generation without the HTTP API:

```bash
uv run --project worker python -m timbre_worker.generate --prompt "nostalgic synthwave skyline"
```
```

Continuous integration runs equivalent commands via `.github/workflows/ci.yml`.
