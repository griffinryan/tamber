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

`make setup` installs Rust dependencies and synchronises the Python worker’s environment using `uv`. If `uv` reports Python 3.14, pin to a supported runtime first:

```bash
uv python pin 3.12
```

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

## Installing Development & Inference Dependencies

Tests and linting rely on the `dev` extra, while real audio generation requires the heavier `inference` extra. Install both once you are ready to work on the backend end-to-end:

```bash
uv sync --project worker --extra dev --extra inference
```

This provides `pytest`, `ruff`, `mypy`, PyTorch, Diffusers, Librosa, Torchaudio, and Hugging Face tooling. MusicGen support (Audiocraft) is not bundled yet because upstream still depends on Pydantic 1; the worker will fall back to placeholder sections for the MusicGen backend until we ship a compatible integration.

Verify the environment by running the bundled smoke test (exits non-zero when the real pipeline cannot load and we fall back to placeholder audio):

```bash
uv run --project worker python ../scripts/riffusion_smoke.py
```

Ensure PyTorch detects the Metal (MPS) backend if you plan to run on GPU:

```python
>>> import torch
>>> torch.backends.mps.is_available()
True
```

If the MPS backend yields distorted or noisy audio, force the worker to fall back to CPU by exporting
`TIMBRE_INFERENCE_DEVICE=cpu` before launching `make worker-serve`. The worker now also prefers
float32 precision on MPS to avoid the white-noise artefacts seen with float16.

## Testing & Linting

```bash
# Rust checks
cargo fmt --check
cargo clippy -- -D warnings

# Python checks
cd worker
uv run ruff check src tests
uv run mypy
uv run python -m pytest

Run a one-off generation without the HTTP API:

```bash
uv run --project worker python -m timbre_worker.generate --prompt "nostalgic synthwave skyline"
```
```

Continuous integration runs equivalent commands via `.github/workflows/ci.yml`.
