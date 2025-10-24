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
uv python pin 3.11
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

The base `make setup` target installs only the lightweight runtime so the API surface stays reachable (it will emit placeholder audio when models are missing). Layer on additional extras as you go:

```bash
# Enable linting + tests
uv sync --project worker --extra dev

# Install full inference stack (text + music generation). Equivalent to `make setup-musicgen`.
uv sync --project worker --extra inference
```

The `inference` extra provides PyTorch, Diffusers, Torchaudio, and Hugging Face tooling required for both Riffusion and MusicGen checkpoints. You can install everything (dev + inference) in one go:

```bash
uv sync --project worker --extra dev --extra inference
```

Alternatively, run the convenience target from the repo root to fetch the inference stack alongside the Rust crates:

```bash
make setup-musicgen
```

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

During automated testing or when you want to skip heavyweight model loads, disable Riffusion
inference entirely by setting `TIMBRE_RIFFUSION_ALLOW_INFERENCE=0`. The worker will emit the
deterministic placeholder audio while keeping the rest of the pipeline reachable.

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
```

Run a one-off generation without the HTTP API:

```bash
uv run --project worker python -m timbre_worker.generate --prompt "nostalgic synthwave skyline"
```
```

Continuous integration runs equivalent commands via `.github/workflows/ci.yml`.
