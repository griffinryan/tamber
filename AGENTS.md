# Repository Guidelines

## Project Structure & Module Organization
- `cli/` hosts the Ratatui client; `src/app.rs` drives state + HTTP, `planner.rs` mirrors the Python planner, `ui/` renders widgets, and `config.rs` loads env/TOML settings. See `cli/README.md` for quick commands.
- `worker/` contains the FastAPI backend (`src/timbre_worker/app` for routers/settings, `services/` for planner v3, orchestrator, backends, and audio utils). `worker/README.md` documents common workflows.
- `docs/` holds the authoritative references: `architecture.md`, `COMPOSITION.md`, setup/testing guides, schemas, and ADRs. Automation helpers live in `scripts/`.

## Build, Test, and Development Commands
- `make setup` installs Python dependencies via `uv sync --project worker` and prefetches Rust crates.
- `make setup-musicgen` installs the full inference stack (PyTorch/Diffusers + transformers) so the MusicGen backend returns real audio.
- `make cli-run` launches the TUI (`cargo run -p timbre-cli`).
- `make worker-serve` starts the API locally at `http://localhost:8000` with auto-reload.
- `make test` runs `cargo test` and `uv run --project worker pytest`; required before pushing.
- `make lint` runs `cargo fmt --check`, `cargo clippy`, `ruff check`, and `mypy`; fix warnings instead of ignoring them.

## Coding Style & Naming Conventions
- Rust targets edition 2021 with `rustfmt` (`rustfmt.toml`); keep modules snake_case, types PascalCase, and favour explicit error propagation with `anyhow::Result`.
- Python follows Ruff with 100-character lines and type hints (`pyproject.toml`); modules snake_case, classes PascalCase, async endpoints `verb_noun`.
- Format with `cargo fmt` and `uv run --project worker ruff check --fix` when safe; commit lints separately from feature work.

## Configuration & Environment
- User config defaults live at `~/.config/timbre/config.toml`; override with `TIMBRE_CONFIG_PATH` or individual keys (`TIMBRE_WORKER_URL`, `TIMBRE_DEFAULT_MODEL`, `TIMBRE_DEFAULT_DURATION`, `TIMBRE_ARTIFACT_DIR`).
- Worker settings mirror these via `pydantic-settings`. Relevant overrides: `TIMBRE_INFERENCE_DEVICE`, `TIMBRE_EXPORT_SAMPLE_RATE`, `TIMBRE_EXPORT_BIT_DEPTH`, default model IDs, etc. Planner v3 now emits orchestration metadata—keep Python + Rust planners in sync when editing templates.
- Never commit secrets (API keys, tokens). Export them via environment variables or shell profiles.

## TUI Interaction Patterns
- Run `make cli-run` to open the chat-style interface. Left pane = conversation history; right rail = job list + status log.
- Submit prompts with `Enter`; use `↑/↓` to change focus; press `Ctrl+P` on a completed job to print the artifact path.
- Slash commands: `/duration 120` (UI clamps 90–180 s), `/model musicgen-stereo-medium`, `/cfg 6.5` or `/cfg off`, `/seed 42`, `/reset`. Inline prompts: `/motif <prompt>` for motif-only previews, `/small|/medium|/large <prompt>` to pick MusicGen sizes. Short clips (< 90 s) remain available via the worker API for tests.
- The CLI polls `/status`, fetches `/artifact/{job_id}`, and copies audio + metadata into `~/Music/Timbre/<job_id>/`; see `metadata.json` for prompt/model details.
- Status lines surface worker health, queue updates, and errors so issues show without leaving the TUI.

## Testing Guidelines
- Rust: add `#[cfg(test)]` modules near implementations (or under `cli/tests`) and name cases `handles_*` / `returns_*`.
- Python: place tests in `worker/tests/test_*.py`, prefer pytest fixtures + `httpx.AsyncClient` for HTTP flows.
- When touching request/response contracts, update JSON schemas, Rust `types.rs`, and worker Pydantic models together. Add round-trip tests on both sides.
- Manual E2E checklist lives in `docs/testing/e2e.md`; run it before releases or after backend upgrades.

## Commit & Pull Request Guidelines
- Follow the concise, imperative log style already in history (`rust scaffolding`, `roadmap`); keep subjects ≤ 72 chars and prefix scope (`cli:`, `worker:`) when touching one side.
- Every PR should describe the change, link issues or roadmap items, note `make test`/`make lint` output, and include screenshots or terminal recordings for notable TUI updates.
- Highlight follow-up tasks and refresh `docs/architecture.md` or ADRs when behaviour or interfaces shift materially.
