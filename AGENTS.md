# Repository Guidelines

## Project Structure & Module Organization
- `cli/` hosts the Ratatui-based terminal client; `src/main.rs` wires the event loop, `app.rs` models shared state/events, `ui/` renders widgets, `config.rs` loads settings, and `api.rs` wraps worker calls.
- `worker/` contains the FastAPI backend under `src/timbre_worker`; `app/` defines routers and startup, `services/` owns business logic, and `tests/` mirrors service modules with `test_*.py`.
- `docs/` stores architectural notes, JSON schemas (`docs/schemas/*.json`), and ADRs; review `docs/architecture.md` before large changes. Automation hooks live in the `Makefile`, with `scripts/` reserved for future helpers.

## Build, Test, and Development Commands
- `make setup` installs Python dependencies via `uv sync --project worker` and prefetches Rust crates.
- `make cli-run` launches the TUI (`cargo run -p timbre-cli`) for manual smoke testing.
- `make worker-serve` starts the API locally at `http://localhost:8000` with autoreload.
- `make test` executes the full Rust + Python test matrix; run before pushing.
- `make lint` runs `cargo fmt --check`, `cargo clippy`, `ruff check`, and `mypy`; resolve warnings or missing fixes.

## Coding Style & Naming Conventions
- Rust targets edition 2021 with `rustfmt` (`rustfmt.toml`); keep modules snake_case, types PascalCase, and favour explicit error propagation with `anyhow::Result`.
- Python follows Ruff with 100-character lines and type hints (`pyproject.toml`); modules snake_case, classes PascalCase, async endpoints `verb_noun`.
- Format with `cargo fmt` and `uv run --project worker ruff check --fix` when safe; commit lints separately from feature work.

## Configuration & Environment
- User config defaults live at `~/.config/timbre/config.toml`; override with `TIMBRE_CONFIG_PATH` or specific keys (`TIMBRE_WORKER_URL`, `TIMBRE_DEFAULT_MODEL`, `TIMBRE_DEFAULT_DURATION`, `TIMBRE_ARTIFACT_DIR`).
- Worker settings mirror these via `pydantic-settings`, ensuring directories exist under `~/Music/Timbre` for artifacts.
- Keep secrets (API keys, tokens) out of config files; export them as env vars within shell profiles instead.

## TUI Interaction Patterns
- Run `make cli-run` to open the chat-style interface: left pane shows conversation history (timestamped), right rail lists active jobs and status logs.
- Submit prompts with `Enter`; use `↑/↓` to focus jobs; press `p` on a completed job to surface the local artifact path for manual playback.
- The CLI polls `/status`, fetches `/artifact/{job_id}`, and copies audio + metadata into `~/Music/Timbre/<job_id>/`; see `metadata.json` for prompt/model details.
- Status lines surface worker health, queue updates, and errors so issues show without leaving the TUI.

## Testing Guidelines
- Rust: add `#[cfg(test)] mod tests` alongside implementation or integration suites under `cli/tests` as the surface grows; name cases `handles_*` or `returns_*`.
- Python: place API tests in `worker/tests` and name files `test_<feature>.py`; lean on `pytest` fixtures and `httpx.AsyncClient` for request flow coverage.
- When touching request/response contracts, add schema assertions in Rust (serde round-trips) and Python (FastAPI validation).
- Manual end-to-end validation steps live in `docs/testing/e2e.md`; run them before cutting releases or after backend upgrades.

## Commit & Pull Request Guidelines
- Follow the concise, imperative log style already in history (`rust scaffolding`, `roadmap`); keep subjects ≤ 72 chars and prefix scope (`cli:`, `worker:`) when touching one side.
- Every PR should describe the change, link issues or roadmap items, note `make test`/`make lint` output, and include screenshots or terminal recordings for notable TUI updates.
- Highlight follow-up tasks and refresh `docs/architecture.md` or ADRs when behaviour or interfaces shift materially.
