.PHONY: setup worker-serve cli-run lint test fmt smoke

setup:
	uv sync --project worker
	cargo fetch

worker-serve:
	uv run --project worker uvicorn timbre_worker.app.main:app --port 8000

worker-serve-reload:
	uv run --project worker uvicorn timbre_worker.app.main:app --reload --port 8000

cli-run:
	cargo run -p timbre-cli

smoke:
	uv run --project worker python scripts/riffusion_smoke.py

fmt:
	cargo fmt

lint:
	cargo fmt --check
	cargo clippy -- -D warnings
	uv run --project worker ruff check
	uv run --project worker mypy

test:
	cargo test
	uv run --project worker pytest
