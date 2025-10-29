.PHONY: setup setup-musicgen worker-serve cli-run lint test fmt

setup:
	uv sync --project worker --extra dev --extra inference
	cargo fetch

setup-musicgen:
	uv sync --project worker --extra dev --extra inference
	cargo fetch

worker-serve:
	uv run --project worker uvicorn timbre_worker.app.main:app --port 8000

worker-serve-reload:
	uv run --project worker uvicorn timbre_worker.app.main:app --reload --port 8000

cli-run:
	cargo run -p timbre-cli

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
