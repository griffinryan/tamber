# End-to-End Smoke Test

Use this checklist to validate the Timbre CLI ↔ worker integration after significant changes.

## Prerequisites
- Python deps: `make setup` (installs worker environment and Rust crates).
- Launch worker: `make worker-serve` (listens on `http://localhost:8000`).
- In a separate shell, ensure the CLI sees the worker (`TIMBRE_WORKER_URL` can override the default).

## Test Flow
1. Run `make cli-run`.
2. Observe the status panel: it should show health info and default model.
3. Enter a prompt (e.g., `dreamy piano over rain`). The conversation pane logs your entry and the job appears in the Jobs panel.
4. Watch the job transition `Queued → Running → Done`. Progress updates stream every few seconds.
5. When complete, the CLI copies outputs into `~/Music/Timbre/<job_id>/` and logs the artifact path. Press `p` with the job highlighted to surface the path again.
6. Play back the WAV manually (`open <path>` on macOS). Placeholder audio includes a noisy sine tone and metadata flag `placeholder=true`; real outputs require the `.[inference]` extras (torch, diffusers, transformers, etc.) and will reflect the prompt more directly.
7. Inspect `metadata.json` in the job directory to confirm prompt/model/duration values.

## Failure Handling
- Worker offline: CLI should surface “Worker health check failed”. Verify prompt submission yields an error toast instead of hanging.
- Job failure: force an error (e.g., stop worker mid-job). Status panel should mark the job as failed with an error message.

Optional: run `uv run python ../scripts/riffusion_smoke.py` from the repository root to validate the pipeline independently of the HTTP stack.

Record findings in an ADR or issue if regressions appear. Remove artifacts under `~/Music/Timbre/` after testing to keep the environment tidy.
