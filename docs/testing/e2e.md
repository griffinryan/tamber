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
6. Play back the WAV manually (`open <path>` on macOS). Expect placeholder audio if inference deps are missing; real audio plays once Riffusion is configured (torch + diffusers).
7. Inspect `metadata.json` in the job directory to confirm prompt/model/duration values.

## Failure Handling
- Worker offline: CLI should surface “Worker health check failed”. Verify prompt submission yields an error toast instead of hanging.
- Job failure: force an error (e.g., stop worker mid-job). Status panel should mark the job as failed with an error message.

Record findings in an ADR or issue if regressions appear. Remove artifacts under `~/Music/Timbre/` after testing to keep the environment tidy.
