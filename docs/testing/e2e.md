# End-to-End Smoke Test

Use this checklist to validate the Timbre CLI ↔ worker integration after significant changes.

## Prerequisites
- Python deps: `make setup` (installs worker environment and Rust crates).
- Install the inference stack when you want real audio (`make setup-musicgen`, or `uv sync --project worker --extra inference`).
- Launch worker: `make worker-serve` (listens on `http://localhost:8000`).
- In a separate shell, ensure the CLI sees the worker (`TIMBRE_WORKER_URL` can override the default).

## Test Flow
1. Run `make cli-run`.
2. Observe the status panel: it should show health info and default model.
3. Enter a prompt (e.g., `dreamy piano over rain`). The conversation pane logs your entry and the job appears in the Jobs panel.
4. Adjust generation settings on the fly with slash commands if desired:
   - `/duration 12` sets clip length (1–30 seconds).
   - `/model riffusion-v1` swaps the backend model id.
   - `/cfg 6.5` or `/cfg off` tunes classifier-free guidance strength.
   - `/seed 42` (or `/seed off`) locks deterministic runs; `/reset` restores defaults.
5. Watch the job transition `Queued → Running → Done`. Progress updates stream every few seconds.
6. When complete, the CLI copies outputs into `~/Music/Timbre/<job_id>/` and logs the artifact path. Press `Ctrl+P` with the job highlighted to surface the path again.
7. Play back the WAV manually (`open <path>` on macOS). Placeholder audio includes a noisy sine tone and metadata flag `placeholder=true`; real outputs require the inference extras (torch, diffusers, transformers) and will reflect the prompt more directly.
8. Inspect `metadata.json` in the job directory to confirm prompt/model/duration values.

## Composition Plan Smoke
1. Launch the worker with the dev + inference extras installed (e.g., `uv sync --project worker --extra dev --extra inference`).
2. In the CLI, submit a descriptive prompt (e.g., `wistful piano resolving to hope`). The system chat should print a plan summary (`N sections · BPM · key · flow`).
3. Focus the job and check the Status panel: it should list each section with role, energy, bars, target seconds, backend, and indicate which section is currently rendering (`▶`).
4. Watch the job status lines update to `rendering 1/N: … (riffusion)` etc., ending with `assembling mixdown` before completion.
5. After completion, open `~/Music/Timbre/<job_id>/metadata.json`:
   - Confirm `plan.sections` matches the status panel, includes backend + placeholder flags, and note that MusicGen entries now expose sampling params (`top_k`, `top_p`, `temperature`, `cfg_coef`).
   - Inspect `extras.mix.crossfades` to verify the orchestrator recorded transition durations.
6. (Optional) Replay generation with `/duration 24` and `/model musicgen-small` to validate cross-backend stitching.

## Test Automation

- Rust unit tests: `cargo test`
- Python unit tests: `uv run --project worker python -m pytest`

## Failure Handling
- Worker offline: CLI should surface “Worker health check failed”. Verify prompt submission yields an error toast instead of hanging.
- Job failure: force an error (e.g., stop worker mid-job). Status panel should mark the job as failed with an error message.

Optional: run `uv run python ../scripts/riffusion_smoke.py` from the repository root to validate the pipeline independently of the HTTP stack.

Record findings in an ADR or issue if regressions appear. Remove artifacts under `~/Music/Timbre/` after testing to keep the environment tidy.
