# API Schemas

JSON Schema documents define the contract between the Ratatui CLI and Python worker. Keep these in sync with the Rust `cli/src/types.rs` and Python `worker/src/timbre_worker/app/models.py` definitions.

- `generation_request.json` — payload submitted by the CLI to `/generate`.
- `generation_status.json` — lifecycle updates returned by `/status/{job_id}`.
- `generation_artifact.json` — metadata describing generated audio accessible via `/artifact/{job_id}`. Includes mix data (`crossfades`, `section_rms`), orchestration payloads, and motif stem descriptors.

Update these files whenever fields are added, removed, or constraints change, and regenerate derived code or documentation as needed.
