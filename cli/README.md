# Timbre CLI

Terminal UI built with Ratatui that talks to the Timbre worker. It mirrors the Python planner for previews, submits jobs over HTTP, and copies finished artifacts to the local filesystem. See `../docs/architecture.md` for the broader picture.

---

## Quick Start

```bash
# from repo root
make cli-run                      # build & run against the local worker

# point at a remote worker
TIMBRE_WORKER_URL="http://192.168.1.20:8000" make cli-run

# run unit tests
cargo test

# lint (used by make lint)
cargo fmt --check
cargo clippy -- -D warnings
```

Artifacts land in `~/Music/Timbre/<job_id>/` alongside `metadata.json` and motif stems when available.

---

## Slash Commands

While the CLI is running type `/command value` in the prompt area:

| Command | Description |
| --- | --- |
| `/duration 120` | Set requested duration (UI clamps to 90–180 s; worker allows shorter via API). |
| `/model musicgen-stereo-medium` | Swap backend model. |
| `/cfg 6.5` / `/cfg off` | Adjust classifier-free guidance. |
| `/seed 42` / `/seed off` | Lock or release deterministic seeds. |
| `/reset` | Restore defaults from config/env. |
| `/motif <prompt>` | Queue a motif-only preview (~16 s) on MusicGen stereo medium. |
| `/small <prompt>` / `/medium <prompt>` / `/large <prompt>` | One-off generation using a specific MusicGen size. |
| `/session start` | Create a new worker-owned session (clears previous tempo/key and clips). |
| `/session status` | Refresh session metadata (tempo, key, clips) from the worker. |
| `/clip <layer> [prompt]` | Queue a loopable clip for the focused scene/layer (`layer` = rhythm, bass, harmony, lead, textures, vocals). |
| `/scene add [name]` / `/scene rename <name>` | Manage scenes in the Session View without leaving the TUI. |

### Session View shortcuts

- Press `i` to enter Insert mode; only then do arrow keys navigate layers/scenes.
- `Enter` in Insert mode toggles the active layer (adds an ASCII `[x]`) for clip generation.
- `Esc` returns to Normal mode so you can tab to the prompt; `Enter` there launches playback.
- `Space` stops playback for the focused layer; `x` stops every clip; `Ctrl+P` reprints the artifact path.

The left rail now includes a Motif pane that tracks session seeding. The first prompt you submit against an active session captures a ~16 s motif; subsequent prompts require an `[x]`-selected layer and render a Scene 1 clip aligned to the motif’s tempo, key, and feel.

The Session grid mirrors the worker-backed session: tracks (layers) down the side, scenes across, per-clip status metadata inside each cell, and the active section indicator (`▶`). The status sidebar continues to surface section roles, bars, durations, orchestration highlights, backend, conditioning flags, and any worker nudges.

---

## Project Layout

```
cli/
├─ src/
│  ├─ app.rs          # AppState, HTTP client, slash command handling
│  ├─ planner.rs      # Rust mirror of CompositionPlanner v3
│  ├─ ui/mod.rs       # Ratatui widgets (session grid, jobs, status, logs)
│  ├─ config.rs       # Config loading (env + TOML)
│  └─ types.rs        # Serde models mirroring worker schemas
└─ Cargo.toml
```

Configuration is cached in `~/.config/timbre/config.toml`. Override defaults with `TIMBRE_WORKER_URL`, `TIMBRE_DEFAULT_DURATION`, `TIMBRE_DEFAULT_MODEL`, or `TIMBRE_ARTIFACT_DIR`.

---

## Docs & References

- System architecture: `../docs/architecture.md`
- Planner/mix deep dive: `../docs/COMPOSITION.md`
- Manual validation flow: `../docs/testing/e2e.md`
- API schemas mirrored in Rust: `../docs/schemas/`

Keep this README and the docs in sync when planner versions, slash commands, or UI behaviour change.
