# Timbre

Timbre is a hybrid text-to-music playground: a Ratatui (Rust) terminal client pairs with a FastAPI (Python) worker to plan, render, and stitch multi-section audio clips. The stack targets Apple Silicon for Phase 0, but the codebase is organised so additional clients and backends can slot in without rewriting the core flow.

---

## Repository At A Glance

```
├─ cli/                 # Ratatui application, planner mirror, HTTP client
├─ worker/              # FastAPI app, composition planner, orchestrator & backends
├─ docs/                # Architecture notes, setup guides, testing playbooks, ADRs
├─ scripts/             # Standalone helpers (reserved for future use)
├─ docs/schemas/        # JSON Schemas kept in sync with Rust/Python models
├─ Makefile             # Convenience entrypoints (setup, lint, test, run targets)
└─ ROADMAP.md, ADRs     # Directional planning material
```

---

## Quick Start

1. **Clone & bootstrap**

   ```bash
   git clone <repo-url>
   cd tamber
   make setup          # cargo fetch + uv sync (runtime deps only)
   ```

   If `uv` chooses an unsupported interpreter, pin Python 3.11 first:

   ```bash
   uv python pin 3.11
   ```

2. **Add extras (optional)**

   ```bash
   # Tooling and lint/test extras
   uv sync --project worker --extra dev

   # Full inference stack: torch, diffusers, transformers, torchaudio
   uv sync --project worker --extra inference   # same as `make setup-musicgen`
   ```

3. **Run the services**

   ```bash
   # Terminal 1 – worker with auto-reload (FastAPI + Uvicorn)
   make worker-serve

   # Terminal 2 – Ratatui CLI
   make cli-run
   ```

   The CLI defaults to `http://localhost:8000`. Point to another machine with

   ```bash
   TIMBRE_WORKER_URL="http://192.168.1.20:8000" make cli-run
   ```

## Runtime Overview

### Planner & Orchestrator

- Planner **v3** emits a `CompositionPlan` with:
  - Long-form templates (intro → motif → chorus → outro with optional bridge) above 90 s.
  - Short-form templates (intro → motif → resolve) for quick clips and unit tests.
  - Theme metadata: motif phrase, instrumentation palette, rhythm tag, dynamic curve, texture.
  - Section orchestration layers (rhythm, bass, harmony, lead, textures, vocals) that drive richer prompts and feed the CLI status panel.
- The orchestrator renders sections sequentially, captures motif seeds, trims/normalises loudness, then butt-joins sections with micro fades (longer crossfades only when conditioning is missing). Mix metadata includes per-section RMS, crossfade strategies, and mastering parameters.

### Backend

- **MusicGen** (transformers) renders text-to-music clips, conditioned by motif audio tails and arrangement sentences. When the checkpoint or dependencies are missing we emit deterministic placeholders so the CLI flow stays testable.
- Force CPU/MPS/CUDA via `TIMBRE_INFERENCE_DEVICE` when you need to override torch auto-detection.

### CLI UX Highlights

- Slash commands let you tweak generations in place:
  - `/duration 120` (range 90–180 in the UI) keeps the worker in long-form mode.
  - `/model musicgen-stereo-medium`, `/cfg 6.5`, `/seed 42`, `/reset`, etc.
- The status pane mirrors the worker planner, listing sections with roles, bar counts, target seconds, backend, placeholder flags, and active render indicator.
- Completed jobs copy exports to `~/Music/Timbre/<job_id>/`, alongside `metadata.json` (plan + mix info) and optional motif stems.

---

## Configuration Reference

| Variable | Description |
| --- | --- |
| `TIMBRE_WORKER_URL` | Override CLI → worker base URL (default `http://localhost:8000`). |
| `TIMBRE_DEFAULT_MODEL` / `TIMBRE_DEFAULT_DURATION` | CLI defaults used on startup (`musicgen-stereo-medium` / `120`). |
| `TIMBRE_ARTIFACT_DIR` | Destination for copied artifacts (CLI). |
| `TIMBRE_INFERENCE_DEVICE` | Force `cpu`, `mps`, or `cuda` regardless of auto-detection. |
| `TIMBRE_EXPORT_SAMPLE_RATE`, `TIMBRE_EXPORT_BIT_DEPTH` | Worker mastering overrides (48 kHz / pcm24 by default). |

See `worker/src/timbre_worker/app/settings.py` and `cli/src/config.rs` for the full settings surface.

---

## Testing & Tooling

```bash
# Combined check
make test            # cargo test + pytest (needs inference extras for live audio)

# Lint pass
make lint            # cargo fmt --check, cargo clippy, ruff, mypy

# Targeted commands
cargo test           # Rust unit tests
uv run --project worker pytest
uv run --project worker ruff check
uv run --project worker mypy
```

## Documentation Map

- `docs/architecture.md` – system topology, planner/orchestrator internals, metadata contracts.
- `docs/COMPOSITION.md` – in-depth composition + mixing pipeline notes (planner v3).
- `docs/setup.md` – environment prerequisites, dependency layers, optional GPU considerations.
- `docs/testing/e2e.md` – manual validation checklist for CLI ↔ worker flows.
- `docs/adrs/` – decision history.
- `docs/schemas/README.md` – JSON schema guidance.

Please keep Markdown files up to date when code changes (planner versions, mix behaviour, slash commands, configuration).

---

## Contributing

1. Branch off, run `make lint` + `make test` before pushing.
2. Update documentation and schemas alongside behavioural changes.
3. Record significant architectural shifts in a new ADR.
4. Use conventional, scoped commit subjects (`cli:`, `worker:`) to match existing history.

---

## License & Usage

The project is private R&D. Review upstream model licences (e.g., Meta MusicGen) before distributing weights or generated assets.

---

## iOS Client (experimental)

We ship a SwiftUI client scaffold under `ios/` that mirrors the CLI flow with a glassmorphic landing screen and the bundled `alagard` display font.

### Run in the Simulator

```bash
# Defaults: simulator "iPhone 15", worker at http://localhost:8000
make ios-run

# Override simulator and worker URL (e.g., when your worker runs remotely)
IOS_SIMULATOR="iPhone 15 Pro" WORKER_URL="http://192.168.1.20:8000" make ios-run

# If you see xcode-select complaining about CommandLineTools:
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

What it does:
- Boots the named simulator (ignoring “already booted” errors).
- Builds the SwiftUI app via `xcodebuild` and installs it to the simulator.
- Launches bundle id `com.timbre.mobile`.

### Tests

```bash
make ios-test
```

Runs the Xcode test action against the simulator destination in Debug. Add unit/snapshot tests under `ios/Tests/` as the app grows.

### Notes for Swift/iOS newcomers
- The app uses SwiftPM (no CocoaPods). Open `ios/TimbreMobile.xcodeproj` in Xcode if you prefer the IDE.
- `WORKER_URL` controls the worker base URL (default `http://localhost:8000`); simulator can reach your host via this address when the worker runs locally.
- The landing hero uses the bundled `public/fonts/alagard.ttf_` (copied to `ios/Resources/Fonts/alagard.ttf`) for display typography. Register new fonts by adding them to `ios/Resources/Fonts/` and `UIAppFonts` in `ios/Resources/Info.plist`.
