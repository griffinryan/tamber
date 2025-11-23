# Mobile (iOS) Plan

This document outlines how to add an iOS app (new `ios/` directory) that talks to the Timbre worker and mirrors the CLI experience with a refined, Apple-native design.

## Goals
- Deliver a SwiftUI app that can run locally with `make ios-run` and connect to a local or remote worker.
- Preserve the core flow (prompt -> job submission -> status -> artifact) while embracing iOS interaction patterns.
- Build a maintainable client that mirrors Rust/Python contracts (planner v3, orchestration metadata) and stays in lockstep with worker schemas.
- Create a visually distinctive landing screen inspired by the CLI’s spinning ASCII note, reimagined as a liquid-glass, glassmorphic, animated note.

## Design Principles
- **Liquid glass**: Layer translucent panels with system blur, rim lighting (inner/outer shadows), and soft highlights. Avoid flat whites; prefer desaturated dark bases with neon edge accents.
- **Motion with intent**: One hero animation (musical note orb) with idle drift + slow spin; brighten and add particle shimmer on generation. Gate heavy motion with Reduce Motion.
- **Typography**: SF Rounded for UI, SF Pro Display for hero labels; tight, limited type ramp tokens to avoid drift.
- **Color system**: Define tokens (base, glass, accent, success, warning, error) and gradients for hero states; accent responds to job status (pending cool, running vivid, done warm).
- **Lighting**: Specular overlays plus subtle vignette to anchor the hero; fine noise to avoid plastic feel.
- **Accessibility**: Dynamic Type, VoiceOver labels for status/prompt/artifacts, contrast-friendly variants.

## UX Direction
- **Visual language**: Layered glassmorphism (blur + soft specular highlights), subtle neon edge lighting, gradients that respond to progress, and tasteful motion (springy transitions, not constant micro-motions).
- **Landing screen**: Full-height canvas with floating, animated musical note orb (CoreAnimation/SwiftUI timeline) that subtly spins/pulses on idle and lights up during generation. Primary prompt field at center, secondary controls in a compact rail.
- **Information layout**:
  - Prompt composer with inline slash-style quick commands (`/duration`, `/model`, `/cfg`, `/seed`, `/motif`) matching CLI semantics.
  - Job rail: live queue + statuses, tap to view detail and artifact metadata.
  - Status strip surfaces worker health and last poll time (similar to CLI right rail).
- **Interactions**: Haptic taps for key actions, pull-to-refresh for queue, progress rings that mirror worker status updates. Respect accessibility (Dynamic Type, reduced motion).
- **Brand consistency**: Typography with character (e.g., SF Rounded + display accent), avoid bland defaults; color system defined via tokens to keep cross-platform parity.

## System Architecture (iOS)
- **Layers**:
  - Presentation: SwiftUI views, theming, animation assets.
  - State/Domain: `Store` or `ObservableObject` view models wrapping async worker calls; unify job state machine.
  - Data: `WorkerClient` using `URLSession` with async/await; decoders validated against worker schemas.
  - Persistence: Lightweight on-device cache for recent prompts, job history, and artifact metadata (e.g., `FileManager` + JSON or `SQLite` via GRDB). Artifacts saved under `~/Library/Application Support/Timbre/`.
  - Configuration: Base URL + defaults from `Info.plist` and `.xcconfig`, override with environment when running in CI or via a dev menu.
- **Networking**:
  - Default base URL: `http://localhost:8000` for simulator; allow custom remote URL (EC2) with HTTPS enforced in production.
  - Endpoints aligned to worker: submit job (prompt + model settings), poll `/status`, fetch `/artifact/{job_id}`, surface metadata JSON.
  - Error handling: graceful degradation on timeouts, exponential backoff for polling, offline indicator with queued retry.
  - Telemetry hooks for later: structured logs (os_log) with job_id correlation.
- **Testing**:
  - Unit tests for models/decoding and state reducers.
  - Integration tests using `URLProtocol` mocks for worker responses.
  - Snapshot tests for key screens on common device sizes; honor Reduced Motion in tests.

## Project Layout (planned `ios/`)
- `ios/App/` – Xcode project and Swift packages.
- `ios/Sources/AppMain/` – entrypoint, app state, environment setup.
- `ios/Sources/Features/Landing/` – prompt composer, animated note, quick commands.
- `ios/Sources/Features/Jobs/` – queue, details, artifact display and playback.
- `ios/Sources/Features/Settings/` – worker URL, model defaults, accessibility toggles.
- `ios/Sources/DesignSystem/` – color/typography tokens, reusable components (glass panels, progress rings, buttons, input fields).
- `ios/Sources/Networking/` – `WorkerClient`, models (Job, Artifact, Status), request builders, decoding.
- `ios/Tests/` – unit + snapshot tests; mock worker.

## Build and Run Flow (target)
- Add a `make ios-run` target that:
  - Ensures SwiftPM dependencies are resolved.
  - Builds and runs the app on a default simulator (e.g., iPhone 15) via `xcodebuild` or `xcrun simctl`.
  - Accepts an optional `WORKER_URL` env var to override the base URL for local testing.
- CI-friendly `make ios-test` for unit/snapshot suites.

### Current scaffold
- `ios/` contains a SwiftUI app skeleton (TimbreMobile.xcodeproj) with a landing screen, animated note, slash-command parser, and worker client stubs.
- `make ios-run` boots a simulator, builds, installs, and launches the app (defaults to `iPhone 15`); override with `IOS_SIMULATOR="iPhone 15 Pro"` and `WORKER_URL` for remote/local worker.
- `make ios-test` runs the Xcode test action on the simulator destination.
- Public font `alagard.ttf` is bundled and used for hero typography; add additional weights under `ios/Resources/Fonts/` and Info.plist `UIAppFonts` if needed.

## Worker Integration Details
- Requests mirror CLI settings:
  - Prompt text plus inline flags: `/duration 120`, `/model musicgen-stereo-medium`, `/cfg 6.5|off`, `/seed 42`, `/motif <prompt>`, `/small|/medium|/large`.
  - Duration clamped to 90–180s to match UI guidance; shorter clips supported for tests.
- Polling cadence mirrors CLI: periodic `/status` checks; stop when job completes/fails and fetch `/artifact/{job_id}`. Persist `metadata.json` alongside audio for later sharing.
- Schema parity:
  - Align Rust `types.rs`, worker Pydantic models, and Swift Decodable structs.
  - Add round-trip tests in Swift when schemas change; update docs/schemas to include iOS clients.
- Security:
  - Use HTTPS for remote (EC2) worker; ATS exceptions only for localhost during development.
  - Do not embed secrets; use environment/Keychain for future auth tokens.

## Implementation Milestones
1) **Scaffold**: Create `ios/` with SwiftPM + Xcode project; set up `Info.plist`, configs, base URL injection, placeholder `make ios-run`.
2) **Design system**: Define color/typography tokens, glass panels, buttons, progress rings, and an animated musical note (CA layers or Metal-based shader if needed).
3) **Networking + models**: Implement `WorkerClient`, Decodable models, error handling, retries, offline indicator, and mock URLProtocol for tests.
4) **Landing experience**: Build prompt composer with quick commands, animated note, status strip, and a primary call-to-action.
5) **Jobs + playback**: Queue list, detail view with artifact metadata, audio playback, and sharing.
6) **Settings**: Worker URL override, model defaults, diagnostics (last poll, logs).
7) **Testing + polish**: Unit/integration/snapshot tests, a11y pass, performance tuning for animations, and make targets wired for local/CI.

## Open Questions
- Confirm exact worker endpoints and payloads for mobile (parity with CLI/planner v3).
- Decide on audio playback stack (AVAudioEngine vs. AVPlayer) and background audio constraints.
- Determine whether to reuse CLI prompt parsing on-device or port parser logic to Swift to ensure identical behavior.

## Detailed Implementation Steps (sequenced)
1) **Bootstrap `ios/`**
   - Create SwiftPM-based Xcode project in `ios/App/`; include shared `xcconfigs` for Debug/Release and a `.env.example` describing `WORKER_URL`.
   - Wire `make ios-run` (default simulator, override `WORKER_URL`) and `make ios-test`.
2) **Design system tokens**
   - Add color palette (base glass, surface, accent, success, warning, error) and gradients (hero idle, hero active).
   - Define type ramp (caption/body/subheadline/title/hero) and spacing tokens.
   - Build primitives: glass card, frosted toolbar, accent button, input field with embedded slash hints, progress ring, shimmer overlay.
3) **Animated hero**
   - Implement musical note orb using SwiftUI + CA layers: slow rotation, parallax drift, particle shimmer gated by Reduce Motion; intensity responds to job status.
   - Add haptic + visual feedback on submission (brief bloom, accent pulse).
4) **Configuration + environment**
   - `AppEnvironment` with base URL (default `http://localhost:8000`), model defaults, duration bounds (90–180), feature flags (reduce motion, enable particles).
   - Dev menu for overriding worker URL and viewing diagnostics.
5) **Networking + models**
   - Define Decodable models mirroring worker schemas (job submission payload, status, artifact metadata).
   - Implement `WorkerClient` with async/await, retry/backoff, cancellation, and offline detection.
   - Add mock URLProtocol + fixtures for tests.
6) **Prompt parsing and composer**
   - Port slash command parser to Swift for parity with CLI (duration, model, cfg, seed, motif, size variants).
   - UI: primary prompt field, quick chips for common commands, inline validation (duration clamp), status strip with worker health.
7) **Job lifecycle**
   - State machine for job submission -> queued -> running -> completed/failed; periodic polling with backoff; cancellation support.
   - Queue view with cards (glass panels), status badges, and tap-through to detail.
8) **Artifact detail + playback**
   - Detail view showing metadata, prompt, model, cfg, seed; download artifact; save to app support; playback with AVAudioEngine/AVPlayer; share sheet.
   - Progress and error states; retry fetch on transient failures.
9) **Settings and diagnostics**
   - Screen for worker URL, defaults, audio settings; show last poll time, log snippets, version/build.
10) **Testing and polish**
   - Unit tests for parsing, networking, state machine; snapshot tests for landing/queue/detail; accessibility labels; performance tuning for animations; ensure Reduced Motion and Dynamic Type are honored.
