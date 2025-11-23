# React Native client plan

This document proposes replacing the current Swift-only `ios/` app with a React Native stack housed in `client/`, with Electron for desktop distribution and shared code for iOS/Android. The goal is to unblock client development, ship a desktop app quickly, and keep a single UI surface maintainable across platforms.

## Goals and constraints
- Ship a desktop app first (Electron host) while keeping the same code ready for iOS and Android.
- Use one codebase (TypeScript + React Native) with shared UI/state and platform shims where needed.
- Preserve worker/CLI parity: same prompt semantics, job lifecycle, and schemas.
- Keep the Makefile entrypoints simple and CI-friendly; avoid bespoke platform scripts where possible.
- Keep the existing Swift `ios/` tree temporarily for reference; remove once new client is feature-complete.

## Stack decisions (recommended)
- **Framework**: React Native (TypeScript) with **React Native Web** to target Electron.
- **Desktop shell**: **Electron** wrapping the web bundle; use `electron-builder` for packaging.
  - Alternative later: React Native macOS/Windows if we need native window chrome or lower resource use.
- **Navigation**: `@react-navigation` or Expo Router for shared routing semantics (works on web).
- **Styling**: Restyle/Dripsy/Tamagui or standard RN + StyleSheet with a token file; prefer design tokens for parity with CLI theming.
- **Data**: React Query for HTTP + caching; unified API client using worker schemas (OpenAPI/`worker` JSON schemas as the source of truth).
- **Audio**: `expo-av` (if using Expo) or `react-native-track-player` for mobile; HTML5/Howler.js or `@electron/remote`-backed audio for desktop.
- **Tooling**: yarn workspaces; TypeScript strict mode; ESLint + Prettier matching repo standards.

## Repository layout (proposed)
```
client/
  apps/
    native/      # React Native app (iOS/Android), Metro entrypoint
    desktop/     # Electron wrapper + web preload; consumes web bundle
  packages/
    ui/          # Shared RN components styled with tokens; RN + RN Web compatible
    api/         # Typed client for worker endpoints (generated from schemas)
    state/       # Hooks/state machines for prompt/job lifecycle
    config/      # Env handling, platform shims, design tokens
```

## Build and run targets (planned Makefile additions)
- `make client-setup`: install JS deps (`yarn install`), generate API types from worker schemas, prebuild assets.
- `make client-run-desktop`: start web bundle + Electron in dev (hot reload).
- `make client-run-native-ios` / `make client-run-native-android`: Metro dev server + platform target.
- `make client-test`: run unit/component tests (Jest/React Testing Library), lint, and typecheck.
- `make client-build-desktop`: produce signed dmg/zip via electron-builder (later).
- `make client-build-mobile`: produce release builds (gradlew assemble, `xcodebuild`/EAS as needed).

## Architecture outline
- **UI layer**: Shared component library (tokens, layout primitives, status rails) reused across web/native. Platform-specific implementations behind `Platform.select` for audio, filesystem, and menu controls.
- **State + data**: React Query + state machines for job submission, polling `/status`, and fetching `/artifact/{job_id}`. Retry/backoff mirrors CLI cadence.
- **Prompt parsing**: Port the CLI slash-command parser to TypeScript; enforce the same duration clamps and defaults. Consider publishing a small shared parser package for CLI + RN.
- **Configuration**: Read `TIMBRE_WORKER_URL`, `TIMBRE_DEFAULT_MODEL`, `TIMBRE_DEFAULT_DURATION`, `TIMBRE_ARTIFACT_DIR` analogs from `.env` files and runtime overrides (desktop menu/dev screen on mobile). Keep defaults aligned with `config.rs`/worker settings.
- **Offline/cache**: Cache recent prompts/jobs and downloaded artifacts per platform (desktop: app data dir via Electron; mobile: `AsyncStorage` + file system module).
- **Packaging**:
  - Desktop: RN Web bundle served locally inside Electron; preload script handles file saves and native dialogs. Distribution via electron-builder targets (macOS dmg/zip, Windows nsis, Linux AppImage).
  - Mobile: Standard RN builds; keep Expo compatibility if we start with Expo for easier onboarding.

## Migration roadmap
1) **Toolchain locked**: yarn workspaces + bare React Native (not Expo) to keep native module flexibility, with React Native Web and Electron for desktop. Use Tamagui or Restyle as the cross-platform styling layer to keep the UI expressive and performant.
2) **Scaffold `client/`**: initialize workspace, add `apps/native` and `apps/desktop` skeletons, set up TypeScript, lint, format, Jest. Add Makefile targets and CI job placeholders.
3) **API client + schemas**: generate TypeScript types from worker schemas/OpenAPI; wire a minimal `WorkerClient` (submit, status, artifact). Add contract tests against a local worker stub.
4) **Prompt parser**: port slash commands (`/duration`, `/model`, `/cfg`, `/seed`, `/motif`, `/small|/medium|/large`) with identical validation to CLI. Add tests mirroring CLI cases.
5) **Core UI + state**: implement prompt composer, job rail/status, artifact view. Use shared UI package to keep desktop/mobile identical aside from layout breakpoints.
6) **Desktop shell**: add Electron preload for filesystem access (save artifacts, open folder), custom window chrome optional. Dev flow: one command starts Metro/web + Electron with HMR.
7) **Mobile targets**: enable iOS/Android builds; ensure audio playback and file saves work with platform-safe modules. Validate simulator/emulator flows.
8) **Polish + parity**: align theme with CLI branding, add accessibility, telemetry hooks, error surfaces. Ensure worker polling cadence and schema parity.
9) **Decommission old `ios/`**: once feature parity reached, remove `ios/`, retire `make ios-run`/`make ios-test`, and update docs/CI.

## Risks and open questions
- Do we require native audio effects beyond what web/Electron can provide? If yes, we may need native modules and a bare RN setup instead of pure Expo.
- Electron bundle size and memory use vs React Native macOS/Windows alternatives; acceptable for first desktop release?
- Packaging/signing for macOS/Windows (certs, notarization) and how that fits our CI.
- Auto-update strategy for desktop (electron-updater?) and where binaries are hosted.
- Are there worker endpoints or streaming modes that require WebSocket/native audio? Plan for those early if needed.
- How much of the CLI prompt parsing can we reuse as a shared package to avoid drift?

## Immediate next steps
- Lock styling kit choice (Tamagui vs Restyle) to keep cross-platform theming crisp and performant.
- Create `client/` scaffold with yarn workspaces and a minimal “hello worker” screen hitting `/status`.
- Add Makefile targets and CI job entries marked experimental.
- Mark `docs/MOBILE.md` as superseded once RN client is underway; keep Swift project only as reference until parity.
