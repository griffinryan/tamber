# Composition Pipeline Notes

These notes track the current composition + mixing pipeline as of planner **v2**. Use them when designing the forthcoming CLI/TUI feedback loop, audio UX, or when refactoring the worker.

## 1. Planning Surface (Planner v2)
- **Tempo quantisation**: we derive seconds-per-beat from the target tempo after clamping to 60–140 BPM. Every section is allocated an integer number of beats, so section joins fall on the grid.
- **Theme descriptor**: planner v2.1 now emits a `ThemeDescriptor` (motif phrase, instrumentation palette, rhythm tag, dynamic curve, texture). This descriptor is baked into every section prompt so the whole plan shares a coherent musical vocabulary.
- **Seeded structure**: planner chooses a tonal centre by hashing the prompt/seed so intro/motif/outro sections remain deterministic for the same prompt.
- **Adaptive beats**: per-role minimum beats guardrails (intro ≥1 bar, motif ≥2 bars, etc.) ensure the short clips still contain musical arcs. Redistribution honours a priority list (motif → development → bridge …) when extending or shrinking plans.
- **Minimum clip length**: global floor of 12 s keeps at least intro/motif/outro alive by default. The CLI now defaults to 24 s so most generations surface three or more sections without manual tweaking.
- **Plan versioning**: planner reports `version="v2"`. Any downstream metadata consumers should check this before assuming bar counts or density.

### Implications for Feedback Loop
1. When the CLI previews a plan, we can cheaply show beat counts and track tempo since everything is grid-aligned.
2. Section edits (re-ordering, duration change) should work in beats – expose that in the future chat commands (`/section s01 length 8b`).
3. Metadata now carries `plan.sections[*].target_seconds` that reflect beat-aligned values; the loop should treat them as canonical when re-rendering partial mixes.

## 2. Rendering Requests
- **Render hints**: the orchestrator asks each backend for `target_seconds + tempo-aware padding` (¾ beat for edges, 1¼ beat for interior sections). Fractional seconds are now passed straight through to Riffusion’s `audio_length_in_s` to avoid early truncation.
- **Seed offsets** keep per-section variation while honouring a user-provided seed (motif seed = base, development = base+1, …). UI should display these offsets for reproducibility.

## 3. Section Conditioning & Mixdown
- **Silence trimming**: we gate each rendered waveform using a 25 ms RMS window and keep ±35/60 ms ramps. Removes leading “model silence” while preserving drum attacks.
- **Loudness alignment**: section stems are normalised to target RMS (0.18) with a reasonable gain cap. Store the pre-normalised RMS in `mix.section_rms` for telemetry or adaptive UX (e.g., flag “quiet” renders).
- **Length fitting**:
  - If the stem is long, we centre-trim to the target length.
  - If short, we loop the tail in tempo-sized slices with a quick crossfade. This fills missing beats instead of padding zeros.
- **Crossfades**: beat-aware fades (0.5–1.5 beats) using the shared `crossfade_append` helper keep transitions musical. `mix.crossfades[*].seconds` documents the actual overlap to help future timeline visualisation.
- **Master polish**: after concatenation we apply another loudness normalisation pass and a soft tanh limiter (0.98 threshold). This ensures headroom for user-side mastering while eliminating obvious clipping.
- **Theme-aware prompting + continuation**: both Riffusion and MusicGen receive the plan’s `ThemeDescriptor` plus the previous section’s render metadata and audio tail. Riffusion now re-encodes the prior section’s spectrogram and uses it as the init image, while MusicGen feeds the tail into `generate_continuation`. We still append motif/instrument/rhythm text, but the audio itself now evolves instead of rebooting each section.

### UX Hooks
- `mix` extras now include `target_rms`, `section_rms`, and exact crossfade durations. The CLI feedback loop can visualise these to highlight overly quiet sections or abrupt transitions.
- Because the mix respects beat alignment, future live preview features can jump to section boundaries by sample index = cumulative beats × `seconds_per_beat`.

## 4. Spectrogram Decoding (Riffusion)
- Decoder now assumes stereo spectrograms when channels are available, producing a wider image.
- Griffin-Lim iteration count bumped to 48 with a gentler 0.3 power curve; trade-off is slower CPU decode, but clarity improves noticeably.
- If torchaudio is unavailable we still fall back to placeholders; error surfaces in section extras (`spectrogram_decoder` field).

## 5. Default Duration & CLI Expectations
- CLI default duration = 24 s, aligning with the planner’s multi-section templates. Users can still `/duration` shorter clips, but the planner enforces the 12 s floor automatically.
- Worker default mirrors the CLI so standalone `uv run --project worker python -m timbre_worker.generate` emits structured clips by default.
- Feedback loop idea: when the user shortens a clip below 12 s, surface a warning that the plan will collapse to fewer sections (pull from `plan.sections.len()`).

## 6. Testing / Regression Safety
- Rust and Python planner tests now tolerate beat rounding by asserting duration ≥ requested − 2 s (the additional padding from tempo quantisation).
- Orchestrator tests assert the new mix metadata (`target_rms`, `section_rms`) to lock behaviour.
- Any future UX change that relies on section lengths should reuse these tests or add integration ones before shipping.

## 7. Roadmap Hooks for Final Feedback Loop
- **Plan editing API**: expose a worker endpoint that accepts beat-level mutations and leverages the `fit_to_length` utility. The CLI can emit quick “trim chorus” actions without re-planning from scratch.
- **Per-section audition**: now that stems are pre-normalised, we can write each `render.waveform` to disk for isolated playback in the CLI (press `p` on motif to audition shell loops).
- **Lyric/arrangement layer**: store an `extras.arrangement` blob parallel to the plan referencing beat offsets. As long as we keep beats integer, we can align lyric syllables or automation curves later.
- **Live meters & motif awareness**: `section_rms` + `crossfades` + the shared theme descriptor let us trend mix energy and stay motif-aware – perfect for streaming UI overlays or quick “this motif is quiet; regenerate?” prompts in the feedback loop.

Keep this document current whenever planner versions, normalization strategy, or default durations change. Future “Ableton-like” loop controls should build directly on these beat-aligned primitives.***
