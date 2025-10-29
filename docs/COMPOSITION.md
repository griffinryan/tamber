# Composition Pipeline Reference

This note captures the current state of Timbre’s composition, rendering, and mastering pipeline. It supplements `docs/architecture.md` with more detail for anyone tweaking the planner, orchestrator, or backend services.

---

## Planner v3 Snapshot

| Concept | Notes |
| --- | --- |
| **Plan Versions** | All plans emitted by the worker carry `version="v3"`. Update this doc and planner tests when bumping the version. |
| **Long-form threshold** | Requests ≥ 90 s use the long-form templates (Intro → Motif → Chorus → Outro, optional Bridge above 150 s). Each section receives ≥ 16 s of runtime after tempo quantisation. |
| **Short-form fallback** | Requests < 90 s reuse compact templates (Intro → Motif → Resolve). Useful for fast experiments and the existing unit test suite. |
| **Tempo clamp** | 68 ≤ BPM ≤ 128. The planner scales bars to hit the requested duration, then rebalances to keep per-role minimums. |
| **Theme descriptor** | `ThemeDescriptor` captures motif, instrumentation tags, rhythm label, dynamic curve, and texture. Both MusicGen and the CLI rely on it for consistent language. |
| **Orchestration layers** | Every section includes `orchestration = {rhythm, bass, harmony, lead, textures, vocals}`. Empty lists are allowed; short-form clips frequently omit vocals. |
| **Motif directives** | Each role maps to a directive, variation axes, and cadence hint (`state motif`, `amplify motif`, `resolve motif`, …) to guide backend prompt builders. |

Planner invariants are enforced in Rust (`cli/src/planner.rs`) and Python (`worker/services/planner.py`). Keep both mirrors aligned.

---

## Section Rendering

1. **Prompt assembly**
   - Template text (from planner) supplies the musical context.
   - Arrangement sentence summarises orchestration focus (`Elevate the arrangement with layered guitars and wordless vocals`).
   - Theme descriptor anchors motif, rhythm, and texture cues.
   - Previous section tails and captured motif stems feed the audio-conditioning path.
2. **Duration hints**
   - Long-form renders target `target_seconds + beat_padding` (¾ beat for edges, 1¼ beat for interior sections).
   - Short-form renders still receive the scaled seconds but never fall below 2 s.
3. **Seed offsets**
   - If users pass `seed`, MusicGen adds per-section offsets so repeated runs remain deterministic while sections still vary.
4. **Placeholder mode**
   - When dependencies are missing, services return deterministic placeholders. Extras expose `placeholder=true` and `placeholder_reason` so the CLI can warn users.

---

## Mixdown Sequence

1. **Shape to target length**
   - Ensure each render meets its planned length. Long stems are trimmed with a short fade; shorter stems are padded with silence after tapering the tail.
2. **Conditioning tails**
   - The orchestrator preserves the last few seconds of each section (up to four beats) so the next section can condition on it.
3. **Section joins**
   - Default is a “butt join” with a micro fade (~10 ms) to avoid clicks. When either section lacks audio conditioning or is a placeholder, we fall back to longer crossfades scaled by tempo.
4. **Loudness**
   - Section RMS values are stored before and after mixing; metadata surfaces both for future analytics.
5. **Master polish**
   - After concatenation, the pipeline runs: RMS normalisation → high-shelf tilt → soft tanh limiter → resample to `Settings.export_sample_rate` → write PCM (default 24-bit WAV).
6. **Motif stem export**
   - The first section flagged as motif seed is written to `<job_id>_motif.wav` with spectral centroid and chroma data. This is useful for reuse in downstream tools.

All steps are declared in code (`worker/services/orchestrator.py`) and recorded in metadata (`GenerationArtifact.metadata.extras`).

---

## Metadata Cheat Sheet

| Field | Meaning |
| --- | --- |
| `plan.sections[*].bars` | Planned bar count after tempo scaling. Useful for beat-accurate UI timelines. |
| `plan.sections[*].orchestration` | Layer assignments that drove the backend prompts. |
| `extras.sections[*].arrangement_text` | Human-readable sentence describing arrangement focus. |
| `extras.sections[*].phrase.seconds` | Effective render length before mixing (includes padding). |
| `extras.mix.crossfades[*].mode` | Either `butt` or `crossfade`. Handy for debugging mismatched transitions. |
| `extras.motif_seed` | Contains `captured`, `path`, `spectral_centroid_hz`, `chroma_vector`, and plan references. |

When adding new metadata, update `docs/schemas/`, the Rust `SectionExtras` struct, and this table.

---

## Common Development Tasks

| Task | Pointers |
| --- | --- |
| Adjust planner bar allocation | Edit `_allocate_bars` / `_build_long_form_plan` in Python and mirror changes in `cli/src/planner.rs`. Update tests under `worker/tests/test_planner.py` and `cli/src/planner.rs` (module tests at bottom). |
| Change mix behaviour | Touch `_shape_to_target_length` / `_butt_join` / `_crossfade_seconds`. Update orchestrator tests in `worker/tests/test_orchestrator.py`. |
| Add backend parameters | Extend `SectionRender.extras` in the backend service and reflect in `cli/src/ui/mod.rs` + docs/schemas. |
| Surface new CLI data | Update `extract_section_extras` in `cli/src/ui/mod.rs` and adjust status rendering tests. |

---

## Troubleshooting

- **Clips sound chopped**: Check `extras.mix.crossfades`—if modes show `butt` but conditioning was expected, inspect `audio_conditioning_applied` flags in section extras.
- **Planner tests failing**: Often due to mismatched template definitions between Python and Rust. Regenerate both sides when editing templates.
- **Placeholder audio surfacing unexpectedly**: Confirm `TIMBRE_INFERENCE_DEVICE` value and inspect section extras for `placeholder_reason`.

Keep this file updated whenever planner templates, orchestration metadata, or mix behaviour changes.
