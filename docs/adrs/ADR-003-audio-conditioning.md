# ADR-003 – Audio Conditioning for Long-Form Coherence

## Status
Accepted – Phase 0

## Context
Timbre generates 90-180 second compositions, but MusicGen produces max ~29.5 seconds per inference call. We need a strategy to create coherent long-form music that maintains:
- Consistent timbre across sections
- Smooth transitions between boundaries
- Preserved thematic material (motif)
- Musical progression without repetition

Alternative approaches considered:
1. **Generate independently:** Each section uses only text prompts → incoherent, sounds like different songs
2. **Same seed across sections:** Deterministic but repetitive, no musical development
3. **Overlap & splice:** Generate overlapping chunks, crossfade aggressively → blurs details, hard to control structure
4. **Audio conditioning (chosen):** Feed reference audio to model as conditioning input

## Decision
Implement two-stream audio conditioning:

1. **Motif Seed Stream:** Capture entire first MOTIF section (~36s), use head (up to 16s) as conditioning for ALL subsequent sections
   - Preserves core musical identity
   - Maintains timbral consistency
   - Anchors thematic material

2. **Previous Tail Stream:** Extract last ~4 beats from immediately previous section, use as conditioning for current section
   - Provides immediate musical context
   - Enables smooth transitions
   - Prevents abrupt energy mismatches

Combined conditioning typically provides 8-20 seconds of reference audio per generation.

## Consequences

### Benefits
✅ **Timbral coherence:** MusicGen hears previous context, preserves instrumentation
✅ **Smooth transitions:** Model generates natural continuations
✅ **Thematic continuity:** Motif seed maintains core identity throughout
✅ **Musical development:** Different prompts per section enable progression
✅ **Butt-join optimization:** When both sections conditioned, crossfade unnecessary (preserves MusicGen's transition quality)

### Costs
⚠️ **Complexity:** More complex than naive independent generation
⚠️ **Memory overhead:** Must store motif seed and tails
⚠️ **Processing time:** Slightly longer due to conditioning encoding

### Impact on Crossfades
Sections that are both audio-conditioned use butt joins (10ms micro-fade) instead of crossfades, preserving MusicGen's carefully generated transitions.

### Metadata Tracking
All conditioning tracked in section extras:
- `audio_conditioning_requested`: Was conditioning attempted?
- `audio_conditioning_applied`: Did it succeed?
- `audio_prompt_segments`: Details of motif seed + tail
- `audio_prompt_seconds`: Total conditioning duration

### Future Considerations
- Experiment with conditioning window lengths (currently 16s motif, 4-beat tail)
- Explore multi-track conditioning (separate rhythm/bass/harmony streams)
- Investigate learned conditioning strength parameters

## Related Documents
- docs/CONDITIONING.md – Full technical deep dive
- docs/MUSICGEN.md – MusicGen integration details
- docs/AUDIO_PIPELINE.md – Crossfade decision logic
