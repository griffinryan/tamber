# Audio Conditioning: The Key to Long-Form Coherence

This document provides a comprehensive deep dive into Timbre's audio conditioning system—the critical innovation that enables coherent 90-180 second compositions from a model that generates ~29 seconds at a time.

---

## 1. The Problem

### Challenge: Long-Form Music Generation
**MusicGen Limitation:** Generates max ~29.5 seconds per inference call

**Timbre Goal:** Create 90-180 second compositions with:
- Consistent timbre across sections
- Smooth transitions between sections
- Maintained motif/thematic material
- Coherent musical progression

**Naive Approach (doesn't work):**
```python
# BAD: Generate sections independently
intro = musicgen.generate("intro music", duration=20)
motif = musicgen.generate("main theme", duration=30)
chorus = musicgen.generate("energetic chorus", duration=40)

# Result: Each section sounds like a different song!
# - Different instruments
# - Different tempo feel
# - Disconnected transitions
# - No thematic continuity
```

**Problem Analysis:**
1. **Timbral drift:** Each generation samples randomly → different instrument textures
2. **Energy mismatch:** No knowledge of previous section → abrupt transitions
3. **Thematic incoherence:** Motif not preserved across sections
4. **Seed limitations:** Same seed across sections → identical output, not musical progression

---

## 2. The Solution: Audio Conditioning

### Core Concept
**Feed reference audio to the model as conditioning input.**

MusicGen supports "audio prompting" — you can provide audio that the model will continue/develop.

```python
# GOOD: Generate with audio conditioning
intro = musicgen.generate("intro music", duration=20)

motif = musicgen.generate(
    "main theme",
    duration=30,
    audio_prompt=intro[-4s:]  # Last 4 seconds of intro
)

chorus = musicgen.generate(
    "energetic chorus",
    duration=40,
    audio_prompt=np.concatenate([
        intro[:6s],      # Motif seed
        motif[-4s:]      # Previous tail
    ])
)

# Result: Coherent composition with smooth transitions!
```

**Why it works:**
- Model "hears" the previous context
- Preserves timbre from reference audio
- Enables smooth continuation
- Maintains motif across sections

---

## 3. Timbre's Two-Stream Conditioning Strategy

Timbre uses TWO types of conditioning audio for maximum coherence:

### Stream 1: Motif Seed (Thematic Identity)
**Purpose:** Capture and maintain the core musical identity

**Source:** Entire first MOTIF section (typically 18-20 bars, 36-40 seconds)

**Usage:** Added to audio prompts for ALL subsequent sections

**Extraction:**
```python
# In orchestrator loop
for index, section in enumerate(plan.sections):
    render = await backend.render_section(section, motif_seed=motif_seed_render, ...)

    # Capture motif seed (first MOTIF section)
    if motif_seed_render is None and section.role == MOTIF:
        motif_seed_render = render  # Store entire render
        motif_seed_section = section
```

**What gets extracted:**
```python
def _prepare_audio_prompt_segment(motif_seed):
    # Extract head (first portion)
    motif_window = min(16.0, phrase_window_seconds(motif_seed))
    motif_audio = motif_seed.waveform[:motif_window_samples]

    # Convert to mono for conditioning
    if waveform.ndim > 1:
        mono = waveform.mean(axis=1)

    # Apply fade at end
    fade_samples = min(256, mono.size // 24)
    mono[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    return mono  # Up to 16s of motif audio
```

**Properties:**
- **Length:** Up to 16 seconds (enough for full musical statement)
- **Position:** HEAD of waveform (captures intro → development)
- **Reuse:** Same motif seed used for ALL sections after MOTIF

### Stream 2: Previous Tail (Immediate Context)
**Purpose:** Provide immediate musical context for smooth transitions

**Source:** Last ~4 beats from the immediately previous section

**Usage:** Added to current section's audio prompt

**Extraction:**
```python
def _conditioning_tail(render, phrase):
    # Calculate tail length (typically 4 beats)
    tail_seconds = min(
        phrase.seconds,  # Max available
        phrase.seconds_per_beat * 4.0  # Target 4 beats
    )
    tail_seconds = min(tail_seconds, 16.0)  # Never exceed 16s

    tail_samples = int(tail_seconds * sample_rate)

    # Extract last portion
    tail = waveform[-tail_samples:].copy()

    # Apply fade-in to smooth start
    fade_samples = min(512, tail.shape[0] // 32)
    fade = np.linspace(0.0, 1.0, fade_samples)
    tail[:fade_samples] *= fade

    return tail
```

**Properties:**
- **Length:** ~4 beats (e.g., 2.0s at 120 BPM, 3.0s at 80 BPM)
- **Position:** TAIL of waveform (captures immediate transition point)
- **Refresh:** Updated for each section

### Combined Conditioning
```python
def _build_audio_prompt(motif_seed, previous_render):
    segments = []

    # Add motif seed (if available)
    if motif_seed:
        motif_head = prepare_segment(motif_seed, max_seconds=16.0, segment="head")
        segments.append(motif_head)
        # Typical: 6-16 seconds

    # Add previous tail (if available)
    if previous_render:
        prev_tail = prepare_segment(previous_render, max_seconds=4.0, segment="tail")
        segments.append(prev_tail)
        # Typical: 2-4 seconds

    # Concatenate
    combined = np.concatenate(segments)

    # Convert to stereo for MusicGen
    stereo = np.stack([combined, combined], axis=0)  # (2, samples)

    return stereo, {
        "audio_conditioning_applied": True,
        "audio_prompt_seconds": len(combined) / sample_rate,
        "audio_prompt_segments": [
            {"source": "motif", "seconds": 6.2, ...},
            {"source": "previous_tail", "seconds": 2.1, ...},
        ]
    }
```

**Total Conditioning:** Typically 8-20 seconds of reference audio

---

## 4. Section-by-Section Conditioning Flow

### Example: 4-Section Composition

**Section 0: INTRO (no conditioning)**
```python
intro_render = musicgen.generate(
    "Establish the world with warm piano...",
    duration=24s,
    audio_prompt=None,  # No conditioning
)
```
- **Motif seed:** Not captured (not a MOTIF section)
- **Previous tail:** N/A (first section)

**Section 1: MOTIF (no conditioning, but captured as seed)**
```python
motif_render = musicgen.generate(
    "Present the dreamy lo-fi piano motif clearly...",
    duration=36s,
    audio_prompt=intro_tail,  # Last 2s of intro
)

# CAPTURE AS MOTIF SEED
motif_seed = motif_render  # Store entire render
```
- **Motif seed:** CAPTURED (first MOTIF section) ⭐
- **Previous tail:** intro[-2s:]

**Section 2: CHORUS (fully conditioned)**
```python
chorus_render = musicgen.generate(
    "Amplify the motif into anthemic chorus...",
    duration=48s,
    audio_prompt=concatenate([
        motif_seed[:16s],  # Motif head
        motif_tail[-2s:],  # Previous section tail
    ])
    # Total: 18s conditioning audio
)
```
- **Motif seed:** motif_seed[:16s] (thematic identity)
- **Previous tail:** motif[-2s:] (transition context)

**Section 3: OUTRO (fully conditioned)**
```python
outro_render = musicgen.generate(
    "Close by reshaping the motif...",
    duration=24s,
    audio_prompt=concatenate([
        motif_seed[:16s],   # Same motif seed!
        chorus_tail[-2s:],  # Previous section tail
    ])
)
```
- **Motif seed:** motif_seed[:16s] (maintained throughout)
- **Previous tail:** chorus[-2s:] (fresh transition context)

### Conditioning Timeline Visualization
```
INTRO (24s)
  [████████████████████████]
                      ↓↓ (tail)
MOTIF (36s) ⭐ SEED CAPTURED
  [███████████████████████████████████]
   ↑↑↑↑↑↑ (motif head)           ↓↓ (tail)

CHORUS (48s)
  Conditioning: [motif_head + motif_tail]
  [████████████████████████████████████████████████]
                                              ↓↓ (tail)

OUTRO (24s)
  Conditioning: [motif_head + chorus_tail]
  [████████████████████████]
```

---

## 5. Implementation Details

### Model-Level Integration
```python
# MusicGen model.generate() signature
audio_tokens = model.generate(
    input_ids=text_inputs,                    # Text prompt
    audio_prompt=conditioning_audio,          # ← Conditioning audio!
    audio_prompt_attention_mask=attention_mask,
    max_new_tokens=tokens,
    temperature=1.0,
    guidance_scale=3.0,
    generator=torch.Generator().manual_seed(seed),
)
```

**Preprocessing:**
```python
# Prepare audio conditioning
audio_features = processor(
    audio=[audio_prompt],           # (2, samples) stereo
    sampling_rate=32000,            # MusicGen native rate
    return_tensors="pt",
)

# Extract features
audio_values = audio_features["audio_values"]           # Encoded audio
attention_mask = audio_features["audio_attention_mask"] # Valid regions

# Pass to model
conditioning_args = {
    "audio_prompt": audio_values.to(device),
    "audio_prompt_attention_mask": attention_mask.to(device),
}
```

### How MusicGen Uses Conditioning
**Internally (simplified):**
1. Audio prompt encoded via EnCodec → latent tokens
2. Tokens prepended to generation context
3. Model attention mechanism sees both:
   - Text prompt embeddings
   - Audio prompt token history
4. Generation continues from audio prompt's "endpoint"

**Result:** Output sounds like a natural continuation of the conditioning audio

### Error Handling
```python
# Conditioning may fail (unsupported model, encoding error)
try:
    audio_features = processor(audio=[audio_prompt], ...)
    conditioning_args["audio_prompt"] = audio_features["audio_values"]
except Exception as exc:
    logger.warning(f"Audio conditioning failed: {exc}")
    extras["audio_conditioning_applied"] = False
    extras["audio_conditioning_error"] = str(exc)
    conditioning_args.clear()  # Fall back to text-only
```

---

## 6. Impact on Crossfade Decisions

Audio conditioning directly influences how sections are joined.

### The Crossfade Decision Matrix

```python
def _crossfade_seconds(left_extras, right_extras):
    left_conditioned = left_extras.get("audio_conditioning_applied")
    right_conditioned = right_extras.get("audio_conditioning_applied")

    # Both conditioned → Trust MusicGen's continuation
    if left_conditioned and right_conditioned:
        return 0.0  # BUTT JOIN (10ms micro-fade only)

    # One or neither conditioned → Blend to smooth mismatch
    else:
        return 0.15  # SHORT CROSSFADE
```

### Why Butt Join When Conditioned?
**Key Insight:** When the right section was generated WITH the left section's tail as conditioning, MusicGen already created a smooth transition!

**Crossfading would:**
- ❌ Blur the carefully crafted transition
- ❌ Reduce clarity/impact
- ❌ Waste MusicGen's conditioning work

**Butt joining:**
- ✅ Preserves MusicGen's transition quality
- ✅ Maintains full dynamic range
- ✅ Honors the model's continuation logic

**Micro-fade (10ms):** Only to prevent digital clicks at sample boundaries

### Crossfade Examples

**Scenario 1: Both Conditioned**
```
Section A generated standalone
Section B generated WITH conditioning = A[-tail:]

A: [████████████████▓]
                     ↓ (MusicGen saw this tail)
B:                  [▓████████████████]
                     ↑ (Continuation starts here)

Join: [████████████████████████████████]
       No crossfade! MusicGen handled it.
```

**Scenario 2: Right Not Conditioned**
```
Section A generated with conditioning
Section B generated standalone (no audio prompt)

A: [████████████████]
B:                  [████████████████]
                     ↑ (No knowledge of A)

Join: [███████████████▒▒▒█████████████]
                      ↑↑↑ 0.15s crossfade to blend
```

---

## 7. Metadata Tracking

Every aspect of conditioning is meticulously tracked for debugging and analysis.

### Per-Section Tracking
```python
extras = {
    # Conditioning status
    "audio_conditioning_requested": True,   # Was conditioning attempted?
    "audio_conditioning_applied": True,     # Did it succeed?

    # Conditioning details
    "audio_prompt_seconds": 10.2,           # Total conditioning audio duration
    "audio_prompt_segments": [
        {
            "source": "motif",              # Segment type
            "seconds": 6.2,                 # Segment duration
            "window_seconds": 16.0,         # Max window size
        },
        {
            "source": "previous_tail",
            "seconds": 4.0,
            "window_seconds": 4.0,
        }
    ],
    "audio_prompt_channels": 2,             # Stereo

    # Optional: motif seed reference
    "motif_seed_section": "s01",            # Which section provided motif seed

    # Error tracking (if conditioning failed)
    "audio_conditioning_error": "EncodingError: ...",
}
```

### Mix-Level Tracking
```python
extras["mix"]["crossfades"] = [
    {
        "from": "s01",
        "to": "s02",
        "seconds": 0.0,
        "mode": "butt",
        "conditioning": {
            "left_audio_conditioned": True,   # Enabled butt join
            "right_audio_conditioned": True,
            "left_placeholder": False,
            "right_placeholder": False,
        }
    },
    ...
]
```

### Motif Seed Metadata
```python
extras["motif_seed"] = {
    "captured": True,
    "path": "/path/to/job_abc123_motif.wav",
    "section_id": "s01",
    "section_label": "Motif",
    "section_role": "MOTIF",

    # Spectral analysis
    "spectral_centroid_hz": 2450.5,
    "chroma_vector": [0.12, 0.08, ...],     # 12-bin pitch distribution
    "dominant_pitch_class": "G",
    "plan_key_alignment": 0.15,

    # Theme reference
    "motif_text": "dreamy lo-fi piano",
    "motif_rhythm": "downtempo pulse",
}
```

---

## 8. Debugging Conditioning Issues

### Issue: Sections Sound Disconnected
**Symptoms:** Each section has different timbre, instruments, or energy

**Diagnosis:**
```python
# Check if conditioning was applied
for section in extras["sections"]:
    print(f"{section['section_id']}: {section.get('audio_conditioning_applied')}")

# Expected output (after s01):
# s00: False  (intro, no conditioning available)
# s01: False or True  (motif, may have intro tail)
# s02: True  (should have motif seed + s01 tail)
# s03: True  (should have motif seed + s02 tail)
```

**Common Causes:**
1. **Motif seed not captured:**
   - Check `extras["motif_seed"]["captured"]`
   - Verify MOTIF section has directive "state motif"

2. **Conditioning failed silently:**
   - Check for `audio_conditioning_error` in section extras
   - Verify processor can encode audio format

3. **Placeholder audio:**
   - If `placeholder: True`, conditioning skipped
   - Install inference dependencies

**Solutions:**
```python
# Verify motif seed capture
if extras.get("motif_seed", {}).get("captured") != True:
    # Check planner directives
    motif_sections = [s for s in plan.sections if s.role == "MOTIF"]
    for section in motif_sections:
        print(f"Directive: {section.motif_directive}")
    # Should be "state motif" for at least one MOTIF section

# Verify conditioning chain
prev_tail = None
for index, section in enumerate(sections):
    has_motif = motif_seed is not None
    has_prev = prev_tail is not None
    print(f"s{index:02d}: motif={has_motif}, prev={has_prev}")
```

### Issue: Harsh Transitions Despite Conditioning
**Symptoms:** Clicks, pops, or abrupt changes between sections

**Diagnosis:**
```python
# Check crossfade modes
for xfade in extras["mix"]["crossfades"]:
    if xfade["mode"] == "butt" and xfade["seconds"] == 0.0:
        # Should only happen when both conditioned
        cond = xfade["conditioning"]
        if not (cond["left_audio_conditioned"] and cond["right_audio_conditioned"]):
            print(f"⚠️ Butt join without full conditioning: {xfade['from']}→{xfade['to']}")
```

**Common Causes:**
1. **Crossfade logic error:** Butt joining when shouldn't
2. **Insufficient micro-fade:** 10ms may not be enough for all material
3. **Phase cancellation:** Rare, but possible with certain audio

**Solutions:**
```python
# Force minimum crossfade
def _crossfade_seconds(...):
    # ...existing logic...

    # Always apply at least 50ms fade
    if fade_seconds == 0.0:
        fade_seconds = 0.05  # 50ms minimum

    return fade_seconds
```

### Issue: Motif Not Preserved Across Sections
**Symptoms:** Thematic material changes or disappears

**Diagnosis:**
```python
# Check if motif seed being used
for section in extras["sections"][2:]:  # After motif captured
    segments = section.get("audio_prompt_segments", [])
    has_motif = any(seg["source"] == "motif" for seg in segments)
    print(f"{section['section_id']}: motif_in_prompt={has_motif}")

# All should be True after motif section
```

**Common Causes:**
1. **Motif seed too short:** < 4s may not capture full theme
2. **CFG scale too low:** Model ignoring conditioning (increase to 5-7)
3. **Prompt contradicts motif:** "piano motif" → "guitar solo"

**Solutions:**
1. Increase motif seed window: `min(16.0, ...)` → `min(24.0, ...)`
2. Boost CFG: `guidance_scale=5.0`
3. Align prompts with theme descriptor

---

## 9. Best Practices

### For Maximum Conditioning Benefit
✅ **DO:**
- Ensure MOTIF section is substantial (≥16s, ≥8 bars)
- Use consistent instrumentation in prompts
- Maintain theme descriptor across sections
- Trust butt joins when both sections conditioned

❌ **DON'T:**
- Skip the MOTIF section (no seed to capture)
- Dramatically change instrumentation mid-composition
- Override conditioning with contradictory prompts
- Force crossfades when conditioning worked well

### Prompt Engineering with Conditioning
**Good approach:**
```python
# Section prompts build on each other
intro: "Establish with warm piano and atmospheric textures..."
motif: "State the dreamy piano motif clearly..."
chorus: "Amplify the motif into anthemic peak..."  # "the motif" references what model heard
outro: "Dissolve the motif into ambience..."       # Still referencing same motif
```

**Bad approach:**
```python
# Contradictory prompts fight conditioning
intro: "piano and strings..."
motif: "guitar and drums..."      # Ignores piano from intro
chorus: "synth and bass..."        # Ignores everything prior
outro: "orchestra..."              # Completely different
```

### Seed Management with Conditioning
**Strategy 1: Locked Seed (Recommended)**
```python
base_seed = 12345
for index, section in enumerate(sections):
    section_seed = base_seed + index
    # Deterministic but varied per section
```
- Reproducible compositions
- Sections still vary (different offsets)
- Conditioning provides coherence

**Strategy 2: Random Seeds**
```python
for section in sections:
    section_seed = random.randint(0, 2**32)
    # Maximum variation
```
- More exploration
- Conditioning essential for coherence
- May drift from theme

---

## 10. Advanced Topics

### Conditioning Window Optimization
**Question:** How much conditioning audio is optimal?

**Current approach:**
- Motif seed: up to 16s
- Previous tail: ~4 beats

**Trade-offs:**
| Window Size | Benefits | Drawbacks |
|-------------|----------|-----------|
| Small (2-4s) | Fast encoding, less VRAM | May miss theme, weak conditioning |
| Medium (6-12s) | Good balance | Standard approach ✓ |
| Large (16-24s) | Strong thematic preservation | Slower, more VRAM, may over-constrain |

**Experimental tuning:**
```python
# Try different motif windows
motif_window = min(24.0, phrase_window_seconds(...))  # Longer
# vs
motif_window = min(8.0, phrase_window_seconds(...))   # Shorter
```

### Multi-Track Conditioning
**Future possibility:** Condition rhythm, bass, harmony separately

```python
# Hypothetical
rhythm_prompt = separate_stems(motif_seed, track="drums")
bass_prompt = separate_stems(motif_seed, track="bass")

chorus_render = musicgen.generate(
    prompt,
    audio_prompt_rhythm=rhythm_prompt,
    audio_prompt_bass=bass_prompt,
)
```

**Benefits:**
- Finer control over what's preserved
- Could maintain rhythm while changing harmony
- Enable partial re-arrangements

**Challenges:**
- Requires multi-track MusicGen variant
- Increased complexity
- Stem separation quality

### Conditioning Strength Control
**Future enhancement:** Adjust how strongly conditioning is applied

```python
# Hypothetical parameter
render = musicgen.generate(
    prompt,
    audio_prompt=conditioning,
    conditioning_strength=0.7,  # 0.0 = ignore, 1.0 = strong influence
)
```

**Use cases:**
- `0.9-1.0`: Strict continuation (default)
- `0.5-0.7`: Gentle guidance
- `0.0-0.3`: Minimal influence (more variation)

**Current limitation:** MusicGen doesn't expose this parameter (yet)

---

## 11. Comparison with Other Approaches

### Alternative 1: No Conditioning (Naive)
```python
# Each section independent
for section in sections:
    render = musicgen.generate(section.prompt)
```

**Result:** Incoherent mess
- ❌ Different timbres
- ❌ Abrupt transitions
- ❌ No thematic continuity

### Alternative 2: Same Seed (Repetitive)
```python
# Lock seed across all sections
seed = 12345
for section in sections:
    render = musicgen.generate(section.prompt, seed=seed)
```

**Result:** Identical or very similar sections
- ❌ Lacks musical development
- ❌ Boring, repetitive
- ⚠️ May work for ambient music

### Alternative 3: Overlap & Splice (Timbre's Old Approach)
```python
# Generate overlapping chunks, crossfade aggressively
chunk1 = musicgen.generate("intro", 29s)
chunk2 = musicgen.generate("intro continuing", 29s)  # Same prompt
result = crossfade(chunk1, chunk2, fade=5s)
```

**Result:** Better than naive, but issues:
- ⚠️ Long crossfades blur details
- ⚠️ No thematic evolution
- ⚠️ Difficult to control structure

### Timbre's Approach: Audio Conditioning (Best)
```python
# Two-stream conditioning with sectional prompts
for section in sections:
    render = musicgen.generate(
        section.prompt,
        audio_prompt=build_conditioning(motif_seed, previous_tail)
    )
```

**Result:** Professional quality
- ✅ Timbral coherence (from conditioning)
- ✅ Smooth transitions (model handles it)
- ✅ Thematic continuity (motif seed)
- ✅ Musical development (different prompts)
- ✅ Structural control (planner-driven)

---

## 12. Future Directions

### Research Opportunities
1. **Learned conditioning windows:** ML model to predict optimal window size
2. **Adaptive conditioning strength:** Vary by section role (stronger for CHORUS, weaker for BRIDGE)
3. **Multi-scale conditioning:** Short-term (tail) + medium-term (motif) + long-term (full composition memory)
4. **Cross-model conditioning:** Use MusicGen conditioning for Riffusion, vice versa

### Potential Enhancements
1. **Conditioning preview:** Show user what audio will be used as conditioning
2. **Manual conditioning:** Allow users to upload custom conditioning audio
3. **Conditioning visualization:** Waveform display showing motif seed + tail regions
4. **A/B testing:** Generate with/without conditioning, let user choose

---

## References

- **MusicGen Paper (Section 3.3 - Melody Conditioning):** https://arxiv.org/abs/2306.05284
- **Audio Conditioning in Transformers:** https://arxiv.org/abs/2301.12503
- **Continuation vs Generation:** https://research.google/pubs/pub51132/

---

**Document Version:** 1.0
**Last Updated:** 2025
**Planner Version:** v3
**Conditioning Strategy:** Two-stream (motif seed + previous tail)
**Default Windows:** 16s motif, 4-beat tail
