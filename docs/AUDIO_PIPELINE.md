# Audio Pipeline Reference

This document provides a comprehensive technical reference for Timbre's audio mixing and mastering pipeline, covering section preparation, joining strategies, and the complete mastering chain.

---

## 1. Pipeline Overview

**High-Level Flow:**
```
Section Renders (44.1kHz stereo)
    ↓
Phrase Planning (timing metadata)
    ↓
Section Preparation (shape to target length)
    ↓
Section Joining (butt-join or crossfade)
    ↓
Mastering Chain:
  • RMS Normalization (target 0.2)
  • High-Frequency Tilt (+2.5dB)
  • Soft Limiter (0.98 threshold)
    ↓
Resampling (to 48kHz export)
    ↓
WAV Export (24-bit PCM + TPDF dither)
```

**Implementation:** `worker/services/orchestrator.py:generate()`

---

## 2. Phrase Planning

Before rendering, the orchestrator calculates precise timing metadata for each section.

### Phrase Metadata Structure
```python
@dataclass
class SectionPhrase:
    section_id: str
    tempo_bpm: int
    bars: int
    beats: float
    seconds: float                      # Base duration from bars * seconds_per_beat
    seconds_per_beat: float
    padding_seconds: float              # Extra time for smooth transitions
    conditioning_tail_seconds: float    # Tail length for next section's conditioning

    @property
    def duration_with_padding(self) -> float:
        return self.seconds + self.padding_seconds
```

### Calculation Process
```python
def _build_phrase_plan(plan: CompositionPlan) -> List[SectionPhrase]:
    tempo = plan.tempo_bpm
    beats_per_bar = _beats_per_bar(plan.time_signature)  # Usually 4
    seconds_per_beat = 60.0 / tempo

    phrases = []
    for index, section in enumerate(plan.sections):
        # Base timing from bars
        beats = float(section.bars * beats_per_bar)
        base_seconds = beats * seconds_per_beat
        target_seconds = max(base_seconds, float(section.target_seconds))

        # Padding calculation
        padding = _phrase_padding_seconds(
            seconds_per_beat,
            index,
            total_sections=len(plan.sections)
        )

        # Conditioning tail (for next section)
        conditioning_tail = min(target_seconds, seconds_per_beat * 4.0)

        phrases.append(SectionPhrase(
            section_id=section.section_id,
            tempo_bpm=tempo,
            bars=section.bars,
            beats=beats,
            seconds=target_seconds,
            seconds_per_beat=seconds_per_beat,
            padding_seconds=padding,
            conditioning_tail_seconds=conditioning_tail,
        ))

    return phrases
```

### Padding Strategy
```python
def _phrase_padding_seconds(
    seconds_per_beat: float,
    index: int,
    total_sections: int,
) -> float:
    """Calculate padding based on section position."""

    # Edge sections (intro/outro): less padding
    if index == 0 or index == total_sections - 1:
        padding = seconds_per_beat * 0.75
    # Interior sections: more padding for smoother transitions
    else:
        padding = seconds_per_beat * 1.25

    # Clamp to reasonable bounds
    return float(min(1.5, max(0.35, padding)))
```

**Example at 120 BPM (0.5s per beat):**
- **Edge sections:** 0.75 × 0.5 = 0.375s padding
- **Interior sections:** 1.25 × 0.5 = 0.625s padding

**Why padding?**
- Prevents abrupt cuts during crossfades
- Provides "breathing room" for MusicGen to complete musical phrases
- Allows backend to exceed target slightly without clipping

---

## 3. Section Preparation

After rendering, sections must be shaped to exact target lengths before joining.

### The Problem
- **Backend output:** Variable length (up to 29.5s per MusicGen call, may be shorter)
- **Target length:** Precisely calculated from bars/tempo (e.g., 36.0s for 18-bar section at 120 BPM)
- **Need:** Exact length for beat-aligned mixing

### Solution: `_shape_to_target_length()`
```python
def _shape_to_target_length(
    waveform: np.ndarray,
    target_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Trim or pad waveform to exact target length."""

    data, was_mono = _as_two_dimensional(waveform)

    # Case 1: Too long → Trim with fade
    if data.shape[0] > target_samples:
        trimmed = data[:target_samples].copy()

        # Apply fade-out at trim point (20ms)
        fade_samples = min(int(sample_rate * 0.02), trimmed.shape[0])
        if fade_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
            trimmed[-fade_samples:] *= fade

        return trimmed if not was_mono else trimmed.reshape(-1)

    # Case 2: Too short → Pad with silence
    elif data.shape[0] < target_samples:
        shaped = data.copy()

        # Taper end before padding (20ms fade)
        fade_samples = min(int(sample_rate * 0.02), shaped.shape[0])
        if fade_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
            shaped[-fade_samples:] *= fade

        # Pad with silence
        deficit = target_samples - shaped.shape[0]
        pad = np.zeros((deficit, shaped.shape[1]), dtype=np.float32)
        combined = np.vstack((shaped, pad))

        return combined if not was_mono else combined.reshape(-1)

    # Case 3: Perfect length
    return data if not was_mono else data.reshape(-1)
```

**Key Design Decisions:**
1. **Gentle fades:** Prevent clicks when trimming or before padding
2. **Silent padding:** Simpler than looping, preserves original material
3. **Preserve format:** Return mono if input was mono

**Alternative approach (not currently used):**
- Loop the waveform to fill gaps
- Requires crossfading the loop seam
- More complex, may introduce artifacts

---

## 4. Conditioning Tail Extraction

Each section's tail becomes the conditioning audio for the next section.

### Extraction Process
```python
def _conditioning_tail(
    render: SectionRender,
    phrase: SectionPhrase,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract conditioning tail from rendered section."""

    waveform = ensure_waveform_channels(render.waveform)
    sample_rate = render.sample_rate

    # Calculate tail length (typically last 4 beats)
    tail_seconds = max(0.0, phrase.conditioning_tail_seconds)
    tail_samples = int(round(tail_seconds * sample_rate))

    # Clamp to available audio
    tail_samples = min(tail_samples, waveform.shape[0])

    if tail_samples <= 0:
        return None, sample_rate

    # Extract tail
    tail = waveform[-tail_samples:].copy()

    # Apply fade-in to prevent abrupt start
    fade_samples = min(512, max(1, tail.shape[0] // 32))
    if fade_samples > 0 and tail.shape[0] >= fade_samples:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        if tail.ndim == 1:
            tail[:fade_samples] *= fade
        else:
            tail[:fade_samples] *= fade[:, None]

    return tail, sample_rate
```

**Tail Properties:**
- **Length:** Up to 4 beats (e.g., 2.0s at 120 BPM)
- **Fade-in:** Smooth start (512 samples ≈ 12ms at 44.1kHz)
- **Purpose:** Provides immediate musical context for next section

**Storage:**
```python
@dataclass
class SectionTrack:
    section_id: str
    phrase: SectionPhrase
    render: SectionRender
    backend: str
    conditioning_tail: Optional[np.ndarray]  # Stored here
    conditioning_rate: Optional[int]
```

---

## 5. Section Joining Strategies

Timbre uses two joining methods depending on conditioning status:

### Decision Matrix
```python
def _crossfade_seconds(
    plan: CompositionPlan,
    left_extras: Dict,
    right_extras: Dict,
) -> float:
    """Determine crossfade duration based on conditioning and quality."""

    seconds_per_beat = 60.0 / max(plan.tempo_bpm, 1)

    # Check conditioning status
    left_conditioned = bool(left_extras.get("audio_conditioning_applied"))
    right_conditioned = bool(right_extras.get("audio_conditioning_applied"))

    # Check placeholder status
    placeholder = (
        bool(left_extras.get("placeholder")) or
        bool(right_extras.get("placeholder"))
    )

    # Decision logic
    if placeholder:
        # Placeholders always crossfade (longer)
        return float(min(seconds_per_beat * 0.5, 0.35))

    if not (left_conditioned and right_conditioned):
        # One or neither conditioned → short crossfade
        return float(min(seconds_per_beat * 0.3, 0.15))

    # Both conditioned → butt join (no crossfade)
    return 0.0
```

**Summary Table:**

| Left Conditioned | Right Conditioned | Placeholder? | Join Method | Duration |
|------------------|-------------------|--------------|-------------|----------|
| ✓ | ✓ | No | **Butt Join** | 10ms micro-fade |
| ✓ | ✗ | No | Crossfade | 0.15s |
| ✗ | ✓ | No | Crossfade | 0.15s |
| ✗ | ✗ | No | Crossfade | 0.15s |
| Any | Any | Yes | Crossfade | 0.35s |

### Method 1: Butt Join (Audio-Conditioned Transitions)
**When:** Both sections are audio-conditioned
**Why:** MusicGen already generated a smooth continuation via audio conditioning—preserve it!

```python
def _butt_join(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Join with micro-fade to prevent clicks."""

    left_data = ensure_waveform_channels(left)
    right_data = ensure_waveform_channels(right)

    # 10ms micro-fade for click prevention
    fade_samples = min(
        int(round(sample_rate * 0.01)),
        left_data.shape[0],
        right_data.shape[0]
    )

    if fade_samples <= 0:
        return np.concatenate([left_data, right_data], axis=0)

    # Create crossfade envelopes
    envelope_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
    envelope_in = 1.0 - envelope_out

    # Blend overlap region
    overlap = (
        left_data[-fade_samples:] * envelope_out +
        right_data[:fade_samples] * envelope_in
    )

    # Concatenate: left (except tail) + overlap + right (except head)
    joined = np.concatenate([
        left_data[:-fade_samples],
        overlap,
        right_data[fade_samples:]
    ], axis=0)

    return joined.astype(np.float32)
```

**Key Point:** 10ms is imperceptible but prevents digital clicks at the boundary.

### Method 2: Linear Crossfade (Non-Conditioned Transitions)
**When:** One or neither section is audio-conditioned, or placeholder audio
**Why:** Risk of timbral discontinuity—blend to smooth the transition

```python
def crossfade_append(
    left: np.ndarray,
    right: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Append with linear crossfade."""

    left_data, left_mono = _as_two_dimensional(left)
    right_data, right_mono = _as_two_dimensional(right)

    # Clamp fade to available audio
    fade_samples = min(
        fade_samples,
        max(1, left_data.shape[0] - 1),
        max(1, right_data.shape[0] - 1),
    )

    # Create fade envelopes
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
    fade_in = 1.0 - fade_out

    # Extract regions
    left_main = left_data[:-fade_samples]
    left_tail = left_data[-fade_samples:]
    right_head = right_data[:fade_samples]
    right_rest = right_data[fade_samples:]

    # Blend overlap
    blended = left_tail * fade_out + right_head * fade_in

    # Concatenate
    merged = np.vstack((left_main, blended, right_rest))

    # Restore original format
    if left_mono and right_mono:
        return merged.reshape(-1)
    return merged
```

**Crossfade Visualization:**
```
Left Section:        [███████████████▓▓▓▓]
                                        ↓↓↓↓  (fade-out)
Right Section:                      [▓▓▓▓███████████████]
                                     ↑↑↑↑  (fade-in)

Result:              [███████████████████████████████████]
                                    Blended region
```

### Metadata Recording
Every transition is documented:
```python
crossfade_record = {
    "from": "s01",  # Section ID
    "to": "s02",
    "seconds": 0.0,  # 0.0 = butt join, >0 = crossfade
    "mode": "butt",  # or "crossfade"
    "conditioning": {
        "left_audio_conditioned": True,
        "right_audio_conditioned": True,
        "left_placeholder": False,
        "right_placeholder": False,
    }
}
```

**Stored in:** `extras.mix.crossfades[]`

---

## 6. Mastering Chain

After joining all sections, Timbre applies a professional mastering chain.

### Step 1: RMS Normalization
**Purpose:** Bring overall loudness to target level

```python
def normalise_loudness(
    waveform: np.ndarray,
    target_rms: float = 0.18,
    max_gain: float = 4.0,
) -> np.ndarray:
    """Scale waveform RMS toward target with gain ceiling."""

    data, was_mono = _as_two_dimensional(waveform)

    # Calculate current RMS
    current = rms_level(data)  # √(mean(signal²))

    if current <= 1e-6:
        return waveform  # Silent, don't amplify noise

    # Calculate gain
    gain = min(target_rms / current, max_gain)

    # Apply
    adjusted = data * gain

    return adjusted.reshape(-1) if was_mono else adjusted
```

**Timbre Default:** `target_rms = 0.2` (moderately loud, room for limiting)

**Gain Ceiling:** `4.0` (max 12dB boost) prevents over-amplifying quiet sections

**Why not just normalize to peak = 1.0?**
- Peak normalization doesn't account for perceived loudness
- RMS better represents average energy/loudness
- Allows consistent loudness across different compositions

### Step 2: High-Frequency Tilt
**Purpose:** Restore brilliance lost during encoding/generation

```python
def tilt_highs(
    waveform: np.ndarray,
    sample_rate: int,
    gain_db: float = 2.5,
    exponent: float = 1.2,
) -> np.ndarray:
    """Apply gentle high-frequency tilt via FFT."""

    data = ensure_waveform_channels(waveform).astype(np.float32)
    mono = False
    if data.ndim == 1:
        data = data[:, None]
        mono = True

    # FFT
    spectrum = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(data.shape[0], d=1.0/sample_rate)

    # Build tilt curve
    nyquist = float(freqs[-1]) if freqs[-1] > 0 else float(sample_rate) / 2.0
    ratio = np.clip(freqs / max(nyquist, 1.0), 0.0, 1.0)

    # Exponential curve (gentle tilt, not harsh shelf)
    high_gain = 10.0 ** (gain_db / 20.0)  # 2.5dB = 1.33x
    curve = 1.0 + (high_gain - 1.0) * np.power(ratio, exponent)

    # Apply to spectrum
    if spectrum.ndim == 1:
        spectrum *= curve
    else:
        spectrum *= curve[:, None]

    # Inverse FFT
    boosted = np.fft.irfft(spectrum, n=data.shape[0], axis=0).real

    # Prevent clipping
    peak = float(np.max(np.abs(boosted)))
    if peak > 1.0:
        boosted = boosted / peak

    return boosted[:, 0] if mono else boosted
```

**Tilt Curve Visualization:**
```
Gain (dB)
  2.5 |                                    ╱
      |                               ╱
  2.0 |                          ╱
      |                     ╱
  1.5 |                ╱
      |           ╱
  1.0 |      ╱
      | ╱
  0.0 |___________________
      0 Hz              Nyquist (22.05kHz at 44.1kHz)
```

**Why exponent = 1.2?**
- Creates gentle slope (not a harsh high-shelf)
- Boosts air/presence frequencies without harshness
- Compensates for lossy encoding roll-off

### Step 3: Soft Limiter
**Purpose:** Prevent clipping while preserving dynamics

```python
def soft_limiter(
    waveform: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    """Apply tanh-based soft knee compression above threshold."""

    if threshold <= 0.0:
        return np.clip(waveform, -1.0, 1.0)

    data = ensure_waveform_channels(waveform)

    # Find samples exceeding threshold
    over = np.abs(data) > threshold

    if np.any(over):
        # Apply tanh compression to peaks
        exceeded = data[over]
        data = data.astype(np.float32, copy=True)
        data[over] = threshold * np.tanh(exceeded / threshold)

    return data
```

**How tanh limiting works:**
```python
# Linear region (below threshold):
if abs(x) <= threshold:
    output = x

# Soft knee (above threshold):
else:
    output = threshold * tanh(x / threshold)
    # tanh asymptotically approaches ±1.0
    # Result stays below threshold but never hard-clips
```

**Timbre Default:** `threshold = 0.98`
- Catches peaks above -0.17dB
- Prevents digital clipping (>1.0)
- Maintains dynamics (doesn't brick-wall compress)

**Why not brick-wall limiting?**
- Preserves transients and dynamics
- Sounds more natural
- Avoids pumping artifacts

### Mastering Chain Summary
```python
# Complete chain (orchestrator.py:generate)
waveform = normalise_loudness(waveform, target_rms=0.2)  # 1. Loudness
waveform = tilt_highs(waveform, sample_rate)             # 2. Presence
waveform = soft_limiter(waveform, threshold=0.98)        # 3. Safety
```

**Typical signal flow:**
```
Input:    Peak ≈ 0.5,  RMS ≈ 0.1   (quiet, dull)
   ↓ Normalize (gain ≈ 2.0x)
         Peak ≈ 1.0,  RMS ≈ 0.2   (louder)
   ↓ Tilt (+2.5dB HF)
         Peak ≈ 1.05, RMS ≈ 0.21  (brighter)
   ↓ Soft Limit
Output:   Peak ≈ 0.98, RMS ≈ 0.20  (safe, polished)
```

---

## 7. Resampling

After mastering, resample to export sample rate.

### Why Resample?
- **Backend output:** 44.1kHz (MusicGen standard after Timbre conversion)
- **Professional standard:** 48kHz (video, broadcast, modern DAWs)
- **Higher fidelity:** Better Nyquist frequency (24kHz vs 22.05kHz)

### Implementation
```python
def resample_waveform(
    waveform: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """Resample using torchaudio if available, else linear interpolation."""

    if src_rate == dst_rate:
        return ensure_waveform_channels(waveform)

    data = ensure_waveform_channels(waveform)
    mono = data.ndim == 1

    if mono:
        data = data[:, None]

    # Preferred: torchaudio (high-quality polyphase)
    if torchaudio is not None:
        tensor = torch.as_tensor(data.T, dtype=torch.float32)
        resampled = torchaudio.functional.resample(
            tensor,
            orig_freq=src_rate,
            new_freq=dst_rate,
        )
        result = resampled.T.cpu().numpy()
        return result[:, 0] if mono else result

    # Fallback: linear interpolation
    scale = dst_rate / src_rate
    target_length = int(round(data.shape[0] * scale))

    x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_length, endpoint=False)

    resampled = np.empty((target_length, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        resampled[:, channel] = np.interp(x_new, x_old, data[:, channel])

    return resampled[:, 0] if mono else resampled
```

**Quality Comparison:**
- **torchaudio:** Polyphase filter, minimal aliasing
- **Linear interp:** Simple, slight quality loss, no dependencies

### Post-Resample Safety
```python
# Resample may introduce peaks > 0.98
if sample_rate != export_rate:
    waveform = resample_waveform(waveform, sample_rate, export_rate)
    sample_rate = export_rate
    waveform = soft_limiter(waveform, threshold=0.98)  # Re-limit
```

---

## 8. WAV Export

Final step: write to disk with professional quality.

### Export Parameters
```python
# Defaults (Settings)
export_sample_rate: int = 48000     # 48kHz
export_bit_depth: str = "pcm24"     # 24-bit signed integer
export_format: str = "wav"           # WAV container
```

### Bit Depth Options
| Setting | Format | Range | Use Case |
|---------|--------|-------|----------|
| `pcm16` | Signed 16-bit | -32768 to +32767 | Compatibility, smaller files |
| `pcm24` | Signed 24-bit | -8388608 to +8388607 | **Professional standard** |
| `pcm32` | Signed 32-bit | Full int32 | Maximum precision |
| `float32` | IEEE float | -1.0 to +1.0 | DAW import, no quantization |

**Timbre Default:** `pcm24` (professional quality, 50% smaller than float32)

### TPDF Dithering
**Purpose:** Minimize quantization artifacts when reducing bit depth

```python
def _apply_tpdf_dither(data: np.ndarray, bit_depth: int) -> np.ndarray:
    """Triangular Probability Density Function dither."""

    if bit_depth <= 0:
        return data

    # LSB step size
    step = 1.0 / float(2 ** (bit_depth - 1))

    # Generate two uniform random arrays
    rng = np.random.default_rng()
    noise1 = rng.random(data.shape, dtype=np.float32) * step
    noise2 = rng.random(data.shape, dtype=np.float32) * step

    # TPDF = difference of two uniform distributions (triangular PDF)
    dither = noise1 - noise2

    return np.clip(data + dither, -1.0, 1.0)
```

**Why TPDF?**
- Decorrelates quantization noise from signal
- Noise floor sounds like smooth hiss (not harsh distortion)
- Preserves low-level detail

**When to skip dithering:**
- `float32` export (no quantization)
- Special cases where determinism required

### Complete Export Flow
```python
def write_waveform(
    path: Path,
    waveform: np.ndarray,
    sample_rate: int,
    bit_depth: str = "pcm24",
    audio_format: str = "wav",
    dither: bool = True,
) -> None:
    """Persist waveform with professional quality."""

    data = ensure_waveform_channels(waveform)

    # Preferred: soundfile (supports all formats)
    if sf is not None and audio_format == "wav":
        subtype = _soundfile_subtype(bit_depth)  # "PCM_24", etc.
        export = data.astype(np.float32)

        if dither and subtype != "FLOAT":
            export = _apply_tpdf_dither(export, _bit_depth_to_int(bit_depth))

        sf.write(path, export, sample_rate, subtype=subtype)
        return

    # Fallback: wave module (16-bit only)
    target_bit_depth = _bit_depth_to_int(bit_depth)
    if target_bit_depth != 16:
        logger.warning("PCM export requires soundfile; falling back to 16-bit")
        target_bit_depth = 16

    export = data.astype(np.float32, copy=True)
    if dither:
        export = _apply_tpdf_dither(export, target_bit_depth)

    # Quantize to integer
    scale = float(2 ** (target_bit_depth - 1) - 1)
    pcm = np.clip(export, -1.0, 1.0)
    pcm = (pcm * scale).astype(np.int16 if target_bit_depth == 16 else np.int32)

    # Write WAV
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        wav_file.setsampwidth(2 if target_bit_depth == 16 else 4)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
```

**Result:** Professional-quality WAV file ready for distribution or further production

---

## 9. Metadata Tracking

Throughout the pipeline, Timbre meticulously tracks every step for debugging and reproducibility.

### Section-Level Metadata
```python
# Per section (extras.sections[i])
{
    "backend": "musicgen",
    "device": "mps",
    "placeholder": False,

    "section_id": "s02",
    "section_role": "CHORUS",
    "orchestration": {...},
    "arrangement_text": "Elevate the arrangement with...",

    "phrase": {
        "seconds": 48.0,
        "beats": 96.0,
        "bars": 24,
        "tempo_bpm": 120,
        "seconds_per_beat": 0.5,
        "padding_seconds": 0.625,
        "conditioning_tail_seconds": 2.0,
        "render_hint_seconds": 48.625,
    },

    "audio_conditioning_applied": True,
    "audio_prompt_seconds": 10.2,
    "audio_prompt_segments": [...],

    "target_seconds": 48.0,
    "render_seconds": 48.3,
}
```

### Mix-Level Metadata
```python
# Overall mix (extras.mix)
{
    "sample_rate": 48000,
    "duration_seconds": 172.5,
    "target_rms": 0.2,

    "section_rms": [0.18, 0.21, 0.23, 0.19],  # Per-section loudness

    "crossfades": [
        {
            "from": "s00",
            "to": "s01",
            "seconds": 0.15,
            "mode": "crossfade",
            "conditioning": {
                "left_audio_conditioned": False,
                "right_audio_conditioned": True,
                "left_placeholder": False,
                "right_placeholder": False,
            }
        },
        {
            "from": "s01",
            "to": "s02",
            "seconds": 0.0,
            "mode": "butt",
            "conditioning": {
                "left_audio_conditioned": True,
                "right_audio_conditioned": True,
                ...
            }
        },
        ...
    ]
}
```

### Motif Seed Metadata
```python
# Extracted motif (extras.motif_seed)
{
    "captured": True,
    "path": "/path/to/job_abc123_motif.wav",
    "section_id": "s01",
    "section_role": "MOTIF",
    "sample_rate": 44100,

    # Spectral analysis
    "duration_seconds": 36.0,
    "rms": 0.21,
    "spectral_centroid_hz": 2450.5,
    "chroma_vector": [0.12, 0.08, 0.15, ...],  # 12 pitch classes
    "dominant_pitch_class": "G",
    "plan_key_alignment": 0.15,  # How well motif aligns with planned key

    # Theme reference
    "motif_text": "dreamy lo-fi piano",
    "motif_rhythm": "downtempo pulse",
}
```

---

## 10. Troubleshooting

### Issue: Audible Clicks at Section Boundaries
**Symptoms:** Short clicks or pops between sections

**Diagnosis:**
```python
# Check crossfade metadata
extras["mix"]["crossfades"]

# Look for:
- mode: "butt" with very short sections (< 1s)
- conditioning: both False but using butt join
```

**Solutions:**
1. Check conditioning was actually applied
2. Increase minimum crossfade duration
3. Verify fade calculations aren't rounding to 0

### Issue: Inconsistent Loudness Across Sections
**Symptoms:** Some sections much louder/quieter than others

**Diagnosis:**
```python
# Check per-section RMS
extras["mix"]["section_rms"]
# [0.05, 0.25, 0.22, 0.06]  ← Outliers!
```

**Solutions:**
1. Check backend output quality
2. Verify RMS normalization applied to full mix
3. Inspect individual section renders for issues

### Issue: Loss of High Frequencies
**Symptoms:** Dull, muffled sound

**Diagnosis:**
```python
# Check if HF tilt applied
# Look for tilt_highs() in orchestrator.py:generate()
```

**Solutions:**
1. Verify tilt_highs() called after RMS normalization
2. Increase gain_db (default 2.5dB)
3. Check resampling quality (use torchaudio if available)

### Issue: Digital Clipping (> 1.0)
**Symptoms:** Harsh distortion, waveform peaks clipped

**Diagnosis:**
```python
# Check peak levels
peak = np.max(np.abs(waveform))
print(f"Peak: {peak}")  # Should be ≤ 0.98
```

**Solutions:**
1. Verify soft_limiter() called after all processing
2. Re-limit after resampling
3. Check gain ceiling in normalise_loudness() (max_gain=4.0)

---

## 11. Best Practices

### For Optimal Audio Quality
1. **Keep source quality high:** Use `musicgen-stereo-medium` or better
2. **Don't skip mastering:** Always apply the full chain
3. **Monitor RMS:** Target 0.2 is loud enough without distortion
4. **Use conditioning:** Enables butt joins (preserves MusicGen quality)
5. **Export at 48kHz/24-bit:** Professional standard

### For Debugging
1. **Check metadata first:** `extras.mix.crossfades`, `extras.sections[*]`
2. **Listen to sections individually:** Export motif seed, inspect each render
3. **Track RMS throughout:** Section RMS → Mix RMS
4. **Verify conditioning:** Should be True for all sections after MOTIF

### For Performance
1. **Batch operations:** Process all sections before mixing
2. **Use torchaudio:** Much faster resampling than numpy interp
3. **Cache models:** Don't reload between sections

---

## 12. Future Enhancements

### Potential Improvements
1. **Adaptive crossfade lengths:** Based on spectral similarity
2. **Multiband compression:** Independent processing per frequency band
3. **Stereo width control:** Adjust stereo field
4. **Automated mastering presets:** Genre-specific chains
5. **Stem export:** Separate rhythm/bass/harmony tracks

### Research Areas
- Optimal RMS target for different genres
- Machine learning for crossfade point detection
- Real-time mastering preview
- Advanced dithering algorithms (noise shaping)

---

## References

- **Digital Audio Theory:** https://www.dspguide.com/
- **Mastering Techniques:** https://www.soundonsound.com/techniques/mastering
- **Dithering Explained:** http://www.rane.com/note148.html
- **Audio Resampling:** https://ccrma.stanford.edu/~jos/resample/

---

**Document Version:** 1.0
**Last Updated:** 2025
**Planner Version:** v3
**Default Export:** 48kHz, 24-bit PCM with TPDF dither
