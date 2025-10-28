# MusicGen Integration Guide

This document provides a comprehensive technical reference for Timbre's integration with Meta's MusicGen, the primary text-to-music generation backend.

---

## 1. What is MusicGen?

**MusicGen** is Meta AI's open-source text-to-music generation model, part of the AudioCraft toolkit. It represents a significant advancement in audio generation through a novel single-stage architecture.

### Architecture Overview

**Core Components:**
- **EnCodec Neural Codec**: Compresses audio to discrete tokens
  - 4 codebooks sampled at 50 Hz
  - 32 kHz output sampling rate
  - Enables efficient sequence modeling

- **Autoregressive Transformer**: Single-stage language model
  - Predicts all 4 codebooks in parallel with small delays
  - ~50 autoregressive steps per second of audio (highly efficient)
  - No self-supervised semantic representation required (unlike MusicLM)

**Key Innovation:** Unlike cascaded approaches, MusicGen generates music in a single pass by introducing small delays between codebooks, achieving both high quality and efficiency.

**Model Variants:**
| Model ID | Parameters | VRAM (inference) | Quality | Speed |
|----------|------------|------------------|---------|-------|
| `musicgen-small` | 300M | ~4 GB | Good | Fast |
| `musicgen-medium` | 1.5B | ~8 GB | Better | Medium |
| `musicgen-large` | 3.3B | ~12 GB | Best | Slow |
| `musicgen-stereo-small` | 300M | ~4 GB | Good (stereo) | Fast |
| `musicgen-stereo-medium` | 1.5B | ~8 GB | **Better (stereo)** ⭐ | Medium |
| `musicgen-stereo-large` | 3.3B | ~12 GB | Best (stereo) | Slow |

**Timbre Default:** `musicgen-stereo-medium` (balanced quality/speed for long-form compositions)

**Training Data:**
- ~20,000 hours of licensed music
- 400,000 recordings with text descriptions
- Proprietary Meta dataset + licensed sources

---

## 2. Timbre's MusicGen Integration

### Implementation (`worker/services/musicgen.py`)

**Service Architecture:**
```python
class MusicGenService:
    """Text-to-music generation via transformers MusicGen checkpoints."""

    - Model Registry: Maps model IDs → Hugging Face repository paths
    - Model Caching: Lazy-loads models, keeps in memory per model_id
    - Device Selection: Auto-detects CPU/MPS/CUDA, respects TIMBRE_INFERENCE_DEVICE
    - Graceful Degradation: Returns deterministic placeholders when unavailable
```

**Key Methods:**
- `warmup()` → Load default model, return BackendStatus
- `render_section()` → Generate audio for a single composition section
- `_ensure_model()` → Load/cache model handle
- `_generate_waveform()` → Call model.generate() with full parameters
- `_placeholder_waveform()` → Fallback when dependencies missing

### Generation Flow (Per Section)

#### Step 1: Model Loading
```python
async def _ensure_model(model_id: str) -> Tuple[ModelHandle, Optional[str]]:
    # Check cache
    if model_id in self._handles:
        return self._handles[model_id], None

    # Load from Hugging Face
    resolved = MODEL_REGISTRY.get(model_id, model_id)  # e.g., "facebook/musicgen-stereo-medium"
    processor = AutoProcessor.from_pretrained(resolved)
    model = MusicgenForConditionalGeneration.from_pretrained(resolved)
    model = model.to(device).eval()  # Move to GPU/MPS/CPU

    # Cache for reuse
    handle = ModelHandle(model, processor, sample_rate=32000, frame_rate=50)
    self._handles[model_id] = handle
    return handle, None
```

**Device Selection Logic:**
```python
def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    return "cpu"       # CPU fallback
```

#### Step 2: Prompt Composition
Timbre constructs rich, structured prompts from multiple sources:

```python
def _compose_prompt(section, arrangement, theme, previous) -> str:
    segments = [
        section.prompt,        # Base template from planner
        arrangement,           # Arrangement sentence
        theme_clauses,         # Theme descriptor hints
        continuation_hint,     # Smooth transition instruction
    ]
    return " ".join(segments)
```

**Example Assembled Prompt:**
```
"Lift the dreamy lo-fi piano motif into an anthemic chorus where warm piano,
 soft keys shaping the harmony, deep bass grounding the low end and tight drums,
 organic percussion driving the rhythm. Keep the dreamy lo-fi piano motif aligned
 with the downtempo pulse inside a dusty vignette using warm piano, deep bass.
 Continue smoothly from the previous Motif section, preserving timbre while
 evolving the motif."
```

**Arrangement Sentence Generation:**
Action verbs vary by section role to guide musical development:

| Section Role | Action Verb | Example |
|--------------|-------------|---------|
| INTRO | Introduce | "Introduce the arrangement with atmospheric textures colouring..." |
| MOTIF | Feature | "Feature the arrangement with warm piano carrying the lead..." |
| CHORUS | Elevate | "Elevate the arrangement with tight drums driving the rhythm..." |
| BRIDGE | Transform | "Transform the arrangement with layered strings shaping..." |
| DEVELOPMENT | Develop | "Develop the arrangement with expressive guitar exploring..." |
| RESOLUTION | Resolve | "Resolve the arrangement with soft keys easing..." |
| OUTRO | Reimagine | "Reimagine the arrangement with granular noise beds dissolving..." |

**Layer Descriptions:**
```python
phrases = []
if orchestration.rhythm:
    phrases.append(f"{instruments} driving the rhythm")
if orchestration.bass:
    phrases.append(f"{instruments} grounding the low end")
if orchestration.harmony:
    phrases.append(f"{instruments} shaping the harmony")
if orchestration.lead:
    phrases.append(f"{instruments} carrying the lead lines")
if orchestration.textures:
    phrases.append(f"{instruments} colouring the textures")
if orchestration.vocals:
    phrases.append(f"{instruments} adding vocal nuance")
```

#### Step 3: Audio Conditioning
**Critical for long-form coherence.** Timbre feeds reference audio to maintain timbral consistency across sections.

```python
def _build_audio_prompt(handle, motif_seed, previous_render):
    segments = []

    # Motif seed: first 16s from MOTIF section
    if motif_seed:
        motif_window = min(16.0, phrase_window_seconds(motif_seed))
        motif_audio = prepare_segment(motif_seed, max_seconds=motif_window, segment="head")
        segments.append(motif_audio)

    # Previous tail: last ~4 beats from prior section
    if previous_render:
        tail_window = min(16.0, phrase_window_seconds(previous_render, "conditioning_tail_seconds"))
        tail_audio = prepare_segment(previous_render, max_seconds=tail_window, segment="tail")
        segments.append(tail_audio)

    # Concatenate & convert to stereo
    combined = np.concatenate(segments).astype(np.float32)
    stereo = np.stack([combined, combined], axis=0)  # (2, samples)

    return stereo, {
        "audio_conditioning_applied": True,
        "audio_prompt_seconds": len(combined) / sample_rate,
        "audio_prompt_segments": segment_metadata,
    }
```

**Why this works:**
- **Motif seed** captures the core musical identity established in the MOTIF section
- **Previous tail** provides immediate context for smooth continuation
- **Total conditioning**: Typically 16-20 seconds of reference audio
- **Model behavior**: MusicGen "hears" both the thematic material AND the transition point

**Conditioning Metadata Tracked:**
```python
extras["audio_conditioning_requested"] = True/False
extras["audio_conditioning_applied"] = True/False
extras["audio_prompt_segments"] = [
    {"source": "motif", "seconds": 6.0, "window_seconds": 16.0},
    {"source": "previous_tail", "seconds": 4.2, "window_seconds": 4.0}
]
extras["audio_prompt_seconds"] = 10.2
```

#### Step 4: Token Generation
```python
def _generate_waveform(handle, inputs, conditioning_args, prompt, duration, seed, ...):
    # Convert duration to tokens
    tokens = int(duration_seconds * handle.frame_rate)  # e.g., 30s * 50Hz = 1500 tokens
    tokens = min(tokens, int(29.5 * 50))  # Cap at ~29.5s per call

    # Prepare random generator (deterministic if seed provided)
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    # Build generation kwargs
    generation_kwargs = {
        "max_new_tokens": tokens,
        "do_sample": True,
        "top_k": top_k,              # Vocabulary filtering (None = all tokens)
        "top_p": top_p,              # Nucleus sampling (None = disabled)
        "temperature": temperature,  # Sampling randomness (1.0 = standard)
        "guidance_scale": cfg_coef,  # Classifier-Free Guidance strength
    }

    # Generate audio tokens
    with torch.no_grad():
        audio_tokens = handle.model.generate(
            **inputs,                    # Text inputs (processed prompt)
            **conditioning_args,         # Audio conditioning (if provided)
            generator=generator,
            **generation_kwargs
        )

    # Decode tokens → waveform
    waveform = handle.processor.batch_decode(audio_tokens)[0]
    waveform = waveform.cpu().numpy().T  # (samples, channels)

    # Ensure stereo
    if waveform.ndim == 1:
        waveform = waveform.reshape(-1, 1)
    if waveform.shape[1] == 1:
        waveform = np.repeat(waveform, 2, axis=1)

    # Resample to 44.1kHz (Timbre standard)
    if sample_rate != 44100:
        waveform = resample_waveform(waveform, sample_rate, 44100)
        sample_rate = 44100

    return waveform, sample_rate, extras
```

#### Step 5: Post-Processing & Metadata
```python
# Return comprehensive metadata
extras = {
    "sample_rate": 44100,
    "backend": "musicgen",
    "device": "mps",
    "placeholder": False,

    # Section info
    "section_id": "s02",
    "section_label": "Chorus",
    "section_role": "CHORUS",
    "plan_version": "v3",

    # Orchestration
    "orchestration": section.orchestration.model_dump(),
    "arrangement_text": "Elevate the arrangement with...",

    # Theme
    "theme": theme.model_dump(),

    # Sampling parameters
    "top_k": 250,
    "top_p": None,
    "temperature": 1.0,
    "cfg_coef": 3.0,
    "two_step_cfg": False,

    # Conditioning
    "audio_conditioning_requested": True,
    "audio_conditioning_applied": True,
    "audio_prompt_segments": [...],
    "audio_prompt_seconds": 10.2,

    # Timing
    "target_seconds": 48.0,
    "render_seconds": 48.3,

    # Hashes
    "prompt_hash": "a1b2c3d4",
    "seed": 42,
}
```

---

## 3. Sampling Parameters

### Classifier-Free Guidance (CFG)
**Purpose:** Strengthens prompt adherence by amplifying the difference between conditioned and unconditioned predictions.

```python
musicgen_cfg_coef: float = 3.0  # Default

# Higher values (5-10): Stronger prompt adherence, less variation
# Lower values (1-3): More creative freedom, potential drift
# Value 1.0: No guidance (pure sampling)
```

**How it works:**
```
output = unconditioned_output + cfg_coef * (conditioned_output - unconditioned_output)
```

**Timbre Default:** `3.0` (balanced prompt adherence with musical coherence)

### Top-K Filtering
**Purpose:** Limits vocabulary to the K most likely tokens at each step.

```python
musicgen_top_k: Optional[int] = None  # Default (no filtering)

# Common values:
# - None: Full vocabulary (maximum diversity)
# - 250: Moderate filtering
# - 50: Aggressive filtering (more repetitive but coherent)
```

**Timbre Default:** `None` (allows full model expressivity)

### Nucleus Sampling (Top-P)
**Purpose:** Limits vocabulary to tokens whose cumulative probability ≥ P.

```python
musicgen_top_p: Optional[float] = None  # Default (disabled)

# Common values:
# - None: Disabled
# - 0.9: High diversity
# - 0.95: Standard nucleus sampling
```

**Timbre Default:** `None` (not used by default, top-k preferred for music)

### Temperature
**Purpose:** Controls randomness in token selection.

```python
musicgen_temperature: float = 1.0  # Default

# Values:
# - 1.0: Standard sampling (recommended)
# - 0.8: More focused, less variation
# - 1.2: More creative, higher variation
# - 0.0: Greedy decoding (deterministic, not recommended for music)
```

**Timbre Default:** `1.0` (standard temperature)

### Two-Step CFG
**Purpose:** Apply classifier-free guidance in two stages for finer control.

```python
musicgen_two_step_cfg: bool = False  # Default

# Experimental feature, not widely used
```

**Timbre Default:** `False`

### Per-Request Overrides
Users can override any parameter via `GenerationRequest`:

```python
request = GenerationRequest(
    prompt="dreamy synthwave",
    duration_seconds=120,
    model_id="musicgen-stereo-medium",
    seed=42,
    musicgen_top_k=100,          # Override top-k
    musicgen_temperature=0.9,    # Override temperature
    musicgen_cfg_coef=5.0,       # Override CFG scale
)
```

All overrides are tracked in `extras` metadata for debugging.

---

## 4. The 29.5-Second Chunking Strategy

**Problem:** MusicGen generates max ~29.5 seconds per call, but Timbre compositions are 90-180 seconds.

**Solution:** Sequential rendering with audio conditioning

### Chunking Process
```python
# Orchestrator loop (simplified)
tracks = []
motif_seed_render = None

for section in plan.sections:
    # Each section is 16-48 seconds (exceeds 29.5s limit)
    target_seconds = section.target_seconds  # e.g., 36s for Motif

    # Add padding for smooth transitions
    render_seconds = target_seconds + padding  # e.g., 37.25s

    # MusicGen renders up to 29.5s, we handle longer sections by:
    # 1. Rendering with padding (may exceed 29.5s in planning but model caps it)
    # 2. Using audio conditioning to maintain coherence
    # 3. Shaping output to target length in post-processing

    previous_render = tracks[-1].render if tracks else None
    render = await musicgen.render_section(
        section,
        previous_render=previous_render,
        motif_seed=motif_seed_render,
        render_seconds=render_seconds,
    )

    # Extract conditioning tail for next section
    tail = render.waveform[-tail_samples:]

    tracks.append(SectionTrack(section, render, tail))
```

**Key Insight:** While individual calls are capped at ~29.5s, Timbre uses:
1. **Audio conditioning** to provide context across boundaries
2. **Overlapping tails** to ensure smooth transitions
3. **Post-processing** to shape sections to exact target lengths

### How Longer Sections Work
For sections > 29.5s, Timbre doesn't splice multiple renders together during generation. Instead:

1. **Planning** allocates longer target durations (e.g., 48s for Chorus)
2. **Rendering** caps at model max (~29.5s actual generation)
3. **Post-processing** pads or loops to reach target duration using `_shape_to_target_length()`

**Alternative approach (future consideration):**
- Render multiple overlapping 29.5s chunks
- Crossfade them together
- Currently not implemented (audio conditioning + padding sufficient)

---

## 5. Placeholder Fallback System

**Purpose:** Keep Timbre functional when ML dependencies unavailable (testing, CI, lightweight environments).

### When Placeholders Activate
```python
# Triggers:
- torch not installed
- transformers not installed
- Model load failure
- GPU unavailable and CPU too slow
```

### Placeholder Generation
```python
def _placeholder_waveform(prompt, duration, seed):
    sample_rate = 32000
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Deterministic frequency from prompt hash
    base_freq = 110 + (hash(prompt + str(seed)) % 440)  # 110-550 Hz

    # Multi-harmonic synthesis
    waveform = 0.18 * np.sin(2 * np.pi * base_freq * t)          # Fundamental
    waveform += 0.06 * np.sin(2 * np.pi * base_freq * 0.5 * t)   # Sub-harmonic
    waveform += 0.03 * np.sin(2 * np.pi * base_freq * 1.5 * t)   # Overtone
    waveform += 0.02 * np.random.default_rng(seed).standard_normal(len(t))  # Noise

    return np.clip(waveform, -1.0, 1.0), sample_rate, {
        "placeholder": True,
        "placeholder_reason": "torch_unavailable",
        "prompt_hash": hash(prompt),
    }
```

**Benefits:**
- **Deterministic:** Same prompt+seed → same placeholder (reproducible tests)
- **Metadata complete:** All extras populated as if real render
- **UI testable:** CLI can display progress, status, metadata without GPU

**Detection:**
```python
# Check extras
if render.extras.get("placeholder") == True:
    reason = render.extras.get("placeholder_reason")
    # "torch_unavailable", "transformers_unavailable", "load_error:*"
```

---

## 6. Advanced Topics

### Seed Management
**Per-section determinism:**
```python
# Base seed from request or prompt hash
base_seed = request.seed or deterministic_seed(prompt)

# Section-specific seeds
for index, section in enumerate(sections):
    section_seed = base_seed + section.seed_offset  # seed_offset = index

    # Generate with section seed
    generator = torch.Generator().manual_seed(section_seed)
```

**Benefits:**
- Same base seed → reproducible composition
- Different sections vary (due to offsets)
- User can lock seed via `/seed 12345` command

### Memory Management
**Model caching:**
```python
# Models stay in memory after first load
self._handles: Dict[str, ModelHandle] = {}

# Lazy loading
if model_id not in self._handles:
    self._handles[model_id] = load_model(model_id)

# Reuse across sections (crucial for multi-section compositions)
```

**VRAM Considerations:**
- **Model size:** 1.5B parameters ≈ 6 GB float32 (3 GB float16)
- **Activation memory:** ~2-4 GB during generation
- **Total VRAM:** Recommend 8 GB minimum for `stereo-medium`

**CPU Fallback:**
- Functional but slow (2-10x slower than GPU)
- Useful for development/testing
- Set `TIMBRE_INFERENCE_DEVICE=cpu` to force

### Quantization (Future)
**Potential optimizations:**
```python
# bitsandbytes 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = MusicgenForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
)
```

**Benefits:** Reduce VRAM by ~4x (12 GB → 3 GB)
**Tradeoffs:** Slight quality degradation, slower inference

---

## 7. Debugging & Troubleshooting

### Common Issues

**Issue: Model too large for GPU**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Switch to smaller model: `musicgen-small`
2. Force CPU: `TIMBRE_INFERENCE_DEVICE=cpu`
3. Use quantization (future)

**Issue: Distorted audio on MPS (Apple Silicon)**
```
Output sounds like white noise or crackles
```
**Solutions:**
1. Check dtype: MPS should use float32, not float16
2. Force CPU: `TIMBRE_INFERENCE_DEVICE=cpu`

**Issue: Slow generation**
```
30s audio takes 5+ minutes to generate
```
**Diagnosis:**
- Check device: `extras["device"]` should be "mps" or "cuda", not "cpu"
- Check model size: "large" models are inherently slower

**Issue: Sections sound disconnected**
```
Each section has different timbre/style
```
**Diagnosis:**
- Check conditioning: `extras["audio_conditioning_applied"]` should be True
- Check motif seed: First MOTIF section should be captured
- Verify prompt quality: Theme descriptor should be consistent

### Metadata Inspection
```python
# Check render metadata
print(render.extras)

# Key fields:
{
    "backend": "musicgen",
    "device": "mps",
    "placeholder": False,
    "audio_conditioning_applied": True,
    "audio_prompt_seconds": 10.2,
    "top_k": None,
    "temperature": 1.0,
    "cfg_coef": 3.0,
}
```

### Performance Profiling
```python
# Add timing in render_section()
import time

start = time.time()
audio_tokens = model.generate(...)
elapsed = time.time() - start

extras["generation_time_seconds"] = elapsed
extras["tokens_per_second"] = max_new_tokens / elapsed
```

---

## 8. Best Practices

### Prompt Engineering
✅ **DO:**
- Use specific instrument names ("warm piano", "tight drums")
- Include rhythm descriptors ("downtempo pulse", "four-on-the-floor")
- Specify textures ("dreamy haze", "cinematic expanse")
- Provide continuation hints when conditioning

❌ **DON'T:**
- Be vague ("make it sound good")
- Contradict yourself ("upbeat and melancholic")
- Overload with too many instruments (>5-6)

### Parameter Tuning
- **Start with defaults:** CFG 3.0, temp 1.0, no top-k
- **Increase CFG (5-7)** if prompt not being followed
- **Decrease CFG (1-2)** if output too rigid/repetitive
- **Add top-k (50-250)** if output too chaotic
- **Temperature:** Rarely needs adjustment for music

### Seed Management
- **Lock seed** for consistent variations
- **Randomize seed** for exploration
- **Note seed** in metadata for reproducibility

### Resource Planning
- **Small model:** Quick iterations, prototyping
- **Medium model:** Production use (default)
- **Large model:** Maximum quality, special projects

---

## 9. Future Directions

### Potential Enhancements
1. **Streaming generation:** Deliver audio in chunks as it generates
2. **Multi-track stems:** Generate rhythm, bass, harmony separately
3. **Fine-tuning:** LoRA adapters for style customization
4. **Longer context:** >29.5s single-pass generation (if model updated)
5. **Real-time controls:** Adjust tempo, key during generation

### Research Areas
- Optimal conditioning window lengths
- Crossfade vs. conditioning tradeoffs
- Hybrid MusicGen + Riffusion rendering
- Learned section boundary detection

---

## References

- **MusicGen Paper:** https://arxiv.org/abs/2306.05284
- **AudioCraft GitHub:** https://github.com/facebookresearch/audiocraft
- **Hugging Face Docs:** https://huggingface.co/docs/transformers/model_doc/musicgen
- **Model Cards:** https://huggingface.co/facebook/musicgen-stereo-medium

---

**Document Version:** 1.0
**Last Updated:** 2025
**Planner Version:** v3
**MusicGen Default:** facebook/musicgen-stereo-medium
