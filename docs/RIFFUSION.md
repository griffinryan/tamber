# Riffusion Backend Guide

This document provides a technical reference for Timbre's Riffusion integration—a spectrogram-based diffusion model serving as an alternative backend to MusicGen.

---

## 1. What is Riffusion?

**Riffusion** is a text-to-music model based on Stable Diffusion, fine-tuned to generate music by creating mel-spectrogram images that are then inverted to audio.

### Architecture Overview

**Core Approach:**
1. **Fine-tuned Stable Diffusion v1.5:** Trained on spectrograms instead of natural images
2. **Mel-spectrogram generation:** Creates visual representation of audio
3. **Phase reconstruction:** Inverts spectrogram back to audio using Griffin-Lim algorithm

**Key Characteristics:**
- **Model size:** ~1GB (smaller than MusicGen)
- **Inference speed:** Moderate (50-100 diffusion steps)
- **Quality:** Good for ambient/textural music
- **Device support:** Works on CPU, MPS, CUDA

### Training Data
- Spectrograms paired with text descriptions
- Focused on shorter clips (5-10 seconds typical)
- Optimized for ambient, electronic, experimental genres

---

## 2. Why Riffusion is Secondary in Timbre

### Advantages
✅ **Lighter weight:** Smaller model, less VRAM
✅ **CPU-friendly:** Usable without GPU
✅ **Unique aesthetic:** Different sound character than MusicGen
✅ **Faster experimentation:** Quick iterations for ambient pieces

### Limitations
❌ **Phase reconstruction artifacts:** Griffin-Lim introduces approximation errors
❌ **Conditioning quality:** Spectrogram init images less precise than direct audio
❌ **Slower than MusicGen:** 50-75 diffusion steps vs. single transformer pass
❌ **MPS instability:** float16 causes distortion on Apple Silicon (requires float32)
❌ **Limited training data:** Less diverse than MusicGen's 20k hours

**Result:** Riffusion works well for specific use cases but MusicGen is the primary backend.

---

## 3. Implementation (`worker/services/riffusion.py`)

### Service Architecture

```python
class RiffusionService:
    """Spectrogram diffusion pipeline for text-to-music generation."""

    - Diffusion Pipeline: DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
    - Spectrogram Decoder: SpectrogramImageDecoder (two modes: "default", "fast")
    - Device Selection: Auto-detects CPU/MPS/CUDA, MPS uses float32
    - Allow Inference: Can be disabled via TIMBRE_RIFFUSION_ALLOW_INFERENCE=0
```

**Key Methods:**
- `warmup()` → Load pipeline, verify spectrogram decoder
- `render_section()` → Generate audio for a section
- `_load_pipeline()` → Load diffusion model
- `_run_inference()` → Execute diffusion process
- `_prepare_init_image()` → Convert audio → spectrogram image for conditioning
- `_audio_from_images()` → Invert spectrogram → audio

---

## 4. Generation Flow

### Step 1: Pipeline Loading

```python
async def _ensure_pipeline(model_id: str):
    # Check if inference allowed
    if not self._allow_inference:
        return None, "inference_disabled"

    # Load diffusion pipeline
    from diffusers import DiffusionPipeline

    resolved = MODEL_REGISTRY.get(model_id, model_id)  # "riffusion/riffusion-model-v1"

    pipeline = DiffusionPipeline.from_pretrained(
        resolved,
        torch_dtype=self._dtype,  # float32 on MPS, float16 on CUDA
        safety_checker=None,
        trust_remote_code=True,
    )

    pipeline = pipeline.to(self._device)
    pipeline.set_progress_bar_config(disable=True)

    # Get sample rate
    sample_rate = getattr(pipeline, "sample_rate", 44100)

    return PipelineHandle(pipeline=pipeline, sample_rate=sample_rate), None
```

**Device/Dtype Selection:**
```python
def _select_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        # MPS requires float32 to avoid distortion
        return "mps", torch.float32
    return "cpu", torch.float32
```

### Step 2: Spectrogram Decoder Initialization

Riffusion requires bidirectional audio ↔ spectrogram conversion.

```python
def _resolve_spectrogram_decoder(sample_rate: int):
    params = SpectrogramParams(sample_rate=sample_rate)

    try:
        decoder = SpectrogramImageDecoder(params, device="cpu")
    except RuntimeError as exc:
        logger.warning(f"Spectrogram decoder unavailable: {exc}")
        return None

    # Detect decoder mode ("default" or "fast")
    self._spectrogram_mode = getattr(decoder._inverse_mel, "decoder_mode", "unknown")

    return decoder
```

**SpectrogramParams** (from `riffusion_spectrogram.py`):
```python
@dataclass
class SpectrogramParams:
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    win_length: int = 2048
    n_mels: int = 128
    f_min: float = 0.0
    f_max: Optional[float] = None  # Defaults to sample_rate / 2
```

### Step 3: Init Image Preparation (Conditioning)

Convert previous render's tail to spectrogram image for conditioning.

```python
def _prepare_init_image(previous_render: SectionRender):
    # Get last ~4 seconds of previous render
    waveform = ensure_waveform_channels(previous_render.waveform)
    total_seconds = waveform.shape[0] / previous_render.sample_rate

    tail_seconds = min(4.0, total_seconds)
    tail_samples = int(tail_seconds * previous_render.sample_rate)
    tail = waveform[-tail_samples:]

    # Get decoder
    decoder = self._resolve_spectrogram_decoder(previous_render.sample_rate)
    if not decoder:
        return None

    # Encode audio → spectrogram image
    try:
        init_image = decoder.encode(tail, previous_render.sample_rate)
        return init_image  # PIL Image
    except Exception as exc:
        logger.debug(f"Failed to encode continuation spectrogram: {exc}")
        return None
```

**Encoder (audio → spectrogram image):**
```python
def encode(waveform: np.ndarray, sample_rate: int) -> PILImage:
    # 1. Convert to mono if stereo
    if waveform.ndim > 1:
        mono = waveform.mean(axis=1)
    else:
        mono = waveform

    # 2. Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=mono,
        sr=sample_rate,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        n_mels=params.n_mels,
        fmin=params.f_min,
        fmax=params.f_max,
    )

    # 3. Convert to dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 4. Normalize to 0-255 uint8
    normalized = ((mel_db + 80) / 80 * 255).clip(0, 255).astype(np.uint8)

    # 5. Convert to RGB PIL Image
    image = Image.fromarray(normalized).convert("RGB")

    return image
```

### Step 4: Diffusion Inference

```python
def _run_inference(
    handle: PipelineHandle,
    prompt: str,
    duration_seconds: float,
    guidance_scale: float,
    seed: Optional[int],
    init_image: Optional[PILImage],
    init_strength: float,  # 0.55 default
    num_inference_steps: int,  # 75 default
    scheduler_name: Optional[str],
):
    # Configure scheduler if specified
    if scheduler_name:
        self._configure_scheduler(handle.pipeline, scheduler_name)

    # Setup random generator
    generator = None
    if seed is not None:
        generator = torch.Generator(device=self._device).manual_seed(seed)

    # Build kwargs
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "audio_length_in_s": max(1.0, float(duration_seconds)),
        "generator": generator,
    }

    # Add init image conditioning if available
    if init_image is not None:
        kwargs["image"] = init_image
        kwargs["strength"] = init_strength  # 0.55 = blend 55% new, 45% init

    # Run diffusion
    result = handle.pipeline(**kwargs)

    # Extract audio
    audio_list = getattr(result, "audios", None)
    if audio_list:
        waveform = np.asarray(audio_list[0], dtype=np.float32)
    else:
        # Some pipelines return images, need to invert
        waveform, sample_rate = self._audio_from_images(result, handle.sample_rate)

    return waveform, sample_rate, extras
```

**Scheduler Options** (can be configured):
- `dpmpp_2m` (default): DPM++ 2M scheduler (fast, high quality)
- `euler_a`: Euler Ancestral (stochastic)
- `ddim`: DDIM (deterministic)
- `pndm`: PNDM (older, slower)

### Step 5: Spectrogram Inversion (Image → Audio)

**The Griffin-Lim Algorithm:**
```python
def _audio_from_images(result, default_sample_rate):
    # Extract spectrogram image
    pil_image = result.images[0]  # PIL Image (RGB)

    # Get decoder
    decoder = self._resolve_spectrogram_decoder(default_sample_rate)
    if not decoder:
        raise GenerationFailure("spectrogram_decoder_unavailable")

    # Decode image → audio
    waveform, sample_rate = decoder.decode(
        pil_image,
        refine_phase=self._enable_phase_refinement,  # Default True
    )

    return waveform, sample_rate
```

**Decoder Implementation:**
```python
def decode(
    spectrogram_image: PILImage,
    refine_phase: bool = True,
) -> Tuple[np.ndarray, int]:
    # 1. Convert PIL Image → numpy array
    image_array = np.array(spectrogram_image.convert("L"))  # Grayscale

    # 2. Denormalize (reverse encoding)
    mel_db = (image_array.astype(np.float32) / 255.0) * 80.0 - 80.0

    # 3. Convert dB → power
    mel_spec = librosa.db_to_power(mel_db)

    # 4. Invert mel-spectrogram → STFT
    stft = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=params.sample_rate,
        n_fft=params.n_fft,
        fmin=params.f_min,
        fmax=params.f_max,
    )

    # 5. Griffin-Lim phase reconstruction
    waveform = librosa.griffinlim(
        stft,
        n_iter=32 if refine_phase else 16,  # More iterations = better quality
        hop_length=params.hop_length,
        win_length=params.win_length,
    )

    return waveform, params.sample_rate
```

**Griffin-Lim Limitations:**
- **Phase loss:** Spectrograms don't contain phase information
- **Approximation:** Iterative algorithm estimates phase
- **Artifacts:** Can introduce metallic/robotic sound
- **Quality trade-off:** More iterations = better but slower

---

## 5. Configuration & Parameters

### Environment Variables

```bash
# Enable/disable Riffusion inference
TIMBRE_RIFFUSION_ALLOW_INFERENCE=1  # Default: enabled

# Guidance scale (prompt adherence)
TIMBRE_RIFFUSION_GUIDANCE_SCALE=8.5  # Default

# Number of diffusion steps
TIMBRE_RIFFUSION_NUM_INFERENCE_STEPS=75  # Default

# Scheduler algorithm
TIMBRE_RIFFUSION_SCHEDULER=dpmpp_2m  # Default

# Enable phase refinement (more Griffin-Lim iterations)
TIMBRE_RIFFUSION_ENABLE_PHASE_REFINEMENT=true  # Default
```

### Per-Request Overrides

```python
request = GenerationRequest(
    prompt="ambient dreamy textures",
    model_id="riffusion-v1",
    riffusion_guidance_scale=10.0,       # Stronger prompt adherence
    riffusion_num_inference_steps=100,  # More steps (slower, better quality)
    riffusion_scheduler="euler_a",       # Different scheduler
)
```

### Init Image Conditioning

**Strength Parameter (0.0 - 1.0):**
- `0.0`: Ignore init image completely (pure text generation)
- `0.3`: Light influence from init image
- `0.55`: **Default** - balanced blend
- `0.7`: Strong influence (close to init image)
- `1.0`: Maximum influence (minimal changes)

**Timbre uses 0.55:** Good balance for section transitions

---

## 6. Comparison with MusicGen

| Feature | MusicGen | Riffusion |
|---------|----------|-----------|
| **Approach** | Direct audio tokens | Spectrogram diffusion |
| **Model Size** | 1.5GB (medium) | 1GB |
| **Inference Speed** | ~5s for 30s (GPU) | ~15s for 30s (GPU) |
| **Audio Quality** | High (clear, coherent) | Good (some artifacts) |
| **Conditioning** | Direct audio prompting | Spectrogram init images |
| **CPU Performance** | Very slow (10x+) | Slow (3-5x) |
| **MPS (Apple Silicon)** | float16 OK | **float32 required** |
| **Best for** | Long-form, structured music | Ambient, textures, experiments |
| **Phase reconstruction** | N/A (direct audio) | Griffin-Lim artifacts |

**When to use Riffusion:**
- ✅ Lighter-weight experimentation
- ✅ Ambient/textural pieces
- ✅ CPU-only environments
- ✅ Unique aesthetic preference

**When to use MusicGen:**
- ✅ Production quality
- ✅ Long-form compositions
- ✅ Structured music (intro/chorus/outro)
- ✅ GPU available

---

## 7. Troubleshooting

### Issue: White Noise / Distorted Output on MPS

**Symptoms:** Output sounds like static or harsh noise

**Cause:** MPS backend with float16 precision

**Solution:**
```bash
# Force float32 (already default in Timbre)
# Or force CPU:
export TIMBRE_INFERENCE_DEVICE=cpu
```

**Verification:**
```python
# Check dtype in extras
extras["dtype"]  # Should be "torch.float32" on MPS
```

### Issue: Metallic/Robotic Artifacts

**Symptoms:** Audio has unnatural metallic sheen

**Cause:** Griffin-Lim phase reconstruction limitations

**Solutions:**
1. **Enable phase refinement:**
   ```bash
   export TIMBRE_RIFFUSION_ENABLE_PHASE_REFINEMENT=true
   ```

2. **Increase diffusion steps:**
   ```python
   request.riffusion_num_inference_steps = 100  # vs default 75
   ```

3. **Try different scheduler:**
   ```python
   request.riffusion_scheduler = "euler_a"  # vs default "dpmpp_2m"
   ```

4. **Consider switching to MusicGen** for cleaner output

### Issue: Slow Generation

**Symptoms:** Taking >30s to generate 10s audio

**Diagnosis:**
- Check device: `extras["device"]` should be "mps" or "cuda", not "cpu"
- Check steps: Higher `num_inference_steps` = slower

**Solutions:**
1. **Reduce steps:**
   ```python
   request.riffusion_num_inference_steps = 50  # vs 75
   ```

2. **Use GPU:**
   ```bash
   export TIMBRE_INFERENCE_DEVICE=mps  # or cuda
   ```

3. **Consider MusicGen** for faster generation

### Issue: Spectrogram Decoder Unavailable

**Symptoms:**
```
ERROR: Spectrogram decoder unavailable: ...
```

**Cause:** Missing dependencies (librosa, PIL)

**Solution:**
```bash
uv sync --project worker --extra inference
```

**Verification:**
```python
# Check status
status = await riffusion_service.warmup()
print(status.details["decoder_ready"])  # Should be True
```

---

## 8. Placeholder Fallback

When Riffusion unavailable, Timbre generates deterministic placeholders.

### Triggers
```python
# Inference disabled
TIMBRE_RIFFUSION_ALLOW_INFERENCE=0

# torch unavailable
# diffusers unavailable
# Pipeline load failed
```

### Placeholder Generation

Same as MusicGen placeholder:
```python
def _placeholder_waveform(prompt, duration, seed):
    # Multi-harmonic sine synthesis
    base_freq = 220 + (hash(prompt + str(seed)) % 360)

    waveform = 0.2 * np.sin(2π * base_freq * t)
    waveform += 0.05 * np.sin(2π * base_freq * 0.5 * t)
    waveform += 0.02 * np.random.default_rng(seed).standard_normal(len(t))

    return waveform, 44100, {
        "placeholder": True,
        "placeholder_reason": "torch_unavailable",
    }
```

**Detection:**
```python
if render.extras.get("placeholder") == True:
    reason = render.extras.get("placeholder_reason")
```

---

## 9. Future Enhancements

### Potential Improvements

1. **Neural phase reconstruction:**
   - Replace Griffin-Lim with learned neural vocoder (WaveGlow, HiFi-GAN)
   - Eliminate artifacts
   - Higher quality output

2. **Stereo spectrograms:**
   - Current: mono spectrograms
   - Future: Left/right channel spectrograms
   - Better stereo imaging

3. **Higher resolution:**
   - Increase n_mels (128 → 256)
   - Better frequency resolution
   - More detail

4. **Faster schedulers:**
   - Explore DDPM-based schedulers
   - Reduce steps (75 → 25)
   - Maintain quality

5. **Fine-tuning:**
   - Custom Riffusion model for Timbre's aesthetic
   - Train on curated dataset
   - Better prompt adherence

---

## 10. References

- **Riffusion Project:** https://www.riffusion.com/
- **Riffusion GitHub:** https://github.com/riffusion/riffusion
- **Model Card:** https://huggingface.co/riffusion/riffusion-model-v1
- **Griffin-Lim Paper:** http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.306.7858
- **Stable Diffusion:** https://github.com/Stability-AI/stablediffusion

---

**Document Version:** 1.0
**Last Updated:** 2025
**Model:** riffusion/riffusion-model-v1
**Default Settings:** 75 steps, guidance 8.5, strength 0.55
**Status:** Secondary backend (MusicGen primary)
