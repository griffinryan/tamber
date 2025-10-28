# Troubleshooting Guide

Common issues and solutions for Timbre development and usage.

---

## Setup & Installation

### Issue: `uv sync` fails with Python version error

**Error:**
```
Python 3.14 is not compatible. Requires Python 3.11.
```

**Solution:**
```bash
uv python pin 3.11
uv sync --project worker
```

### Issue: PyTorch not detecting MPS (Apple Silicon)

**Check:**
```python
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

**If False:**
- Ensure macOS 13+ on Apple Silicon
- Reinstall PyTorch: `uv sync --project worker --extra inference --reinstall`
- Check `TIMBRE_INFERENCE_DEVICE` not set to `cpu`

### Issue: Missing dependencies error

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Install inference dependencies
uv sync --project worker --extra inference

# Or development dependencies
uv sync --project worker --extra dev
```

---

## Worker/Backend Issues

### Issue: White noise or distorted audio on MPS

**Symptoms:** Generated audio sounds like static

**Cause:** MPS backend with incorrect dtype (float16 instead of float32)

**Solution:**
```bash
# Force CPU (temporary workaround)
export TIMBRE_INFERENCE_DEVICE=cpu
make worker-serve

# Or verify float32 is being used (check logs)
# Should see: "Using dtype torch.float32 on device mps"
```

### Issue: Worker returns placeholder audio

**Symptoms:** Audio is simple sine waves, not real music

**Cause:** Inference dependencies not installed or disabled

**Check:**
```bash
# Verify inference dependencies installed
uv pip list | grep -E '(torch|transformers|diffusers)'

# Check if Riffusion inference disabled
echo $TIMBRE_RIFFUSION_ALLOW_INFERENCE  # Should be 1 or empty (default enabled)
```

**Solution:**
```bash
# Install inference stack
uv sync --project worker --extra inference

# Enable Riffusion if disabled
unset TIMBRE_RIFFUSION_ALLOW_INFERENCE
# Or explicitly:
export TIMBRE_RIFFUSION_ALLOW_INFERENCE=1
```

### Issue: Slow generation (>5 minutes for 30s audio)

**Diagnosis:**
1. Check device in use:
   ```bash
   # Look for "device" in worker logs
   # Should see "mps" or "cuda", NOT "cpu"
   ```

2. Check model size:
   ```bash
   # Large models are slower
   # musicgen-large: ~10GB VRAM, slow
   # musicgen-medium: ~6GB VRAM, balanced
   # musicgen-small: ~3GB VRAM, fast
   ```

**Solutions:**
1. Force GPU:
   ```bash
   export TIMBRE_INFERENCE_DEVICE=mps  # or cuda
   ```

2. Use smaller model:
   ```bash
   # In CLI:
   /model musicgen-small
   ```

3. Accept CPU slowness (10-50x slower than GPU)

### Issue: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. Switch to smaller model:
   ```bash
   /model musicgen-small
   ```

2. Force CPU:
   ```bash
   export TIMBRE_INFERENCE_DEVICE=cpu
   ```

3. Free GPU memory:
   - Close other GPU applications
   - Restart worker

---

## CLI/TUI Issues

### Issue: CLI can't connect to worker

**Error:**
```
Failed to connect to worker at http://localhost:8000
```

**Check:**
1. Worker running?
   ```bash
   curl http://localhost:8000/health
   # Should return JSON with status
   ```

2. Correct URL?
   ```bash
   echo $TIMBRE_WORKER_URL  # Should match worker address
   ```

**Solutions:**
1. Start worker:
   ```bash
   make worker-serve
   ```

2. Set correct URL:
   ```bash
   export TIMBRE_WORKER_URL="http://localhost:8000"
   make cli-run
   ```

### Issue: UI rendering glitches

**Symptoms:** Garbled text, overlapping panes

**Causes:**
- Terminal too small
- Unsupported terminal emulator

**Solutions:**
1. Resize terminal (minimum 100x30)
2. Use supported terminal:
   - iTerm2 (macOS)
   - Alacritty
   - WezTerm
   - Terminal.app (macOS)

3. Force redraw: `Ctrl+L` (if implemented)

### Issue: Can't play audio

**Error:**
```
Failed to play audio: No audio output device found
```

**Solutions:**
1. Check audio device connected
2. Verify file exists:
   ```bash
   ls ~/Music/Timbre/<job_id>/
   ```

3. Manually play with external tool:
   ```bash
   ffplay ~/Music/Timbre/<job_id>/artifact.wav
   ```

---

## Audio Quality Issues

### Issue: Sections sound disconnected

**Symptoms:** Each section has different instruments, style, or energy

**Diagnosis:**
```python
# Check metadata
extras["sections"][i]["audio_conditioning_applied"]
# Should be True for all sections after MOTIF
```

**Causes:**
1. Motif seed not captured
2. Conditioning failed silently
3. Placeholder audio in use

**Solutions:**
1. Verify MOTIF section exists:
   ```python
   plan.sections[1].role == "MOTIF"  # Should be True
   ```

2. Check for conditioning errors:
   ```python
   extras["sections"][i].get("audio_conditioning_error")
   ```

3. Install inference dependencies (see above)

### Issue: Harsh clicks/pops between sections

**Symptoms:** Audible clicks at section boundaries

**Diagnosis:**
Check crossfade modes:
```python
extras["mix"]["crossfades"][i]["mode"]
# "butt" with both conditioned is normal
# "crossfade" with placeholders is normal
# "butt" WITHOUT conditioning is problematic
```

**Solutions:**
1. Ensure conditioning applied (see above issue)
2. If persistent, increase minimum crossfade:
   ```python
   # In orchestrator.py (requires code change)
   min_crossfade = 0.05  # 50ms minimum
   ```

### Issue: Audio too quiet or too loud

**Diagnosis:**
Check RMS levels:
```python
extras["mix"]["target_rms"]  # Should be 0.2
extras["mix"]["section_rms"]  # Per-section levels
```

**Causes:**
- RMS normalization not applied
- Soft limiter too aggressive/lenient

**Solutions:**
1. Verify mastering chain enabled (should be by default)
2. Check for errors in orchestrator logs
3. Manually normalize:
   ```bash
   ffmpeg -i input.wav -af loudnorm output.wav
   ```

### Issue: Dull, muffled sound

**Symptoms:** Lacks high frequencies, sounds muddy

**Diagnosis:**
Check if HF tilt applied:
```python
# Should be in orchestrator.py:generate()
waveform = tilt_highs(waveform, sample_rate)
```

**Solutions:**
1. Verify mastering chain (HF tilt should be automatic)
2. Increase tilt gain (requires code change):
   ```python
   waveform = tilt_highs(waveform, sample_rate, gain_db=5.0)  # vs default 2.5
   ```

---

## Development Issues

### Issue: Rust tests failing

**Error:**
```
test planner::test_long_form_plan ... FAILED
```

**Common Causes:**
1. Planner out of sync with Python
2. Constants changed
3. Test expectations outdated

**Solutions:**
1. Run sync validation:
   ```bash
   ./scripts/test_planner_sync.sh  # If exists
   ```

2. Update test expectations:
   ```rust
   // Match current output
   assert_eq!(plan.tempo_bpm, 120);  // Update expected value
   ```

3. See docs/PLANNER_SYNC.md for synchronization protocol

### Issue: Python tests failing

**Error:**
```
FAILED tests/test_planner.py::test_long_form_plan
```

**Solutions:**
1. Ensure dev dependencies installed:
   ```bash
   uv sync --project worker --extra dev
   ```

2. Run specific test:
   ```bash
   cd worker
   uv run pytest tests/test_planner.py::test_long_form_plan -v
   ```

3. Update test expectations if planner changed intentionally

### Issue: Linting errors

**Rust:**
```bash
cargo fmt --check  # Format check
cargo clippy -- -D warnings  # Linting

# Auto-fix formatting:
cargo fmt
```

**Python:**
```bash
cd worker
uv run ruff check src tests  # Linting
uv run ruff format src tests  # Formatting
uv run mypy  # Type checking
```

---

## Performance Issues

### Issue: High memory usage

**Symptoms:** Worker using >16GB RAM

**Causes:**
- Multiple models cached
- Large job queue
- Memory leak (rare)

**Solutions:**
1. Restart worker periodically
2. Limit concurrent jobs
3. Use smaller models

### Issue: Slow UI responsiveness

**Symptoms:** CLI feels laggy, delayed input

**Causes:**
- Blocking operations in event loop
- Too many chat entries
- Heavy rendering

**Solutions:**
1. Clear chat history: `/reset` (if implemented)
2. Limit jobs displayed
3. Report issue (may be bug)

---

## Debugging Tips

### Enable Verbose Logging

**Worker:**
```bash
# Set log level
export LOGURU_LEVEL=DEBUG
make worker-serve
```

**CLI:**
```bash
# Rust logging (if implemented)
export RUST_LOG=debug
make cli-run
```

### Inspect Metadata

**After generation:**
```bash
# View metadata JSON
cat ~/Music/Timbre/<job_id>/metadata.json | jq .

# Check specific fields
jq '.extras.sections[].audio_conditioning_applied' metadata.json
jq '.extras.mix.crossfades' metadata.json
```

### Test Components Independently

**Worker:**
```bash
# Test planner
uv run --project worker python -m timbre_worker.services.planner

# Test generation
uv run --project worker python -m timbre_worker.generate --prompt "test" --duration 10
```

**Rust:**
```bash
# Test planner
cargo test planner::

# Test specific module
cargo test --package timbre-cli --lib types
```

---

## Getting Help

### Before Reporting Issues

1. ✅ Check this troubleshooting guide
2. ✅ Review relevant documentation:
   - docs/setup.md
   - docs/MUSICGEN.md
   - docs/CONDITIONING.md
3. ✅ Check existing issues: https://github.com/your-repo/issues
4. ✅ Gather diagnostic information:
   - Worker logs
   - CLI output
   - Metadata JSON
   - System info (OS, Python version, Rust version)

### Reporting Issues

Include:
- **Timbre version:** `git rev-parse HEAD`
- **Environment:**
  - OS: macOS 13.5, Ubuntu 22.04, etc.
  - Python: `python --version`
  - Rust: `rustc --version`
  - Device: CPU, MPS, CUDA
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Logs** (relevant excerpts, not full logs unless requested)
- **Metadata** (if audio quality issue)

### Useful Commands

```bash
# System info
uname -a
python --version
rustc --version
cargo --version

# Timbre status
make test  # Run all tests
make lint  # Check code quality

# Health check
curl http://localhost:8000/health | jq .

# View logs (if using systemd/docker)
journalctl -u timbre-worker -f
docker logs timbre-worker --follow
```

---

## Known Issues & Limitations

### MusicGen
- Max ~29.5s per generation call
- Large models require significant VRAM
- CPU inference very slow (10-50x)

### Riffusion
- Griffin-Lim phase reconstruction artifacts
- MPS requires float32 (slower than float16)
- Best for ambient/textural music

### CLI
- Mouse support limited
- Some terminals not fully supported
- Playback quality depends on rodio/cpal

### General
- Long-form compositions (180s) take 3-10 minutes
- First generation slower (model loading)
- Placeholder audio when dependencies missing

---

**Document Version:** 1.0
**Last Updated:** 2025
**Covers:** Timbre v0.x (Phase 0)
