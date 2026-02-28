# Engine-Specific Test Requirements

This document describes the system dependencies required to run engine-specific smoke tests. These tests are skipped when dependencies are not available.

> **For comprehensive setup instructions**, see [ENGINE_ENVIRONMENTS.md](ENGINE_ENVIRONMENTS.md) which covers:
> - Multi-environment setup (Python 3.10, 3.11, 3.12)
> - Detailed per-engine installation steps
> - System dependency installation for all platforms
> - Troubleshooting common issues

## Test Status Summary

| Engine | Test File | Status | Dependency |
|--------|-----------|--------|------------|
| Piper | `test_engine_piper_smoke.py` | ✅ Pass | None (CPU-only) |
| F5-TTS | `test_engine_f5tts_smoke.py` | ⚠️ Skip | ffmpeg |
| StyleTTS2 | `test_engine_styletts2_smoke.py` | ⚠️ Skip | espeak-ng |
| Legacy XTTS v2 | *(no dedicated smoke test)* | ⚠️ Manual | coqui-tts, model download |
| CosyVoice | `test_engine_cosyvoice_smoke.py` | ⚠️ Skip | Manual installation |
| Chatterbox | `test_engine_chatterbox_smoke.py` | ⚠️ Skip | Python 3.11 |
| Kokoro | `test_engine_kokoro_smoke.py` | ⚠️ Skip | kokoro-onnx + model files |
| Qwen3-TTS | `test_engine_qwen3tts_smoke.py` | ⚠️ Skip | GPU + qwen-tts |
| VibeVoice | `test_engine_vibevoice_smoke.py` | ⚠️ Skip | GPU + vibevoice |

---

## F5-TTS

**Skip Reason:** `F5-TTS requires working ffmpeg installation`

F5-TTS uses ffmpeg for audio processing and Whisper for reference audio transcription.

### Requirements
- ffmpeg and ffprobe binaries
- Working pydub audio processing

### Installation

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

**Windows:**
```powershell
# Via Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH
```

### Known Issues
- Windows conda ffmpeg package may have DLL loading issues
- Ensure ffprobe is also available in PATH

---

## StyleTTS2

**Skip Reason:** `StyleTTS2 requires espeak-ng for phonemization`

StyleTTS2 uses espeak-ng as the phonemizer backend for text-to-phoneme conversion.

### Requirements
- espeak-ng binary
- PyTorch < 2.6 (or compatible checkpoint format)

### Installation

**Linux:**
```bash
sudo apt-get install espeak-ng
```

**macOS:**
```bash
brew install espeak-ng
```

**Windows:**
```powershell
# Via Chocolatey
choco install espeak-ng

# Or download installer from:
# https://github.com/espeak-ng/espeak-ng/releases
```

### Known Issues
- PyTorch 2.6+ changed `torch.load` default `weights_only=True`, causing checkpoint loading failures
- Workaround: Downgrade PyTorch or wait for styletts2 library update

---

## CosyVoice

**Skip Reason:** `could not import 'cosyvoice': No module named 'cosyvoice'`

CosyVoice is not available as a pip package and requires manual installation from the repository.

### Requirements
- Manual clone and installation from GitHub
- Model weights download

### Installation

```bash
# Clone the repository
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Install dependencies
pip install -r requirements.txt

# Download model weights
# Follow instructions in CosyVoice README
```

### Notes
- CosyVoice requires significant VRAM (6-8 GB recommended)
- Model weights are large (~10GB)

---

## Chatterbox

**Skip Reason:** `Chatterbox requires Python 3.11 (numpy incompatibility with Python 3.12)`

Chatterbox has numpy build issues on Python 3.12 due to deprecated APIs.

### Requirements
- Python 3.11 environment
- CUDA recommended (slow on CPU)

### Installation

```bash
# Create Python 3.11 environment
conda create -n tts-chatterbox python=3.11 -y
conda activate tts-chatterbox

# Install chatterbox
pip install chatterbox-tts

# Run tests
TTS_MODEL_TYPE=chatterbox python -m pytest tests/test_engine_chatterbox_smoke.py -v
```

### Verified Working
Chatterbox was successfully tested in Python 3.11 environment:
```
Python version: 3.11.14
ChatterboxTTS imported successfully!
Model loaded successfully!
Generated audio shape: torch.Size([1, 43200])
Chatterbox smoke test PASSED!
```

---

## Kokoro

**Skip Reason:** `Kokoro requires kokoro-onnx package and model files`

Kokoro uses ONNX Runtime for CPU-based inference with preset voices.

### Requirements
- kokoro-onnx pip package
- Model files from HuggingFace (~300MB)

### Installation

```bash
pip install kokoro-onnx huggingface-hub

# Download model files
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('onnx-community/Kokoro-82M-v1.0-ONNX', 'kokoro-v1.0.onnx', local_dir='models/kokoro'); \
  hf_hub_download('onnx-community/Kokoro-82M-v1.0-ONNX', 'voices-v1.0.bin', local_dir='models/kokoro')"
```

### Notes
- CPU-only, no GPU required
- Multiple preset voices available (af_sarah, am_adam, etc.)

---

## Qwen3-TTS

**Skip Reason:** `Qwen3-TTS requires GPU and qwen-tts package`

Qwen3-TTS is Alibaba's TTS model with preset speakers and voice cloning.

### Requirements
- GPU with CUDA support (~3-4 GB VRAM)
- qwen-tts pip package

### Installation

```bash
pip install torch torchaudio qwen-tts
```

### Notes
- Preset speakers: Vivian, Serena, Ethan, Chelsie
- Supports voice cloning from reference audio
- Model auto-downloads from HuggingFace on first run

---

## VibeVoice

**Skip Reason:** `VibeVoice requires GPU and vibevoice package`

VibeVoice is Microsoft's research TTS model with multiple preset speakers.

### Requirements
- GPU with CUDA support (~7 GB VRAM)
- vibevoice pip package
- ffmpeg

### Installation

```bash
pip install vibevoice
```

### Notes
- Research-only license - check Microsoft's terms before commercial use
- Preset speakers: Alice, Frank, Mary, Carter, Enrique, Eric
- Model auto-downloads from HuggingFace on first run

---

## Running Engine-Specific Tests

To run a specific engine's smoke test:

```bash
# Set the engine type and run
TTS_MODEL_TYPE=piper python -m pytest tests/test_engine_piper_smoke.py -v
TTS_MODEL_TYPE=f5tts python -m pytest tests/test_engine_f5tts_smoke.py -v
TTS_MODEL_TYPE=styletts2 python -m pytest tests/test_engine_styletts2_smoke.py -v
TTS_MODEL_TYPE=cosyvoice python -m pytest tests/test_engine_cosyvoice_smoke.py -v
TTS_MODEL_TYPE=chatterbox python -m pytest tests/test_engine_chatterbox_smoke.py -v
TTS_MODEL_TYPE=kokoro python -m pytest tests/test_engine_kokoro_smoke.py -v
TTS_MODEL_TYPE=qwen3tts python -m pytest tests/test_engine_qwen3tts_smoke.py -v
TTS_MODEL_TYPE=vibevoice python -m pytest tests/test_engine_vibevoice_smoke.py -v
```

## Full Test Suite

To run all tests (engine-specific tests will skip if dependencies unavailable):

```bash
# With Piper (recommended for CI)
TTS_MODEL_TYPE=piper python -m pytest -v

# Results: 470 passed, 10 skipped
```
