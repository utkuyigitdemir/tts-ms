# Engine-Specific Test Requirements

This document describes the system dependencies required to run engine-specific smoke tests. These tests are skipped when dependencies are not available.

## Test Status Summary

| Engine | Test File | Status | Dependency |
|--------|-----------|--------|------------|
| Piper | `test_engine_piper_smoke.py` | ✅ Pass | None (CPU-only) |
| F5-TTS | `test_engine_f5tts_smoke.py` | ⚠️ Skip | ffmpeg |
| StyleTTS2 | `test_engine_styletts2_smoke.py` | ⚠️ Skip | espeak-ng |
| CosyVoice | `test_engine_cosyvoice_smoke.py` | ⚠️ Skip | Manual installation |
| Chatterbox | `test_engine_chatterbox_smoke.py` | ⚠️ Skip | Python 3.11 |

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
- CosyVoice requires significant VRAM (8GB+ recommended)
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

## Running Engine-Specific Tests

To run a specific engine's smoke test:

```bash
# Set the engine type and run
TTS_MODEL_TYPE=piper python -m pytest tests/test_engine_piper_smoke.py -v
TTS_MODEL_TYPE=f5tts python -m pytest tests/test_engine_f5tts_smoke.py -v
TTS_MODEL_TYPE=styletts2 python -m pytest tests/test_engine_styletts2_smoke.py -v
TTS_MODEL_TYPE=cosyvoice python -m pytest tests/test_engine_cosyvoice_smoke.py -v
TTS_MODEL_TYPE=chatterbox python -m pytest tests/test_engine_chatterbox_smoke.py -v
```

## Full Test Suite

To run all tests (engine-specific tests will skip if dependencies unavailable):

```bash
# With Piper (recommended for CI)
TTS_MODEL_TYPE=piper python -m pytest -v

# Results: 446 passed, 4 skipped
```
