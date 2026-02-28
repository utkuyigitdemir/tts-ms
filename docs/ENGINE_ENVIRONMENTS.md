# TTS Engine Environments

This document provides comprehensive setup instructions for running all 9 TTS engines across multiple conda environments. Each engine has specific Python version and dependency requirements.

## Quick Start: Automatic Setup

The tts-ms CLI includes built-in engine setup commands:

```bash
# Check status of all engines
tts-ms --engines

# Setup a specific engine (checks requirements)
tts-ms --setup piper

# Setup with automatic pip package installation
tts-ms --setup f5tts --auto-install

# Enable auto-install via environment variable
export TTS_MS_AUTO_INSTALL=1
```

When starting the server, the engine factory automatically checks requirements and provides clear error messages with fix instructions if something is missing.

## Quick Reference

| Environment | Python | Engines | Status |
|-------------|--------|---------|--------|
| `tts` | 3.12 | Piper, F5-TTS, StyleTTS2, Legacy XTTS v2, Kokoro, Qwen3-TTS, VibeVoice | Primary development env |
| `tts311` | 3.11 | Chatterbox | Required for numpy compatibility |
| `tts310` | 3.10 | CosyVoice | Required for CosyVoice dependencies |

## Engine Overview

| Engine | Python | PyTorch | GPU Required | Test Time (CPU) | Notes |
|--------|--------|---------|--------------|-----------------|-------|
| Piper | 3.12 | N/A | No | ~1.4s | CPU-only, fastest |
| F5-TTS | 3.12 | 2.5.0 | Recommended | ~77s | Voice cloning, needs ffmpeg |
| StyleTTS2 | 3.12 | 2.5.0 | Recommended | ~21s | Needs espeak-ng |
| Legacy XTTS v2 | 3.12 | 2.5.0 | Recommended | ~4s | Uses coqui-tts fork |
| Chatterbox | 3.11 | 2.6.0 | Recommended | ~77s | numpy<1.26 constraint |
| CosyVoice | 3.10 | Latest | Recommended | TBD | Manual git installation |
| Kokoro | 3.12 | N/A | No | ~2s | CPU-only ONNX, preset voices |
| Qwen3-TTS | 3.12 | Latest | Recommended | TBD | Alibaba, preset + voice cloning |
| VibeVoice | 3.12 | Latest | Required | TBD | Microsoft, research-only |

---

## Environment Setup

### Environment 1: `tts` (Python 3.12)

Primary development environment for most engines.

```bash
# Create environment
conda create -n tts python=3.12 -y
conda activate tts

# Install PyTorch (CPU version for development)
pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# Install tts-ms with dev dependencies
cd /path/to/tts-ms
pip install -e ".[dev]"

# Install engine-specific packages
pip install piper-tts          # Piper
pip install f5-tts             # F5-TTS
pip install styletts2          # StyleTTS2
pip install coqui-tts          # Legacy XTTS v2

# Download NLTK data for StyleTTS2
python -c "import nltk; nltk.download('punkt_tab')"
```

### Environment 2: `tts311` (Python 3.11)

Required for Chatterbox due to numpy compatibility issues.

```bash
# Create environment
conda create -n tts311 python=3.11 -y
conda activate tts311

# Install PyTorch
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install tts-ms
cd /path/to/tts-ms
pip install -e ".[dev]"

# Install Chatterbox
pip install chatterbox-tts
```

### Environment 3: `tts310` (Python 3.10)

Required for CosyVoice which has strict Python version requirements.

```bash
# Create environment
conda create -n tts310 python=3.10 -y
conda activate tts310

# Install PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Clone CosyVoice repository
git clone https://github.com/FunAudioLLM/CosyVoice ~/CosyVoice
cd ~/CosyVoice

# Install CosyVoice dependencies
pip install -r requirements.txt

# Install tts-ms
cd /path/to/tts-ms
pip install -e ".[dev]"
```

---

## Per-Engine Details

### 1. Piper (Environment: `tts`)

**The simplest engine - CPU-only, no external dependencies.**

#### Installation
```bash
conda activate tts
pip install piper-tts
```

#### System Dependencies
None required.

#### Test Command
```bash
TTS_MODEL_TYPE=piper pytest -v tests/test_engine_piper_smoke.py -m slow
```

#### Expected Output
```
tests/test_engine_piper_smoke.py::test_piper_engine_synthesis PASSED
================================ 1 passed in 1.42s ================================
```

---

### 2. F5-TTS (Environment: `tts`)

**Voice cloning model - requires reference audio for synthesis.**

#### Installation
```bash
conda activate tts
pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install f5-tts
```

#### System Dependencies
- **ffmpeg**: Required for audio processing

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

**Windows (using imageio-ffmpeg):**
```bash
# imageio-ffmpeg is already included in tts-ms dependencies
# Copy ffmpeg binary to conda environment
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
# Copy the printed path's binary to your conda env's Scripts folder
```

#### Test Command
```bash
TTS_MODEL_TYPE=f5tts pytest -v tests/test_engine_f5tts_smoke.py -m slow
```

#### Known Issues
- **torchaudio 2.9+ on Windows**: Has torchcodec DLL loading issues. Use torchaudio 2.5.0.
- **No default speakers**: F5-TTS is a voice cloning model. Reference audio is generated via edge-tts for testing.

---

### 3. StyleTTS2 (Environment: `tts`)

**High-quality TTS with style transfer capabilities.**

#### Installation
```bash
conda activate tts
pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install styletts2

# Download required NLTK data
python -c "import nltk; nltk.download('punkt_tab')"
```

#### System Dependencies
- **espeak-ng**: Required for phonemization

**Linux:**
```bash
sudo apt-get install espeak-ng
```

**macOS:**
```bash
brew install espeak-ng
```

**Windows (without admin rights):**
```powershell
# Download MSI from https://github.com/espeak-ng/espeak-ng/releases
# Extract without installation:
msiexec /a espeak-ng-X64.msi /qn TARGETDIR=C:\Users\%USERNAME%\espeak-ng

# Set environment variable
$env:ESPEAK_DATA_PATH = "C:\Users\$env:USERNAME\espeak-ng\eSpeak NG\espeak-ng-data"

# Or add to system PATH permanently
[Environment]::SetEnvironmentVariable("ESPEAK_DATA_PATH", "C:\Users\$env:USERNAME\espeak-ng\eSpeak NG\espeak-ng-data", "User")
```

#### Test Command
```bash
TTS_MODEL_TYPE=styletts2 pytest -v tests/test_engine_styletts2_smoke.py -m slow
```

#### Known Issues
- **PyTorch 2.6+ checkpoint loading**: Changed default `weights_only=True`. StyleTTS2 package may need updates for compatibility.

---

### 4. Legacy XTTS v2 (Environment: `tts`)

**Coqui TTS fork with built-in multilingual support.**

#### Installation
```bash
conda activate tts
pip install coqui-tts
```

Note: The original `TTS` package doesn't support Python 3.12. Use the `coqui-tts` fork instead.

#### System Dependencies
None required (models download automatically).

#### Test Command
```bash
# Manual test (no dedicated smoke test file)
TTS_MODEL_TYPE=legacy python -c "
from tts_ms.tts.engine import _create_engine
engine = _create_engine('legacy')
engine.load()
print('Legacy XTTS v2 loaded successfully!')
"
```

#### Notes
- Built-in Turkish support with multiple speaker options
- Large model download on first run (~2GB)

---

### 5. Chatterbox (Environment: `tts311`)

**Requires Python 3.11 due to numpy compatibility issues.**

#### Installation
```bash
conda activate tts311
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install chatterbox-tts
pip install -e /path/to/tts-ms
```

#### System Dependencies
None required.

#### Test Command
```bash
conda activate tts311
TTS_MODEL_TYPE=chatterbox pytest -v tests/test_engine_chatterbox_smoke.py -m slow
```

#### Expected Output
```
tests/test_engine_chatterbox_smoke.py::test_chatterbox_engine_load PASSED
tests/test_engine_chatterbox_smoke.py::test_chatterbox_engine_synthesis PASSED
tests/test_engine_chatterbox_smoke.py::test_chatterbox_engine_with_reference PASSED
================================ 3 passed in 77.42s ================================
```

#### Known Issues
- **numpy<1.26 requirement**: Incompatible with Python 3.12's numpy
- **CPU support**: Engine code patched with `map_location` for CPU inference

---

### 6. CosyVoice (Environment: `tts310`)

**Requires Python 3.10 and manual installation from source.**

#### Installation
```bash
conda activate tts310

# Install PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Clone repository WITH submodules (Matcha-TTS is required)
git clone --recurse-submodules https://github.com/FunAudioLLM/CosyVoice ~/CosyVoice
cd ~/CosyVoice

# Install setuptools first (required by openai-whisper's pkg_resources)
pip install setuptools

# Install ruamel.yaml separately (HyperPyYAML compatibility)
pip install "ruamel.yaml<0.19"

# Install dependencies (some packages may need to be stripped on certain platforms)
# On CPU-only systems, remove these from requirements.txt first:
#   tensorrt, deepspeed, onnxruntime-gpu, extra-index-url lines
pip install -r requirements.txt

# Install onnxruntime separately for CPU
pip install onnxruntime==1.18.0

# Set PYTHONPATH (required for CosyVoice imports)
export PYTHONPATH=~/CosyVoice:~/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

# Install tts-ms
pip install -e /path/to/tts-ms
```

#### System Dependencies
- **sox** (Linux only):
```bash
sudo apt-get install sox libsox-dev
```

#### Test Command
```bash
conda activate tts310
export PYTHONPATH=~/CosyVoice:~/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
TTS_MODEL_TYPE=cosyvoice pytest -v tests/test_engine_cosyvoice_smoke.py -m slow
```

#### Known Issues
- **No pip package**: Must install from git repository
- **Submodules required**: `--recurse-submodules` is mandatory for Matcha-TTS
- **PYTHONPATH**: Must include both CosyVoice root and Matcha-TTS third_party
- **grpcio on Windows**: May require building from source
- **Large model download**: ~10GB for full model weights
- **requirements.txt**: May contain GPU-only packages (tensorrt, deepspeed) that need to be removed on CPU systems

---

### 7. Kokoro (Environment: `tts`)

**CPU-only ONNX engine with multiple preset voices.**

#### Installation
```bash
conda activate tts
pip install kokoro-onnx huggingface-hub

# Download model files
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('onnx-community/Kokoro-82M-v1.0-ONNX', 'kokoro-v1.0.onnx', local_dir='models/kokoro'); \
  hf_hub_download('onnx-community/Kokoro-82M-v1.0-ONNX', 'voices-v1.0.bin', local_dir='models/kokoro')"
```

#### System Dependencies
None required.

#### Test Command
```bash
TTS_MODEL_TYPE=kokoro pytest -v tests/test_engine_kokoro_smoke.py -m slow
```

#### Notes
- No GPU required (ONNX Runtime, CPU inference)
- Model files are ~300MB total
- Preset voices: af_sarah, af_heart, am_adam, etc.

---

### 8. Qwen3-TTS (Environment: `tts`)

**Alibaba's TTS model with preset speakers and voice cloning.**

#### Installation
```bash
conda activate tts
pip install torch torchaudio qwen-tts
```

#### System Dependencies
None required.

#### Test Command
```bash
TTS_MODEL_TYPE=qwen3tts pytest -v tests/test_engine_qwen3tts_smoke.py -m slow
```

#### Notes
- GPU recommended (~3-4 GB VRAM for 0.6B model)
- Preset speakers: Vivian, Serena, Ethan, Chelsie
- Supports voice cloning from reference audio
- Model auto-downloads from HuggingFace

---

### 9. VibeVoice (Environment: `tts`)

**Microsoft's research TTS model with multiple preset speakers.**

#### Installation
```bash
conda activate tts
pip install torch torchaudio vibevoice
```

#### System Dependencies
- **ffmpeg**: Required for audio processing

#### Test Command
```bash
TTS_MODEL_TYPE=vibevoice pytest -v tests/test_engine_vibevoice_smoke.py -m slow
```

#### Known Issues
- **Research-only license**: Check Microsoft's license terms before commercial deployment
- GPU required (~7 GB VRAM for 1.5B model)

---

## System Dependencies Summary

### ffmpeg

| OS | Installation |
|----|--------------|
| Ubuntu/Debian | `sudo apt-get install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Windows | Use imageio-ffmpeg or `choco install ffmpeg` |

### espeak-ng

| OS | Installation |
|----|--------------|
| Ubuntu/Debian | `sudo apt-get install espeak-ng` |
| macOS | `brew install espeak-ng` |
| Windows | Extract MSI with `msiexec /a`, set `ESPEAK_DATA_PATH` |

### sox (CosyVoice only)

| OS | Installation |
|----|--------------|
| Ubuntu/Debian | `sudo apt-get install sox libsox-dev` |
| macOS | `brew install sox` |
| Windows | Not typically required |

---

## Running All Tests

### Quick Test (All Engines)

```bash
# Environment: tts (Python 3.12)
conda activate tts
TTS_MODEL_TYPE=piper pytest -v tests/test_engine_piper_smoke.py -m slow
TTS_MODEL_TYPE=f5tts pytest -v tests/test_engine_f5tts_smoke.py -m slow
TTS_MODEL_TYPE=styletts2 pytest -v tests/test_engine_styletts2_smoke.py -m slow
TTS_MODEL_TYPE=kokoro pytest -v tests/test_engine_kokoro_smoke.py -m slow
TTS_MODEL_TYPE=qwen3tts pytest -v tests/test_engine_qwen3tts_smoke.py -m slow
TTS_MODEL_TYPE=vibevoice pytest -v tests/test_engine_vibevoice_smoke.py -m slow

# Environment: tts311 (Python 3.11)
conda activate tts311
TTS_MODEL_TYPE=chatterbox pytest -v tests/test_engine_chatterbox_smoke.py -m slow

# Environment: tts310 (Python 3.10)
conda activate tts310
TTS_MODEL_TYPE=cosyvoice pytest -v tests/test_engine_cosyvoice_smoke.py -m slow
```

### Full Test Suite (Piper - CI Recommended)

```bash
conda activate tts
TTS_MODEL_TYPE=piper pytest -v
# Expected: 470 passed, 10 skipped
```

---

## Troubleshooting

### Windows ffmpeg Issues

**Problem:** F5-TTS fails with "ffmpeg not found"

**Solution:** Use imageio-ffmpeg bundled binary:
```python
import imageio_ffmpeg
import shutil
import os

# Get ffmpeg path
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

# Copy to conda Scripts folder
conda_scripts = os.path.join(os.environ['CONDA_PREFIX'], 'Scripts')
shutil.copy(ffmpeg_path, os.path.join(conda_scripts, 'ffmpeg.exe'))
```

### Windows espeak-ng Issues

**Problem:** StyleTTS2 fails with "espeak-ng not found"

**Solution:** Extract MSI without admin rights:
```powershell
# Download MSI from GitHub releases
msiexec /a espeak-ng-X64.msi /qn TARGETDIR=C:\Users\%USERNAME%\espeak-ng

# Set environment variable (PowerShell)
$env:ESPEAK_DATA_PATH = "C:\Users\$env:USERNAME\espeak-ng\eSpeak NG\espeak-ng-data"
```

### torchaudio 2.9+ DLL Issues (Windows)

**Problem:** `torchcodec` DLL loading errors on Windows

**Solution:** Pin torchaudio to 2.5.0:
```bash
pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
```

### Chatterbox CPU Inference

**Problem:** Chatterbox fails to load on CPU-only machine

**Solution:** Engine code already includes fix:
```python
# In chatterbox_engine.py
torch.load(..., map_location=torch.device('cpu'))
```

### CosyVoice grpcio Build Failures (Windows)

**Problem:** grpcio fails to build from source

**Solution:** Use pre-built wheel or try:
```bash
pip install grpcio --only-binary=:all:
```

---

## Verification Checklist

Use this checklist to verify all engines are working:

- [ ] **tts environment created** (Python 3.12)
- [ ] **tts311 environment created** (Python 3.11)
- [ ] **tts310 environment created** (Python 3.10)
- [ ] **Piper test passes** (tts env)
- [ ] **F5-TTS test passes** (tts env, requires ffmpeg)
- [ ] **StyleTTS2 test passes** (tts env, requires espeak-ng)
- [ ] **Legacy XTTS v2 loads** (tts env)
- [ ] **Chatterbox test passes** (tts311 env)
- [ ] **CosyVoice test passes** (tts310 env)
- [ ] **Kokoro test passes** (tts env, requires model files)
- [ ] **Qwen3-TTS test passes** (tts env, requires GPU)
- [ ] **VibeVoice test passes** (tts env, requires GPU + ffmpeg)

---

## Related Documentation

- [ENGINE_REQUIREMENTS.md](ENGINE_REQUIREMENTS.md) - Quick reference for test skip reasons
- [DOCUMENTATION.md](DOCUMENTATION.md) - Complete user manual
- [WORKFLOW.md](WORKFLOW.md) - Technical architecture
