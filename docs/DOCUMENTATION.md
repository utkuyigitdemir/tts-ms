# TTS-MS Documentation

**Turkish Text-to-Speech Microservice**

A production-grade, multi-engine TTS microservice designed for low-latency, real-time speech synthesis with Turkish language optimization.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [API Reference](#4-api-reference)
5. [CLI Reference](#5-cli-reference)
6. [Configuration Guide](#6-configuration-guide)
7. [TTS Engines](#7-tts-engines)
8. [Docker Deployment](#8-docker-deployment)
9. [Integration Examples](#9-integration-examples)
10. [Caching and Performance](#10-caching-and-performance)
11. [Monitoring and Observability](#11-monitoring-and-observability)
12. [Troubleshooting](#12-troubleshooting)
13. [FAQ](#13-faq)

---

## 1. Introduction

### 1.1 Overview

TTS-MS is a high-performance text-to-speech microservice that provides:

- **Multi-Engine Support** - Choose from 9 different TTS engines (Piper, F5-TTS, CosyVoice, StyleTTS2, Chatterbox, XTTS v2, Kokoro, Qwen3-TTS, VibeVoice)
- **Unified API** - Single `/v1/tts` endpoint works with any engine
- **OpenAI Compatibility** - Drop-in replacement via `/v1/audio/speech`
- **Real-time Streaming** - Server-Sent Events for chunk-by-chunk audio delivery
- **Turkish Optimization** - Native Turkish language support with breath-group chunking
- **Production Ready** - Two-tier caching, concurrency control, Prometheus metrics

### 1.2 Core Principles

| Principle | Description |
|-----------|-------------|
| **Engine Agnostic** | Swap engines without changing client code |
| **One Model Per Run** | Static engine loading prevents VRAM fragmentation |
| **Low Latency** | Sentence-level caching enables near-instant re-generation |
| **Observability** | Structured logging, metrics, and resource monitoring |

### 1.3 Supported Engines

| Engine | GPU Required | Voice Cloning | Turkish Support | Best For |
|--------|--------------|---------------|-----------------|----------|
| Piper | No (CPU) | No | tr_TR model | Fast, lightweight |
| XTTS v2 (Legacy) | Yes | Yes | Native | General purpose |
| F5-TTS | Yes | Yes (required) | Via reference | High-quality cloning |
| CosyVoice | Yes | Yes | Native | Natural prosody |
| StyleTTS2 | Yes | No | Limited | Style control |
| Chatterbox | Yes | Yes | Native | Expressive speech |
| Kokoro | No (CPU/ONNX) | No | No | CPU-only, preset voices |
| Qwen3-TTS | Yes (recommended) | Yes | Auto-detect | Preset voices + cloning |
| VibeVoice | Yes (required) | Yes | Auto-detect | Research, high-quality |

---

## 2. Installation

### 2.1 Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for GPU-based engines
- (Optional) Docker for containerized deployment

### 2.2 Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/tts-ms.git
cd tts-ms

# Install in editable mode (recommended for development)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with GPU monitoring support
pip install -e ".[gpu]"
```

### 2.3 Install Dependencies by Engine

Each engine has specific dependencies:

```bash
# Piper (CPU-only, lightweight)
pip install -e ".[piper]"

# XTTS v2 / Legacy (GPU recommended)
pip install -e ".[legacy]"

# F5-TTS (GPU required)
pip install -e ".[f5tts]"

# CosyVoice (GPU required)
pip install -e ".[cosyvoice]"

# StyleTTS2 (GPU required)
pip install -e ".[styletts2]"

# Chatterbox (GPU required)
pip install -e ".[chatterbox]"

# Kokoro (CPU-only, ONNX)
pip install -e ".[kokoro]"

# Qwen3-TTS (GPU recommended)
pip install -e ".[qwen3tts]"

# VibeVoice (GPU required)
pip install -e ".[vibevoice]"
```

### 2.4 Verify Installation

```bash
# Check CLI is available
tts-ms --help

# Check server can start
python -m uvicorn tts_ms.main:app --host 127.0.0.1 --port 8000
```

---

## 3. Quick Start

### 3.1 Start the Server

```bash
# Set engine type (required)
export TTS_MODEL_TYPE=piper

# Start server
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

### 3.2 Synthesize Speech (cURL)

```bash
# Basic synthesis
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba, nasılsınız?", "language": "tr"}' \
  --output speech.wav

# Play the audio (Linux)
aplay speech.wav

# Play the audio (macOS)
afplay speech.wav

# Play the audio (Windows)
start speech.wav
```

### 3.3 Synthesize Speech (CLI)

```bash
# Simple synthesis
tts-ms "Merhaba dünya" --out hello.wav

# With language and speaker
tts-ms --text "Günaydın!" --language tr --out greeting.wav
```

### 3.4 Check Server Health

```bash
curl http://localhost:8000/health | jq
```

Expected response:
```json
{
  "ok": true,
  "warmed_up": true,
  "engine": "piper",
  "device": "cpu"
}
```

---

## 4. API Reference

### 4.1 POST /v1/tts - Native TTS Endpoint

Synthesize text to speech using the configured TTS engine.

**Request**

```http
POST /v1/tts
Content-Type: application/json
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize (1-4000 chars) |
| `speaker` | string | No | Speaker/voice identifier |
| `language` | string | No | Language code (e.g., "tr", "en") |
| `split_sentences` | boolean | No | Split text at sentence boundaries |
| `speaker_wav_b64` | string | No | Base64-encoded reference audio for voice cloning (max 10MB) |

**Example Request**

```json
{
  "text": "Merhaba, bugün hava çok güzel.",
  "language": "tr",
  "speaker": "default"
}
```

**Success Response (200 OK)**

- Content-Type: `audio/wav`
- Body: WAV audio bytes

| Header | Description |
|--------|-------------|
| `X-Request-Id` | Unique request identifier |
| `X-Sample-Rate` | Audio sample rate (e.g., 22050, 24000) |
| `X-Bytes` | Size of audio data in bytes |

**Error Response**

```json
{
  "ok": false,
  "error": "INVALID_INPUT",
  "message": "Text is required",
  "details": {}
}
```

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | INVALID_INPUT | Invalid request parameters |
| 408 | TIMEOUT | Synthesis timeout |
| 500 | SYNTHESIS_FAILED | Engine synthesis error |
| 503 | QUEUE_FULL | Too many concurrent requests |
| 503 | MODEL_NOT_READY | Engine still warming up |

---

### 4.2 POST /v1/audio/speech - OpenAI-Compatible Endpoint

Drop-in replacement for OpenAI's TTS API.

**Request**

```http
POST /v1/audio/speech
Content-Type: application/json
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | "tts-1" | Model name (ignored, uses configured engine) |
| `input` | string | Yes | - | Text to synthesize (1-4096 chars) |
| `voice` | string | No | "alloy" | Voice: alloy, echo, fable, onyx, nova, shimmer |
| `response_format` | string | No | "wav" | Format: wav, mp3, opus, aac, flac, pcm (**Note:** currently only WAV is fully supported; other formats are accepted but return WAV audio) |
| `speed` | number | No | 1.0 | Speed multiplier (0.25-4.0) (**Note:** accepted but not yet implemented; audio is always generated at 1.0x speed) |

**Example Request**

```json
{
  "model": "tts-1",
  "input": "Hello, how are you today?",
  "voice": "nova",
  "response_format": "wav",
  "speed": 1.0
}
```

**Success Response (200 OK)**

- Content-Type: `audio/wav`
- Body: WAV audio bytes

| Header | Description |
|--------|-------------|
| `X-Request-Id` | Unique request identifier |
| `X-Engine` | TTS engine used |
| `X-Voice-Mapped-To` | Internal speaker ID |

**Error Response (OpenAI Format)**

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "invalid_input"
  }
}
```

---

### 4.3 POST /v1/tts/stream - Streaming Endpoint

Stream audio chunks via Server-Sent Events (SSE).

**Request**

Same as `/v1/tts`

**Response (SSE Stream)**

```
event: meta
data: {"request_id": "abc123", "sample_rate": 22050, "chunks": -1}

event: chunk
data: {"i": 0, "n": 3, "cache": "miss", "t_synth": 0.5, "audio_wav_b64": "UklGRi4..."}

event: chunk
data: {"i": 1, "n": 3, "cache": "hit", "t_synth": 0.0, "audio_wav_b64": "UklGRjQ..."}

event: chunk
data: {"i": 2, "n": 3, "cache": "miss", "t_synth": 0.4, "audio_wav_b64": "UklGRmA..."}

event: done
data: {"chunks": 3, "seconds_total": 0.9}
```

| Event | Data Fields | Description |
|-------|-------------|-------------|
| `meta` | request_id, sample_rate, chunks | Stream metadata |
| `chunk` | i, n, cache, t_synth, audio_wav_b64 | Audio chunk (base64) |
| `done` | chunks, seconds_total | Completion summary |
| `error` | code, message | Error (if occurred) |

---

### 4.4 GET /health - Health Check

Returns service health status and configuration.

**Response (200 OK)**

```json
{
  "ok": true,
  "warmed_up": true,
  "in_progress": false,
  "warmup_seconds": 2.5,
  "engine": "piper",
  "model_name": "tr_TR-dfki-medium",
  "device": "cpu",
  "loaded": true,
  "capabilities": {
    "speaker": true,
    "speaker_reference_audio": false,
    "language": true,
    "streaming": false
  },
  "chunking": {
    "breath_groups": true,
    "first_chunk_max": 80,
    "rest_chunk_max": 180
  },
  "cache": {
    "hits": 42,
    "misses": 8,
    "items": 10,
    "max_items": 256
  },
  "storage": {
    "total_files": 5,
    "total_size_mb": 12.5
  },
  "concurrency": {
    "max_concurrent": 2,
    "active": 1,
    "waiting": 0,
    "total_processed": 1000,
    "total_rejected": 2
  }
}
```

---

### 4.5 GET /metrics - Prometheus Metrics

Returns Prometheus-formatted metrics.

**Response (200 OK, text/plain)**

```
# HELP tts_requests_total Total TTS requests
# TYPE tts_requests_total counter
tts_requests_total{status="success"} 1000
tts_requests_total{status="failed"} 5

# HELP tts_request_duration_seconds Request duration
# TYPE tts_request_duration_seconds histogram
tts_request_duration_seconds_bucket{le="0.1"} 500
tts_request_duration_seconds_bucket{le="0.5"} 900
tts_request_duration_seconds_bucket{le="1.0"} 990

# HELP tts_cache_hits_total Cache hits by tier
# TYPE tts_cache_hits_total counter
tts_cache_hits_total{tier="mem"} 700
tts_cache_hits_total{tier="disk"} 150

# HELP tts_concurrent_requests Current concurrent requests
# TYPE tts_concurrent_requests gauge
tts_concurrent_requests 1
```

---

## 5. CLI Reference

### 5.1 Basic Usage

```bash
tts-ms [TEXT] [OPTIONS]
```

### 5.2 Arguments and Options

| Argument/Option | Description |
|-----------------|-------------|
| `TEXT` | Positional text to synthesize |
| `--text`, `-t` | Text to synthesize (alternative to positional) |
| `--file`, `-f` | Input file (one text per line for batch) |
| `--out`, `-o` | Output file or directory |
| `--speaker`, `-s` | Speaker/voice identifier |
| `--language`, `-l` | Language code |
| `--device`, `-d` | Device: "cuda" or "cpu" |
| `--dry-run` | Validate without synthesis |
| `--json` | Output as JSON |

### 5.3 Examples

**Single Text Synthesis**

```bash
# Using positional argument
tts-ms "Merhaba dünya" --out hello.wav

# Using --text option
tts-ms --text "Günaydın!" --out morning.wav

# With speaker and language
tts-ms "Nasılsınız?" --speaker female --language tr --out greeting.wav

# Force CPU device
tts-ms "Test" --device cpu --out test.wav
```

**Batch Processing**

```bash
# Create input file
echo "Merhaba dünya" > inputs.txt
echo "Nasılsınız?" >> inputs.txt
echo "İyi günler" >> inputs.txt

# Process all lines
tts-ms --file inputs.txt --out output_dir/

# Output files: output_dir/001.wav, output_dir/002.wav, output_dir/003.wav
```

**Dry Run Mode**

```bash
# Check chunking without synthesis
tts-ms --text "Uzun bir metin. İkinci cümle." --dry-run

# JSON output for automation
tts-ms --text "Test" --dry-run --json
```

**Dry Run JSON Output**

```json
{
  "ok": true,
  "dry_run": true,
  "items": [
    {
      "text_len": 23,
      "chunks": 2,
      "speaker": "default",
      "language": "tr",
      "device": "cuda"
    }
  ]
}
```

---

## 6. Configuration Guide

### 6.1 Configuration Sources

Configuration is loaded from multiple sources with the following priority (highest first):

1. **Environment Variables** - `TTS_MODEL_TYPE`, `TTS_MS_LOG_LEVEL`, etc.
2. **YAML File** - `config/settings.yaml`
3. **Defaults** - Built-in default values

### 6.2 Environment Variables

**Core Settings**

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TTS_MODEL_TYPE` | piper, legacy, f5tts, cosyvoice, styletts2, chatterbox, kokoro, qwen3tts, vibevoice | - | TTS engine to use |
| `TTS_DEVICE` | cuda, cpu | cuda | Compute device |
| `TTS_HOME` | path | system default | Model cache directory |

**Logging Settings**

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TTS_MS_LOG_LEVEL` | 1, 2, 3, 4 | 2 | Log verbosity |
| `TTS_MS_NO_COLOR` | 0, 1 | 0 | Disable colored output |
| `TTS_MS_RUNS_DIR` | path | `./runs` | Per-run log directory |

**Log Levels:**
- **1 (MINIMAL)** - Startup, shutdown, critical errors
- **2 (NORMAL)** - Request lifecycle, cache status
- **3 (VERBOSE)** - Per-stage timing, detailed flow
- **4 (DEBUG)** - Internal state, full tracing

**Resource Monitoring**

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TTS_MS_RESOURCES_ENABLED` | 0, 1 | 1 | Enable CPU/GPU/RAM tracking |
| `TTS_MS_RESOURCES_PER_STAGE` | 0, 1 | 1 | Log per-stage resources |
| `TTS_MS_RESOURCES_SUMMARY` | 0, 1 | 1 | Log resource summary |

**Operational**

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TTS_MS_SKIP_WARMUP` | 0, 1 | 0 | Skip engine warmup (testing only) |
| `TTS_MS_AUTO_INSTALL` | 0, 1 | 0 | Auto-install missing pip packages |
| `TTS_MS_SKIP_SETUP` | 0, 1 | 0 | Skip engine requirement checks (testing only) |

### 6.3 YAML Configuration (config/settings.yaml)

```yaml
# TTS Engine Settings
tts:
  engine: "piper"                     # Engine type
  default_language: "tr"              # Default language code
  default_speaker: "default"          # Default speaker
  device: "cuda"                      # cuda or cpu
  sample_rate: 22050                  # Output sample rate
  warmup_text: "Merhaba."             # Warmup text
  split_sentences: true               # Auto-chunk at sentences

# Memory Cache
cache:
  enabled: true
  max_items: 256                      # Maximum cached items
  ttl_seconds: 3600                   # Cache TTL (1 hour)

# Disk Storage
storage:
  enabled: true
  base_dir: "./storage"               # Storage directory
  ttl_minutes: 10080                  # Storage TTL (7 days)

# Concurrency Control
concurrency:
  enabled: true
  max_concurrent: 2                   # Max simultaneous synthesis
  max_queue: 10                       # Max queued requests
  timeout_s: 30.0                     # Acquisition timeout

# Request Batching (Experimental)
batching:
  enabled: false
  window_ms: 50                       # Collection window
  max_batch_size: 8                   # Max requests per batch
  max_workers: 4                      # Thread pool size

# Text Chunking
chunking:
  use_breath_groups: true             # Use breath-group chunking
  first_chunk_max: 80                 # First chunk max chars
  rest_chunk_max: 180                 # Subsequent chunks max chars
  legacy_max_chars: 220               # Legacy mode max chars

# Logging
logging:
  level: 2                            # 1-4 (MINIMAL to DEBUG)
  runs_dir: "./runs"                  # Per-run log directory (recommended)
  # log_dir: "./logs"                 # Legacy single-file mode
  jsonl_file: "tts-ms.jsonl"          # JSON log file (legacy mode only)
  text_preview_chars: 80              # Text preview length

# Resource Monitoring
resources:
  enabled: true
  per_stage: true                     # Per-stage logging
  summary: true                       # Summary logging
```

### 6.4 Engine-Specific Configuration

Each engine has its own configuration section:

```yaml
# Piper Engine
tts:
  piper:
    model_path: "/path/to/model.onnx"
    config_path: "/path/to/model.onnx.json"
    speaker_id: 0
    length_scale: 1.0
    noise_scale: 0.667
    noise_w: 0.8

# Legacy XTTS Engine
tts:
  legacy:
    model_name: "tts_models/multilingual/multi-dataset/xtts_v2"
    default_speaker: "Ana Florence"
    split_sentences: true

# F5-TTS Engine
tts:
  f5tts:
    checkpoint_path: "/path/to/checkpoint.pt"
    precision: "fp16"
    steps: 28
    cfg_strength: 2.0
    temperature: 0.7

# CosyVoice Engine
tts:
  cosyvoice:
    model_id: "CosyVoice-300M"
    precision: "bf16"
    temperature: 0.7
    top_p: 0.9

# StyleTTS2 Engine
tts:
  styletts2:
    checkpoint_path: "/path/to/checkpoint.pt"
    diffusion_steps: 30
    guidance_scale: 1.5

# Chatterbox Engine
tts:
  chatterbox:
    variant: "turbo"
    cfg_weight: 0.5
    exaggeration: 0.5

# Kokoro Engine (CPU/ONNX)
tts:
  kokoro:
    model_path: "models/kokoro/kokoro-v1.0.onnx"
    voices_path: "models/kokoro/voices-v1.0.bin"
    voice: "af_sarah"
    speed: 1.0
    lang: "en-us"

# Qwen3-TTS Engine
tts:
  qwen3tts:
    model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    speaker: "Vivian"
    dtype: "bfloat16"

# VibeVoice Engine
tts:
  vibevoice:
    model_id: "microsoft/VibeVoice-1.5B"
    max_new_tokens: 2048
```

---

## 7. TTS Engines

### 7.1 Piper (CPU-Only)

**Overview**

Piper is a fast, lightweight TTS engine that runs on CPU. It's ideal for deployments without GPU access or when low resource usage is required.

**Features**
- CPU-only operation
- Very fast synthesis
- Multiple Turkish voices
- No voice cloning support

**Configuration**

```yaml
tts:
  engine: "piper"
  piper:
    model_path: "~/.local/share/piper/tr_TR-dfki-medium.onnx"
    config_path: "~/.local/share/piper/tr_TR-dfki-medium.onnx.json"
    speaker_id: 0              # Speaker index (0-based)
    length_scale: 1.0          # Speed: <1.0 faster, >1.0 slower
    noise_scale: 0.667         # Variation amount
    noise_w: 0.8               # Phoneme duration variation
```

**Available Turkish Models**
- `tr_TR-dfki-medium` (recommended)
- `tr_TR-fettah-medium`

**Usage**

```bash
export TTS_MODEL_TYPE=piper
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

---

### 7.2 XTTS v2 / Legacy (GPU)

**Overview**

XTTS v2 (Coqui TTS) is a multilingual TTS model with native Turkish support and voice cloning capabilities.

**Features**
- Native Turkish support
- Multiple pre-trained speakers
- Voice cloning from reference audio
- Streaming support

**Configuration**

```yaml
tts:
  engine: "legacy"
  legacy:
    model_name: "tts_models/multilingual/multi-dataset/xtts_v2"
    default_speaker: "Ana Florence"
    split_sentences: true
    warmup_text: "Merhaba."
```

**Pre-trained Speakers**
- Ana Florence
- Patrick Zimmerman
- Jorge Lucas
- Akemi Okamura
- Rosie Taylor (Turkish-friendly)
- Paola Cortés

**Voice Cloning**

```json
{
  "text": "Merhaba dünya",
  "language": "tr",
  "speaker_wav_b64": "BASE64_ENCODED_AUDIO"
}
```

**Usage**

```bash
export TTS_MODEL_TYPE=legacy
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

---

### 7.3 F5-TTS (GPU)

**Overview**

F5-TTS is a voice cloning model that requires reference audio for all synthesis. It produces high-quality cloned voices.

**Features**
- High-quality voice cloning
- Requires reference audio (no default speakers)
- GPU required

**Configuration**

```yaml
tts:
  engine: "f5tts"
  f5tts:
    checkpoint_path: "/path/to/model.pt"
    precision: "fp16"
    steps: 28
    cfg_strength: 2.0
    ref_audio_seconds: 10
    ref_audio_strategy: "tail"    # head, tail, or concat
    cross_fade_ms: 20
    temperature: 0.7
```

**Reference Audio Requirements**
- WAV or MP3 format
- 10-30 seconds recommended
- Clear speech, minimal background noise
- Same language as synthesis target

**Usage with Reference Audio**

```bash
# Encode reference audio to base64
base64 reference.wav > ref_b64.txt

# Send request with reference
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Merhaba\", \"speaker_wav_b64\": \"$(cat ref_b64.txt)\"}" \
  --output speech.wav
```

---

### 7.4 CosyVoice (GPU)

**Overview**

CosyVoice (Alibaba) provides natural prosody with multiple synthesis modes including zero-shot voice cloning.

**Features**
- Natural prosody
- Pre-trained SFT speakers
- Zero-shot voice cloning
- Cross-lingual synthesis

**Configuration**

```yaml
tts:
  engine: "cosyvoice"
  cosyvoice:
    model_id: "CosyVoice-300M"     # or CosyVoice-300M-SFT
    precision: "bf16"
    temperature: 0.7
    top_p: 0.9
    top_k: 50
    repetition_penalty: 1.05
    length_scale: 1.0
    prosody_strength: 1.0
```

**Synthesis Modes**
- **SFT Mode** - Use pre-trained speakers
- **Zero-shot Mode** - Clone from reference audio
- **Cross-lingual Mode** - Same voice in different languages

---

### 7.5 StyleTTS2 (GPU)

**Overview**

StyleTTS2 is a diffusion-based TTS model with fine-grained style control.

**Features**
- Style control (pitch, energy, speed)
- Diffusion-based synthesis
- High-quality output

**Configuration**

```yaml
tts:
  engine: "styletts2"
  styletts2:
    checkpoint_path: "/path/to/checkpoint.pt"
    precision: "fp16"
    diffusion_steps: 30
    guidance_scale: 1.5
    style_strength: 0.8
    pitch_shift: 0.0              # -24 to +24 semitones
    energy_scale: 1.0
    speed_scale: 1.0
    use_denoiser: true
```

---

### 7.6 Chatterbox (GPU)

**Overview**

Chatterbox provides expressive speech synthesis with voice cloning support.

**Features**
- Expressive speech
- Voice cloning
- Multiple variants (turbo, regular, multilingual)

**Configuration**

```yaml
tts:
  engine: "chatterbox"
  chatterbox:
    variant: "turbo"              # turbo, regular, multilingual
    cfg_weight: 0.5               # 0.3-0.5 recommended
    exaggeration: 0.5             # Expressiveness (0.0-0.7+)
    reference_audio_path: null    # Optional default reference
```

---

### 7.7 Kokoro (CPU/ONNX)

**Overview**

Kokoro is a fast, CPU-friendly TTS engine using ONNX Runtime. It requires no GPU and supports multiple preset voices.

**Features**
- CPU-only inference via ONNX Runtime
- Multiple preset voices (af_sarah, af_heart, am_adam, etc.)
- Language selection support
- Fast synthesis (~2s for typical text)

**Configuration**

```yaml
tts:
  engine: "kokoro"
  kokoro:
    model_path: "models/kokoro/kokoro-v1.0.onnx"
    voices_path: "models/kokoro/voices-v1.0.bin"
    voice: "af_sarah"          # Default preset voice
    speed: 1.0                 # Speed multiplier
    lang: "en-us"              # Language (en-us, en-gb, ja, zh, fr, ko, es, etc.)
```

**Model Setup**

Download model files from HuggingFace:
```bash
pip install kokoro-onnx huggingface-hub
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('onnx-community/Kokoro-82M-v1.0-ONNX', 'kokoro-v1.0.onnx', local_dir='models/kokoro'); \
  hf_hub_download('onnx-community/Kokoro-82M-v1.0-ONNX', 'voices-v1.0.bin', local_dir='models/kokoro')"
```

**Usage**

```bash
export TTS_MODEL_TYPE=kokoro
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

---

### 7.8 Qwen3-TTS (GPU)

**Overview**

Qwen3-TTS is Alibaba's TTS model that supports both preset speaker voices and voice cloning from reference audio.

**Features**
- Preset speakers (Vivian, Serena, Ethan, Chelsie)
- Voice cloning from reference audio
- Multi-language support with automatic detection
- ~3-4 GB VRAM for 0.6B model

**Configuration**

```yaml
tts:
  engine: "qwen3tts"
  qwen3tts:
    model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    speaker: "Vivian"          # Default preset speaker
    dtype: "bfloat16"          # bfloat16, float16, or float32
```

**Preset Speakers**
- Vivian
- Serena
- Ethan
- Chelsie

**Voice Cloning**

```json
{
  "text": "Clone this voice",
  "speaker_wav_b64": "BASE64_ENCODED_AUDIO"
}
```

**Usage**

```bash
export TTS_MODEL_TYPE=qwen3tts
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

---

### 7.9 VibeVoice (GPU)

**Overview**

VibeVoice is Microsoft's research TTS model using a diffusion-based architecture. It always uses reference audio for synthesis (a silent reference is substituted when none is provided).

**Features**
- High-quality voice cloning
- Multi-language support (auto-detect)
- ~7 GB VRAM for 1.5B model
- Research-only license

**Configuration**

```yaml
tts:
  engine: "vibevoice"
  vibevoice:
    model_id: "microsoft/VibeVoice-1.5B"
    max_new_tokens: 2048        # Maximum generation length
```

**License Warning**

VibeVoice uses a research-only license. Check Microsoft's license terms before commercial deployment.

**Usage**

```bash
export TTS_MODEL_TYPE=vibevoice
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

---

## 8. Docker Deployment

### 8.1 Building Docker Images

```bash
# Build for Piper (CPU)
docker build --build-arg TTS_MODEL_TYPE=piper -t tts-ms:piper .

# Build for Legacy XTTS (GPU)
docker build --build-arg TTS_MODEL_TYPE=legacy -t tts-ms:legacy .

# Build for F5-TTS (GPU)
docker build --build-arg TTS_MODEL_TYPE=f5tts -t tts-ms:f5tts .
```

### 8.2 Running Containers

**CPU Deployment (Piper)**

```bash
docker run -d \
  --name tts-ms \
  -p 8000:8000 \
  -e TTS_MS_LOG_LEVEL=2 \
  -v ./storage:/app/storage \
  -v ./logs:/app/logs \
  tts-ms:piper
```

**GPU Deployment (XTTS/F5-TTS/etc.)**

```bash
docker run -d \
  --name tts-ms \
  --gpus all \
  -p 8000:8000 \
  -e TTS_MS_LOG_LEVEL=2 \
  -v ./storage:/app/storage \
  -v ./logs:/app/logs \
  tts-ms:legacy
```

### 8.3 Docker Compose

**Basic Setup (CPU)**

```bash
# Start with Piper
TTS_MODEL_TYPE=piper docker-compose up -d

# Check logs
docker-compose logs -f tts-api

# Stop
docker-compose down
```

**GPU Setup**

```bash
# Start with GPU profile
TTS_MODEL_TYPE=legacy docker-compose --profile gpu up -d
```

**With NGINX Reverse Proxy**

```bash
# Start with NGINX profile
TTS_MODEL_TYPE=piper docker-compose --profile nginx up -d

# Access via NGINX
curl http://localhost/v1/tts ...
```

### 8.4 Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODEL_TYPE` | piper | Engine type |
| `TTS_PORT` | 8000 | API port |
| `TTS_GPU_PORT` | 8001 | GPU service port |
| `TTS_MS_LOG_LEVEL` | 2 | Log verbosity |
| `TTS_MS_NO_COLOR` | 1 | Disable colors (container) |
| `TTS_MS_SKIP_WARMUP` | 0 | Skip warmup |

### 8.5 Health Checks

The Docker image includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); exit(0 if r.status_code == 200 else 1)"
```

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' tts-ms
```

---

## 9. Integration Examples

### 9.1 Python - requests

```python
import requests

# Basic synthesis
response = requests.post(
    "http://localhost:8000/v1/tts",
    json={
        "text": "Merhaba, nasılsınız?",
        "language": "tr"
    }
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print(f"Sample rate: {response.headers['X-Sample-Rate']}")
    print(f"Request ID: {response.headers['X-Request-Id']}")
else:
    print(f"Error: {response.json()}")
```

### 9.2 Python - OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, how are you today?"
)

response.stream_to_file("output.wav")
```

### 9.3 Python - Streaming

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/tts/stream",
    json={"text": "Uzun bir metin. İkinci cümle. Üçüncü cümle."},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if 'audio_wav_b64' in data:
                print(f"Received chunk {data['i']+1}/{data['n']}")
```

### 9.4 Python - Voice Cloning

```python
import requests
import base64

# Read reference audio
with open("reference.wav", "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/v1/tts",
    json={
        "text": "Bu ses referans sese benzeyecek.",
        "language": "tr",
        "speaker_wav_b64": ref_audio_b64
    }
)

with open("cloned_voice.wav", "wb") as f:
    f.write(response.content)
```

### 9.5 cURL Examples

```bash
# Basic synthesis
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba!", "language": "tr"}' \
  --output speech.wav

# OpenAI-compatible
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy"}' \
  --output speech.wav

# Check health
curl -s http://localhost:8000/health | jq '.ok'

# Get metrics
curl -s http://localhost:8000/metrics
```

### 9.6 JavaScript (Browser)

```javascript
// Basic fetch
async function synthesize(text) {
    const response = await fetch('http://localhost:8000/v1/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: 'tr' })
    });

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
}

// Streaming with EventSource
function streamSynthesize(text) {
    const response = await fetch('http://localhost:8000/v1/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        // Parse SSE events
        const lines = chunk.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.audio_wav_b64) {
                    // Play audio chunk
                    playBase64Audio(data.audio_wav_b64);
                }
            }
        }
    }
}
```

### 9.7 Node.js

```javascript
const fs = require('fs');
const fetch = require('node-fetch');

async function synthesize(text, outputPath) {
    const response = await fetch('http://localhost:8000/v1/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: 'tr' })
    });

    const buffer = await response.buffer();
    fs.writeFileSync(outputPath, buffer);

    console.log(`Request ID: ${response.headers.get('X-Request-Id')}`);
    console.log(`Sample Rate: ${response.headers.get('X-Sample-Rate')}`);
}

synthesize('Merhaba dünya', 'output.wav');
```

---

## 10. Caching and Performance

### 10.1 Two-Tier Cache Architecture

TTS-MS uses a two-tier caching system:

| Tier | Storage | Speed | Persistence | Default TTL |
|------|---------|-------|-------------|-------------|
| Memory | In-process LRU | ~1ms | Per-process | 1 hour |
| Disk | Sharded files | ~10ms | Across restarts | 7 days |

### 10.2 Cache Key Generation

Cache keys are SHA256 hashes of:
- Normalized text
- Speaker ID
- Language code
- Engine type and model ID
- Reference audio hash (if provided)
- Normalization version

### 10.3 Cache Flow

```
Request → Check Memory Cache
              ↓ miss
          Check Disk Cache
              ↓ miss
          Synthesize → Store Disk → Store Memory → Return
```

### 10.4 Cache Configuration

```yaml
cache:
  enabled: true
  max_items: 256              # Memory cache size
  ttl_seconds: 3600           # Memory TTL

storage:
  enabled: true
  base_dir: "./storage"
  ttl_minutes: 10080          # Disk TTL (7 days)
```

### 10.5 Performance Tips

1. **Enable Caching** - Repeated requests are served instantly
2. **Use Breath-Group Chunking** - Better audio quality at natural pauses
3. **Pre-warm Common Phrases** - Synthesize frequently used texts at startup
4. **Tune Concurrency** - Match `max_concurrent` to GPU memory
5. **Use Disk Storage** - Persists cache across restarts

---

## 11. Monitoring and Observability

### 11.1 Structured Logging

**Log Levels**

| Level | Value | Output |
|-------|-------|--------|
| MINIMAL | 1 | Startup, shutdown, critical errors |
| NORMAL | 2 | Request lifecycle, cache status |
| VERBOSE | 3 | Per-stage timing, detailed flow |
| DEBUG | 4 | Internal state, full tracing |

**Per-Run Log Directories**

Each server run creates an isolated directory under `runs/`:
```
runs/
└── run_143005_26022026_a1b2c3/    # run_HHMMSS_DDMMYYYY_hex
    ├── app.jsonl                   # All log records
    ├── resources.jsonl             # Resource metrics only (CPU/RAM/GPU)
    └── run_info.json               # Run metadata (engine, level, PID)
```

Configure via `TTS_MS_RUNS_DIR` or `logging.runs_dir` in settings.yaml. Falls back to legacy single-file mode (`log_dir`) if `runs_dir` is not set.

**Log Format (Console)**

```
14:30:05 [ INFO  ] (abc123) request_started chars=150 language=tr
14:30:06 [ INFO  ] (abc123) synthesis_complete cache=miss duration=1.234
```

**Log Format (JSONL)**

```json
{"ts":"2024-01-15T14:30:05+03:00","level":2,"tag":"INFO","request_id":"abc123","event":"request_started","chars":150}
```

### 11.2 Prometheus Metrics

Available at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `tts_requests_total` | Counter | Total requests by status |
| `tts_request_duration_seconds` | Histogram | Request latency |
| `tts_cache_hits_total` | Counter | Cache hits by tier |
| `tts_cache_misses_total` | Counter | Cache misses |
| `tts_concurrent_requests` | Gauge | Current active requests |
| `tts_queue_depth` | Gauge | Requests waiting |

**Grafana Dashboard Example**

```
Rate: rate(tts_requests_total[5m])
Latency P95: histogram_quantile(0.95, rate(tts_request_duration_seconds_bucket[5m]))
Cache Hit Rate: rate(tts_cache_hits_total[5m]) / (rate(tts_cache_hits_total[5m]) + rate(tts_cache_misses_total[5m]))
```

### 11.3 Resource Monitoring

When enabled (`TTS_MS_RESOURCES_ENABLED=1`):

**Per-Stage (VERBOSE level)**

```
resources stage=synth cpu=45.2% ram_delta=+12.3MB gpu=78.5% vram_delta=+256.0MB
```

**Summary (NORMAL level)**

```
resources_summary cpu_percent=52.1 ram_delta_mb=45.2 gpu_percent=65.3 vram_delta_mb=1024
```

### 11.4 Health Check Integration

For Kubernetes/Docker health checks:

```yaml
# Kubernetes
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

---

## 12. Troubleshooting

### 12.1 Common Issues

**"MODEL_NOT_READY" Error (503)**

The engine is still warming up. Solutions:
- Wait for warmup to complete (check `/health`)
- Increase `start-period` in health checks
- Check logs for warmup errors

```bash
# Check warmup status
curl -s http://localhost:8000/health | jq '.warmed_up'
```

**"QUEUE_FULL" Error (503)**

Too many concurrent requests. Solutions:
- Increase `max_queue` in configuration
- Add more replicas
- Enable request batching

**"TIMEOUT" Error (408)**

Synthesis took too long. Solutions:
- Reduce text length
- Increase `timeout_s` in concurrency config
- Check GPU utilization

**Out of Memory (OOM)**

GPU memory exhausted. Solutions:
- Reduce `max_concurrent`
- Use smaller model
- Switch to CPU (Piper)

### 12.2 Debug Mode

Enable debug logging:

```bash
export TTS_MS_LOG_LEVEL=4
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

### 12.3 Common Fixes

**No Audio Output**

```bash
# Check response headers
curl -I -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Test"}'

# Should see: Content-Type: audio/wav
```

**Slow First Request**

This is normal - the engine needs to warm up. Subsequent requests will be faster.

**Cache Not Working**

```bash
# Check cache stats
curl -s http://localhost:8000/health | jq '.cache'

# Verify storage directory exists and is writable
ls -la ./storage
```

### 12.4 Log Analysis

```bash
# Per-run mode — find the latest run
ls -t runs/ | head -1
# e.g. run_143005_26022026_a1b2c3

# View all logs from a run
tail -f runs/run_143005_26022026_a1b2c3/app.jsonl | jq

# View resource metrics only
cat runs/run_143005_26022026_a1b2c3/resources.jsonl | jq

# Filter by request ID
grep "abc123" runs/run_*/app.jsonl | jq

# Find errors across all runs
grep '"tag":"ERROR"' runs/run_*/app.jsonl | jq

# Legacy single-file mode
tail -f logs/tts-ms.jsonl | jq
```

---

## 13. FAQ

### General

**Q: Which engine should I use?**

| Use Case | Recommended Engine |
|----------|-------------------|
| CPU-only / lightweight | Piper |
| CPU-only / preset voices | Kokoro |
| General purpose | XTTS v2 (Legacy) |
| Voice cloning | F5-TTS, XTTS v2, or Qwen3-TTS |
| Natural prosody | CosyVoice |
| Style control | StyleTTS2 |
| Expressive speech | Chatterbox |
| Preset voices + cloning | Qwen3-TTS |
| Research / high-quality | VibeVoice |

**Q: Can I switch engines without changing client code?**

Yes. The `/v1/tts` API is engine-agnostic. Just change `TTS_MODEL_TYPE` and restart.

**Q: Is Turkish supported?**

Not all engines support Turkish natively. Turkish support by engine:

| Turkish Support | Engines |
|----------------|---------|
| **Native** | Piper (tr_TR model), XTTS v2 (Legacy), Chatterbox |
| **Via reference audio** | F5-TTS (voice cloning with Turkish ref) |
| **Not supported** | Kokoro, StyleTTS2, CosyVoice, Qwen3-TTS, VibeVoice |

For Turkish-focused deployments, Piper (CPU) and XTTS v2 (GPU) are recommended.

### Performance

**Q: How can I reduce latency?**

1. Enable caching (default)
2. Use breath-group chunking for streaming
3. Pre-warm common phrases
4. Use GPU (except Piper)

**Q: What's the maximum text length?**

4000 characters for `/v1/tts`, 4096 for `/v1/audio/speech`. Longer texts are automatically chunked.

**Q: How much GPU memory is needed?**

| Engine | VRAM Required |
|--------|---------------|
| Piper | None (CPU) |
| Kokoro | None (CPU/ONNX) |
| XTTS v2 | 4-6 GB |
| F5-TTS | 6-8 GB |
| CosyVoice | 6-8 GB |
| StyleTTS2 | 4-6 GB |
| Chatterbox | 4-6 GB |
| Qwen3-TTS | 3-4 GB |
| VibeVoice | ~7 GB |

### Integration

**Q: Is the API compatible with OpenAI?**

Yes. The `/v1/audio/speech` endpoint is a drop-in replacement for OpenAI's TTS API.

**Q: Can I use this with the OpenAI Python SDK?**

Yes:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
```

**Q: How do I do voice cloning?**

Encode reference audio as base64 and include in `speaker_wav_b64`:

```json
{"text": "...", "speaker_wav_b64": "BASE64_AUDIO"}
```

### Deployment

**Q: Can I run multiple engines simultaneously?**

No. Each process loads one engine. Run multiple containers for different engines.

**Q: How do I scale horizontally?**

Use multiple replicas behind a load balancer. Each replica has its own cache, so consider shared disk storage.

**Q: Is there a size limit for voice cloning audio?**

Yes. Maximum 10MB base64 (approximately 7.5MB decoded, or about 3 minutes at 22kHz).

---

## Appendix

### A. API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tts` | POST | Native TTS synthesis |
| `/v1/audio/speech` | POST | OpenAI-compatible TTS |
| `/v1/tts/stream` | POST | SSE streaming synthesis |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### B. Error Code Reference

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_INPUT | 400 | Invalid request parameters |
| TIMEOUT | 408 | Synthesis timeout |
| SYNTHESIS_FAILED | 500 | Engine synthesis error |
| QUEUE_FULL | 503 | Request queue full |
| MODEL_NOT_READY | 503 | Engine still warming up |
| INTERNAL_ERROR | 500 | Unexpected error |

### C. Voice Mapping (OpenAI → Internal)

| OpenAI Voice | Default Mapping |
|--------------|-----------------|
| alloy | default |
| echo | default |
| fable | default |
| onyx | default |
| nova | default |
| shimmer | default |

Configure custom mappings in `settings.yaml`.

---

**Version:** 1.0.0
**Last Updated:** February 2026
