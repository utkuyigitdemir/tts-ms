<p align="center">
  <img src="https://img.shields.io/badge/TTS--MS-Production%20Ready-brightgreen?style=for-the-badge" alt="Production Ready">
  <img src="https://img.shields.io/badge/Engines-9%20Supported-blue?style=for-the-badge" alt="9 Engines">
  <img src="https://img.shields.io/badge/Turkish-Native%20Support-red?style=for-the-badge" alt="Turkish Support">
  <img src="https://img.shields.io/badge/OpenAI-Compatible-orange?style=for-the-badge" alt="OpenAI Compatible">
</p>

<h1 align="center">TTS-MS</h1>

<p align="center">
  <strong>Enterprise-Grade Multi-Engine Text-to-Speech Microservice</strong>
</p>

<p align="center">
  A production-ready, high-performance TTS microservice featuring 9 interchangeable engines,<br>
  real-time streaming, intelligent caching, and seamless OpenAI API compatibility.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-engines">Engines</a> •
  <a href="#-api">API</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## Why TTS-MS?

| Challenge | TTS-MS Solution |
|-----------|-----------------|
| **Vendor Lock-in** | 9 interchangeable engines with unified API |
| **High Latency** | Two-tier caching + breath-group chunking |
| **GPU Memory Issues** | Smart concurrency control + one-model-per-process |
| **Integration Complexity** | OpenAI-compatible API - zero code changes |
| **Scalability** | Docker + NGINX + Prometheus metrics |
| **Turkish Language** | Native Turkish support with optimized chunking |

---

## Highlights

```
 9 TTS Engines          Piper • XTTS v2 • F5-TTS • CosyVoice • StyleTTS2 • Chatterbox • Kokoro • Qwen3-TTS • VibeVoice
 3 API Endpoints         Native • OpenAI-Compatible • SSE Streaming
 2-Tier Caching         In-Memory LRU + Persistent Disk Storage
 Real-time Streaming    Server-Sent Events with chunk-by-chunk delivery
 Voice Cloning          Zero-shot cloning from 10-30s reference audio
 Production Ready       Docker • NGINX • Prometheus • Health Checks
 Turkish Optimized      Breath-group chunking for natural pauses
 GPU Efficient          Concurrency control prevents VRAM exhaustion
```

---

## Quick Start

### Option 1: Local Installation

```bash
# Install
git clone https://github.com/your-org/tts-ms.git && cd tts-ms
pip install -e .

# Check available engines and their status
tts-ms --engines

# Setup a specific engine (auto-checks dependencies)
tts-ms --setup piper

# Run with Piper (CPU)
export TTS_MODEL_TYPE=piper
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000

# Synthesize
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "language": "en"}' \
  --output speech.wav
```

### Option 2: Docker (Recommended)

```bash
# Build and run
docker build --build-arg TTS_MODEL_TYPE=piper -t tts-ms:piper .
docker run -p 8000:8000 tts-ms:piper

# With GPU (XTTS v2)
docker build --build-arg TTS_MODEL_TYPE=legacy -t tts-ms:legacy .
docker run -p 8000:8000 --gpus all tts-ms:legacy
```

### Option 3: CLI (No Server)

```bash
tts-ms "Hello, world!" --out hello.wav
tts-ms --file texts.txt --out output_dir/
```

---

## Features

### Multi-Engine Architecture

<table>
<tr>
<td width="50%">

**Engine Agnostic Design**
- Swap engines without changing client code
- Unified `/v1/tts` API across all engines
- One model per process prevents VRAM fragmentation
- Hot-swappable via environment variable

</td>
<td width="50%">

```bash
# Switch engines instantly
export TTS_MODEL_TYPE=piper     # CPU, fast
export TTS_MODEL_TYPE=legacy    # GPU, quality
export TTS_MODEL_TYPE=f5tts     # GPU, cloning
export TTS_MODEL_TYPE=cosyvoice # GPU, prosody
```

</td>
</tr>
</table>

### Two-Tier Intelligent Caching

<table>
<tr>
<td width="50%">

**Memory Cache (Tier 1)**
- LRU eviction with configurable size
- TTL-based expiration
- ~1ms lookup time
- Thread-safe operations

**Disk Storage (Tier 2)**
- Sharded directory structure (256 shards)
- SHA256 content-addressable keys
- Atomic writes (crash-safe)
- Survives restarts

</td>
<td width="50%">

```
Request Flow:
┌─────────────┐
│   Request   │
└──────┬──────┘
       ▼
┌─────────────┐     ┌─────────────┐
│ Memory Cache│────▶│    HIT!     │ ~1ms
└──────┬──────┘     └─────────────┘
       │ miss
       ▼
┌─────────────┐     ┌─────────────┐
│ Disk Cache  │────▶│    HIT!     │ ~10ms
└──────┬──────┘     └─────────────┘
       │ miss
       ▼
┌─────────────┐
│  Synthesize │ Engine-dependent
└─────────────┘
```

</td>
</tr>
</table>

### Real-Time SSE Streaming

<table>
<tr>
<td width="50%">

**Server-Sent Events**
- Chunk-by-chunk audio delivery
- Faster time-to-first-audio (TTFA)
- Per-chunk timing and cache status
- Base64-encoded WAV chunks

**Event Types**
- `meta` - Stream metadata
- `chunk` - Audio segment
- `done` - Completion summary
- `error` - Error notification

</td>
<td width="50%">

```javascript
// Browser Example
const response = await fetch('/v1/tts/stream', {
  method: 'POST',
  body: JSON.stringify({ text: 'Long text...' })
});

const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  // Play audio chunk immediately
  playAudioChunk(value);
}
```

</td>
</tr>
</table>

### Turkish-Optimized Breath-Group Chunking

<table>
<tr>
<td width="50%">

**Intelligent Text Splitting**
- Splits at natural breathing points
- Turkish conjunctions: ve, ama, fakat, veya, çünkü, ancak, yani, dolayısıyla
- Clause boundaries: comma, semicolon, colon
- Short first chunk for faster feedback

</td>
<td width="50%">

```
Input: "Merhaba, nasılsınız? Ben iyiyim,
        teşekkür ederim ve siz nasılsınız?"

Chunks:
├── "Merhaba, nasılsınız?"     [80 chars max]
├── "Ben iyiyim,"              [breath point]
├── "teşekkür ederim"          [ve = breath]
└── "ve siz nasılsınız?"       [remaining]
```

</td>
</tr>
</table>

### GPU-Aware Concurrency Control

<table>
<tr>
<td width="50%">

**Smart Resource Management**
- Unified sync/async counter
- Prevents VRAM exhaustion
- Configurable queue depth
- Backpressure with 503 responses

**Statistics Tracked**
- Active requests
- Waiting queue depth
- Total processed
- Rejected count

</td>
<td width="50%">

```yaml
concurrency:
  enabled: true
  max_concurrent: 2    # Parallel synthesis
  max_queue: 10        # Waiting requests
  timeout_s: 30.0      # Acquisition timeout

# Backpressure Strategy:
# 1. Slots available → acquire immediately
# 2. Queue has space → wait for slot
# 3. Queue full → reject with 503
```

</td>
</tr>
</table>

### OpenAI API Compatibility

<table>
<tr>
<td width="50%">

**Drop-in Replacement**
- Same endpoint: `/v1/audio/speech`
- Same request format
- Compatible with OpenAI SDK
- Voice mapping (alloy, echo, nova...)

</td>
<td width="50%">

```python
from openai import OpenAI

# Just change base_url!
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, world!"
)
response.stream_to_file("output.wav")
```

</td>
</tr>
</table>

### Dynamic Request Batching

<table>
<tr>
<td width="50%">

**Experimental Feature**
- Collects requests within time window
- Processes batch together
- Improves GPU utilization
- Configurable window and batch size

</td>
<td width="50%">

```
Timeline:
0ms   ─ Request A arrives ─┐
20ms  ─ Request B arrives ─┼─► Batch
40ms  ─ Request C arrives ─┘
50ms  ─ Window expires ────► Process [A,B,C]
60ms  ─ All results ready
```

</td>
</tr>
</table>

### Voice Cloning

<table>
<tr>
<td width="50%">

**Zero-Shot Cloning**
- 10-30 seconds reference audio
- Base64-encoded in request
- Supported: F5-TTS, CosyVoice, Chatterbox, XTTS v2, Qwen3-TTS, VibeVoice

</td>
<td width="50%">

```python
import base64

with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()

response = requests.post("/v1/tts", json={
    "text": "Clone this voice!",
    "speaker_wav_b64": ref_b64
})
```

</td>
</tr>
</table>

---

## Engines

| Engine | GPU | Cloning | Turkish | Quality | Speed | Best For |
|--------|:---:|:-------:|:-------:|:-------:|:-----:|----------|
| **Piper** | - | - | tr_TR | Good | ⚡⚡⚡⚡⚡ | Production, Edge, High-throughput |
| **XTTS v2** | Yes | Yes | Native | Excellent | ⚡⚡ | General purpose, Voice cloning |
| **F5-TTS** | Yes | Required | Via ref | Excellent | ⚡⚡ | High-quality cloning |
| **CosyVoice** | Yes | Yes | Native | Excellent | ⚡⚡ | Natural prosody, Multilingual |
| **StyleTTS2** | Yes | - | Limited | Excellent | ⚡⚡ | Style control, Expressive |
| **Chatterbox** | Yes | Yes | Native | Excellent | ⚡⚡ | Expressive, Paralinguistics |
| **Kokoro** | - | - | - | Good | ⚡⚡⚡⚡ | CPU-only, Preset voices, ONNX |
| **Qwen3-TTS** | Yes | Yes | Auto | Excellent | ⚡⚡ | Preset voices, Voice cloning |
| **VibeVoice** | Yes | Yes | Auto | Excellent | ⚡⚡ | Research, High-quality cloning |

### Engine Selection Guide

```bash
# CPU-only, maximum speed
export TTS_MODEL_TYPE=piper

# CPU-only, preset voices (ONNX)
export TTS_MODEL_TYPE=kokoro

# GPU, best Turkish support
export TTS_MODEL_TYPE=legacy

# GPU, voice cloning required
export TTS_MODEL_TYPE=f5tts

# GPU, natural prosody
export TTS_MODEL_TYPE=cosyvoice

# GPU, style control
export TTS_MODEL_TYPE=styletts2

# GPU, expressive speech
export TTS_MODEL_TYPE=chatterbox

# GPU, preset voices + voice cloning
export TTS_MODEL_TYPE=qwen3tts

# GPU, high-quality research model
export TTS_MODEL_TYPE=vibevoice
```

### Multi-Environment Setup

Different engines require different Python versions:

| Environment | Python | Engines |
|-------------|--------|---------|
| `tts` | 3.12 | Piper, F5-TTS, StyleTTS2, Legacy XTTS v2, Kokoro, Qwen3-TTS, VibeVoice |
| `tts311` | 3.11 | Chatterbox |
| `tts310` | 3.10 | CosyVoice |

```bash
# Check engine requirements
tts-ms --engines

# Setup with auto-install
tts-ms --setup f5tts --auto-install

# Or enable auto-install globally
export TTS_MS_AUTO_INSTALL=1
```

The system automatically detects Python version mismatches and provides clear instructions:
```
RuntimeError: Engine 'chatterbox' requirements not satisfied.

Chatterbox (ResembleAI): Python 3.12 exceeds maximum 3.11

To fix, create the correct environment:
  conda create -n tts311 python=3.11 -y
  conda activate tts311
  pip install -e /path/to/tts-ms
```

---

## API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tts` | POST | Native TTS synthesis |
| `/v1/tts/stream` | POST | SSE streaming synthesis |
| `/v1/audio/speech` | POST | OpenAI-compatible endpoint |
| `/health` | GET | Health check with stats |
| `/metrics` | GET | Prometheus metrics |

### Request Examples

<details>
<summary><b>Native TTS</b></summary>

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Merhaba, nasılsınız?",
    "language": "tr",
    "speaker": "default"
  }' --output speech.wav
```

Response Headers:
- `X-Request-Id`: Request tracking ID
- `X-Sample-Rate`: Audio sample rate
- `X-Bytes`: Audio size

</details>

<details>
<summary><b>OpenAI Compatible</b></summary>

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, world!",
    "voice": "alloy"
  }' --output speech.wav
```

</details>

<details>
<summary><b>SSE Streaming</b></summary>

```bash
curl -X POST http://localhost:8000/v1/tts/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"text": "Long text to stream..."}'
```

Events:
```
event: meta
data: {"request_id": "abc123", "sample_rate": 22050}

event: chunk
data: {"i": 0, "audio_wav_b64": "UklGRi4A..."}

event: done
data: {"chunks": 3, "seconds_total": 1.5}
```

</details>

<details>
<summary><b>Voice Cloning</b></summary>

```bash
# Encode reference audio
REF_B64=$(base64 -w 0 reference.wav)

curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"This will sound like the reference.\",
    \"speaker_wav_b64\": \"$REF_B64\"
  }" --output cloned.wav
```

</details>

---

## Deployment

### Docker Compose (Production)

```yaml
# docker-compose.yml
services:
  tts-api:
    image: tts-ms:piper
    ports:
      - "8000:8000"
    environment:
      - TTS_MODEL_TYPE=piper
      - TTS_MS_LOG_LEVEL=2
    volumes:
      - ./storage:/app/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Start
docker-compose up -d

# With GPU
docker-compose --profile gpu up -d

# With NGINX reverse proxy
docker-compose --profile nginx up -d
```

### NGINX Configuration

```nginx
upstream tts_backend {
    server tts-api:8000;
    keepalive 32;
}

location /v1/tts/stream {
    proxy_pass http://tts_backend;
    proxy_buffering off;           # SSE streaming
    proxy_read_timeout 300s;
    proxy_set_header Connection '';
}

location /v1/ {
    proxy_pass http://tts_backend;
    proxy_read_timeout 60s;
    client_max_body_size 50M;      # Voice cloning uploads
}
```

### Kubernetes Health Probes

```yaml
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

## Monitoring

### Prometheus Metrics

```
# Request metrics
tts_requests_total{status="success"} 1000
tts_request_duration_seconds_bucket{le="1.0"} 950

# Cache metrics
tts_cache_hits_total{tier="mem"} 700
tts_cache_hits_total{tier="disk"} 200
tts_cache_misses_total 100

# Resource metrics
tts_concurrent_requests 2
tts_queue_depth 0
```

### Structured Logging

```bash
# Log levels
export TTS_MS_LOG_LEVEL=1  # MINIMAL - startup, errors only
export TTS_MS_LOG_LEVEL=2  # NORMAL  - request lifecycle (default)
export TTS_MS_LOG_LEVEL=3  # VERBOSE - per-stage timing
export TTS_MS_LOG_LEVEL=4  # DEBUG   - full tracing
```

Console output:
```
14:30:05 [ INFO  ] (abc123) request_started chars=150 language=tr
14:30:06 [ INFO  ] (abc123) cache_hit tier=mem
14:30:06 [ OK    ] (abc123) synthesis_complete duration=0.05s
```

**Per-Run Log Directories**

Each server run creates an isolated log directory under `runs/`:
```
runs/
└── run_143005_26022026_a1b2c3/
    ├── app.jsonl          # All log records
    ├── resources.jsonl    # Resource metrics only (CPU/RAM/GPU)
    └── run_info.json      # Run metadata (engine, level, PID)
```

Configure via environment variable or settings.yaml:
```bash
export TTS_MS_RUNS_DIR=./runs
```

### Resource Monitoring

```bash
# Enable CPU/GPU/RAM tracking
export TTS_MS_RESOURCES_ENABLED=1
export TTS_MS_RESOURCES_SUMMARY=1
```

Output:
```
resources_summary cpu_percent=45.2 ram_delta_mb=12.3 gpu_percent=78.5 vram_delta_mb=256
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_MODEL_TYPE` | Engine: piper, legacy, f5tts, cosyvoice, styletts2, chatterbox, kokoro, qwen3tts, vibevoice | - |
| `TTS_DEVICE` | Compute device: cuda, cpu | cuda |
| `TTS_MS_LOG_LEVEL` | Log verbosity: 1-4 | 2 |
| `TTS_MS_SKIP_WARMUP` | Skip engine warmup (testing) | 0 |
| `TTS_MS_NO_COLOR` | Disable colored output | 0 |
| `TTS_HOME` | Model cache directory | system default |
| `TTS_MS_RUNS_DIR` | Per-run log directory | `./runs` |
| `TTS_MS_RESOURCES_ENABLED` | Enable resource monitoring | 1 |
| `TTS_MS_RESOURCES_PER_STAGE` | Log per-stage resources (VERBOSE) | 1 |
| `TTS_MS_RESOURCES_SUMMARY` | Log resource summary (NORMAL) | 1 |
| `TTS_MS_AUTO_INSTALL` | Auto-install missing pip packages | 0 |
| `TTS_MS_SKIP_SETUP` | Skip engine requirement checks | 0 |

### YAML Configuration

```yaml
# config/settings.yaml
tts:
  engine: "piper"
  default_language: "tr"
  device: "cuda"

cache:
  max_items: 256
  ttl_seconds: 3600

storage:
  base_dir: "./storage"
  ttl_seconds: 604800

concurrency:
  max_concurrent: 2
  max_queue: 10
  timeout_s: 30.0

chunking:
  use_breath_groups: true
  first_chunk_max: 80
  rest_chunk_max: 180

logging:
  level: 2
  runs_dir: "./runs"              # Per-run log directories
  # log_dir: "./logs"             # Legacy single-file mode
```

---

## Benchmarks

All 9 engines benchmarked on a 22-core CPU machine (2026-02-26):

| Engine | Startup | Avg Synthesis | Tests | Turkish | Notes |
|--------|---------|---------------|-------|---------|-------|
| **Piper** | 3.8s | 0.27s | 20/20 | Yes | CPU-native, fastest |
| **Kokoro** | 3.1s | 1.85s | 10/10 | No | CPU-only ONNX |
| **StyleTTS2** | 106.3s | 8.43s | 10/10 | No | English, style transfer |
| **XTTS v2** | 188.4s | 12.04s | 20/20 | Yes | Voice cloning |
| **Chatterbox** | 261.4s | 32.46s | 20/20 | Yes | Expressive speech |
| **CosyVoice** | 54.0s | 53.97s | 10/10 | No | Natural prosody |
| **F5-TTS** | 351.2s | 92.11s | 10/10 | No | Voice cloning model |
| **VibeVoice** | 450.9s | 198.07s | 10/10 | No | Research model |
| **Qwen3-TTS** | 194.5s | 320.87s | 3/10 | No | Needs GPU |

> Engines without Turkish support ran English-only tests (10 instead of 20). All tests on CPU — GPU engines will be significantly faster with CUDA.

[View detailed benchmarks](docs/EXAMPLES.md) | [Listen to audio samples](https://innovacomtr-my.sharepoint.com/:f:/g/personal/udemir_innova_com_tr/IgAnllV_7lDiQp4Euj_R4WSIAYqyZBrHslTaup9Wqor0CkQ?e=exqlGp)

---

## Documentation

| Document | Description |
|----------|-------------|
| [**USAGE.md**](docs/USAGE.md) | Quick reference guide with step-by-step examples |
| [**DOCUMENTATION.md**](docs/DOCUMENTATION.md) | Complete user manual and API reference |
| [**WORKFLOW.md**](docs/WORKFLOW.md) | Technical architecture and code flow |
| [**EXAMPLES.md**](docs/EXAMPLES.md) | Benchmark results and performance analysis |
| [**ENGINE_ENVIRONMENTS.md**](docs/ENGINE_ENVIRONMENTS.md) | Multi-environment setup guide |
| [**ENGINE_REQUIREMENTS.md**](docs/ENGINE_REQUIREMENTS.md) | Engine-specific test dependencies |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client                                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NGINX Reverse Proxy                               │
│              (Load Balancing, SSL, Buffering)                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI App                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  /v1/tts    │  │ /v1/audio/  │  │ /v1/tts/    │                  │
│  │  (Native)   │  │   speech    │  │   stream    │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         └────────────────┼────────────────┘                          │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      TTSService                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │
│  │  │Normalize │─▶│  Chunk   │─▶│  Cache   │─▶│Synthesize│    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │                                           │
│         ┌────────────────┼────────────────┐                          │
│         ▼                ▼                ▼                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │Memory Cache │  │ Disk Cache  │  │Concurrency  │                  │
│  │   (LRU)     │  │  (Sharded)  │  │  Control    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│                          │                                           │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                        TTS Engine                          │    │
│  │  Piper │ XTTS v2 │ F5-TTS │ CosyVoice │ StyleTTS2        │    │
│  │  Chatterbox │ Kokoro │ Qwen3-TTS │ VibeVoice              │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## CLI Reference

```bash
# Basic synthesis
tts-ms "Hello, world!" --out hello.wav

# With options
tts-ms --text "Merhaba!" --language tr --speaker default --out output.wav

# Batch processing
tts-ms --file inputs.txt --out output_dir/

# Dry run (preview chunking)
tts-ms --text "Long text..." --dry-run --json

# Force CPU
tts-ms --text "Test" --device cpu --out test.wav

# Engine management
tts-ms --engines                      # List all engines and status
tts-ms --setup piper                  # Check engine requirements
tts-ms --setup f5tts --auto-install   # Setup with auto pip install
```

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (excluding slow GPU tests)
TTS_MODEL_TYPE=piper pytest -q -m "not slow"
# Result: ~470 passed, ~10 skipped

# Run specific test
pytest -q tests/test_09_logging_levels.py

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with FastAPI • Powered by 9 TTS Engines • Optimized for Turkish</sub>
</p>

<p align="center">
  <a href="docs/USAGE.md">Quick Start</a> •
  <a href="docs/DOCUMENTATION.md">Full Docs</a> •
  <a href="docs/EXAMPLES.md">Benchmarks</a> •
  <a href="https://innovacomtr-my.sharepoint.com/:f:/g/personal/udemir_innova_com_tr/IgAnllV_7lDiQp4Euj_R4WSIAYqyZBrHslTaup9Wqor0CkQ?e=exqlGp">Audio Samples</a>
</p>
