# TTS-MS Usage Guide

Quick reference for setting up, starting, and using the TTS microservice.

---

## Table of Contents

1. [Quick Start (Local)](#1-quick-start-local)
2. [Quick Start (Docker)](#2-quick-start-docker)
3. [CLI Usage](#3-cli-usage)
4. [API Examples](#4-api-examples)
5. [SSE Streaming](#5-sse-streaming)
6. [Voice Cloning](#6-voice-cloning)
7. [Reference Tables](#7-reference-tables)

---

## 1. Quick Start (Local)

### Step 1: Install

```bash
git clone https://github.com/your-org/tts-ms.git
cd tts-ms
pip install -e .
```

### Step 2: Set Engine

```bash
# Choose one engine
export TTS_MODEL_TYPE=piper        # CPU, fast, no cloning
# export TTS_MODEL_TYPE=legacy     # GPU, Turkish, cloning supported
# export TTS_MODEL_TYPE=f5tts      # GPU, high-quality cloning
```

### Step 3: Start Server

```bash
python -m uvicorn tts_ms.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Test

```bash
# Health check (wait for warmed_up: true)
curl http://localhost:8000/health

# Synthesize
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba dünya!", "language": "tr"}' \
  --output test.wav

# Play audio
aplay test.wav        # Linux
afplay test.wav       # macOS
start test.wav        # Windows
```

---

## 2. Quick Start (Docker)

### Step 1: Build Image

```bash
# CPU engine (Piper)
docker build --build-arg TTS_MODEL_TYPE=piper -t tts-ms:piper .

# GPU engine (XTTS v2)
docker build --build-arg TTS_MODEL_TYPE=legacy -t tts-ms:legacy .
```

### Step 2: Run Container

```bash
# CPU (Piper)
docker run -d --name tts-ms -p 8000:8000 \
  -e TTS_MS_LOG_LEVEL=2 \
  -v $(pwd)/storage:/app/storage \
  tts-ms:piper

# GPU (XTTS v2) - requires nvidia-docker
docker run -d --name tts-ms -p 8000:8000 \
  --gpus all \
  -e TTS_MS_LOG_LEVEL=2 \
  -v $(pwd)/storage:/app/storage \
  tts-ms:legacy
```

### Step 3: Wait for Warmup

```bash
# Check health until warmed_up is true
watch -n 2 'curl -s http://localhost:8000/health | jq ".warmed_up"'

# Or simple loop
while ! curl -s http://localhost:8000/health | grep -q '"warmed_up":true'; do
  echo "Waiting for warmup..."
  sleep 5
done
echo "Ready!"
```

### Step 4: Test

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Docker ile merhaba!", "language": "tr"}' \
  --output docker_test.wav
```

### Step 5: View Logs

```bash
docker logs -f tts-ms
```

### Step 6: Stop

```bash
docker stop tts-ms && docker rm tts-ms
```

### Docker Compose Alternative

```bash
# Start
TTS_MODEL_TYPE=piper docker-compose up -d

# With GPU
TTS_MODEL_TYPE=legacy docker-compose --profile gpu up -d

# Stop
docker-compose down
```

---

## 3. CLI Usage

```bash
# Basic synthesis
tts-ms "Merhaba dünya" --out hello.wav

# With options
tts-ms --text "Günaydın!" --language tr --out output.wav

# Batch processing (one text per line)
tts-ms --file inputs.txt --out output_dir/

# Dry run (no synthesis, shows chunking)
tts-ms --text "Uzun bir metin. İkinci cümle." --dry-run --json

# Engine management
tts-ms --engines                      # Check all engine status
tts-ms --setup piper                  # Check engine requirements
tts-ms --setup f5tts --auto-install   # Setup with auto pip install
```

---

## 4. API Examples

### cURL - Basic

```bash
# Native endpoint
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba!", "language": "tr"}' \
  --output speech.wav

# OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy"}' \
  --output speech.wav
```

### Python - Basic

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/tts",
    json={"text": "Merhaba, nasılsınız?", "language": "tr"}
)
with open("output.wav", "wb") as f:
    f.write(response.content)

print(f"Request ID: {response.headers['X-Request-Id']}")
print(f"Sample Rate: {response.headers['X-Sample-Rate']}")
```

### Python - OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello!")
response.stream_to_file("output.wav")
```

### JavaScript - Fetch

```javascript
const response = await fetch('http://localhost:8000/v1/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'Merhaba!', language: 'tr' })
});
const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
new Audio(audioUrl).play();
```

---

## 5. SSE Streaming

Stream audio chunks in real-time using Server-Sent Events.

### cURL - SSE Stream

```bash
# Stream to terminal (see raw events)
curl -X POST http://localhost:8000/v1/tts/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"text": "Birinci cümle. İkinci cümle. Üçüncü cümle.", "language": "tr"}'
```

**Output format:**
```
event: meta
data: {"request_id": "abc123", "sample_rate": 22050, "chunks": -1}

event: chunk
data: {"i": 0, "n": 3, "cache": "miss", "t_synth": 0.45, "audio_wav_b64": "UklGRi4A..."}

event: chunk
data: {"i": 1, "n": 3, "cache": "miss", "t_synth": 0.38, "audio_wav_b64": "UklGRjQB..."}

event: chunk
data: {"i": 2, "n": 3, "cache": "miss", "t_synth": 0.42, "audio_wav_b64": "UklGRmAC..."}

event: done
data: {"chunks": 3, "seconds_total": 1.25}
```

### Python - SSE Stream

```python
import requests
import json
import base64

response = requests.post(
    "http://localhost:8000/v1/tts/stream",
    json={
        "text": "Birinci cümle. İkinci cümle. Üçüncü cümle.",
        "language": "tr"
    },
    stream=True
)

audio_chunks = []

for line in response.iter_lines():
    if not line:
        continue

    line = line.decode('utf-8')

    if line.startswith('data: '):
        data = json.loads(line[6:])

        if 'audio_wav_b64' in data:
            # Decode audio chunk
            audio_bytes = base64.b64decode(data['audio_wav_b64'])
            audio_chunks.append(audio_bytes)
            print(f"Chunk {data['i']+1}/{data['n']} received ({len(audio_bytes)} bytes)")

        elif 'chunks' in data and 'seconds_total' in data:
            print(f"Done! Total: {data['chunks']} chunks in {data['seconds_total']:.2f}s")

# Save combined audio (note: proper concatenation requires WAV header handling)
print(f"Received {len(audio_chunks)} audio chunks")
```

### Python - SSE with sseclient

```python
import sseclient
import requests
import json
import base64

response = requests.post(
    "http://localhost:8000/v1/tts/stream",
    json={"text": "Merhaba dünya. Nasılsınız?", "language": "tr"},
    stream=True
)

client = sseclient.SSEClient(response)

for event in client.events():
    data = json.loads(event.data)

    if event.event == 'meta':
        print(f"Stream started: {data['request_id']}, sample_rate: {data['sample_rate']}")

    elif event.event == 'chunk':
        audio = base64.b64decode(data['audio_wav_b64'])
        print(f"Chunk {data['i']+1}: {len(audio)} bytes, cache: {data['cache']}")
        # Play or save audio chunk here

    elif event.event == 'done':
        print(f"Complete: {data['chunks']} chunks, {data['seconds_total']:.2f}s")

    elif event.event == 'error':
        print(f"Error: {data['code']} - {data['message']}")
```

### JavaScript - SSE in Browser

```javascript
async function streamTTS(text) {
    const response = await fetch('http://localhost:8000/v1/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: 'tr' })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));

                if (data.audio_wav_b64) {
                    // Convert base64 to audio and play
                    const audioData = atob(data.audio_wav_b64);
                    const audioArray = new Uint8Array(audioData.length);
                    for (let i = 0; i < audioData.length; i++) {
                        audioArray[i] = audioData.charCodeAt(i);
                    }
                    const blob = new Blob([audioArray], { type: 'audio/wav' });
                    const audio = new Audio(URL.createObjectURL(blob));
                    audio.play();
                    console.log(`Playing chunk ${data.i + 1}/${data.n}`);
                }
            }
        }
    }
}

// Usage
streamTTS('Merhaba dünya. Bu bir streaming örneği.');
```

### Node.js - SSE Stream

```javascript
const fetch = require('node-fetch');

async function streamTTS(text) {
    const response = await fetch('http://localhost:8000/v1/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: 'tr' })
    });

    const reader = response.body;

    reader.on('data', (chunk) => {
        const lines = chunk.toString().split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.audio_wav_b64) {
                    const audioBuffer = Buffer.from(data.audio_wav_b64, 'base64');
                    console.log(`Chunk ${data.i + 1}: ${audioBuffer.length} bytes`);
                    // Save or process audioBuffer
                }
            }
        }
    });

    reader.on('end', () => console.log('Stream complete'));
}

streamTTS('Merhaba dünya. Nasılsınız?');
```

---

## 6. Voice Cloning

For engines that support voice cloning (legacy, f5tts, cosyvoice, chatterbox, qwen3tts, vibevoice).

### Step 1: Prepare Reference Audio

- Format: WAV or MP3
- Duration: 10-30 seconds recommended
- Quality: Clear speech, minimal background noise

### Step 2: Encode to Base64

```bash
# Linux/macOS
base64 -w 0 reference.wav > ref_b64.txt

# Windows PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("reference.wav")) > ref_b64.txt
```

### Step 3: Send Request

```bash
# cURL
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Bu ses referans sese benzeyecek.\", \"language\": \"tr\", \"speaker_wav_b64\": \"$(cat ref_b64.txt)\"}" \
  --output cloned.wav
```

### Python Example

```python
import requests
import base64

# Read and encode reference audio
with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode('utf-8')

# Synthesize with cloned voice
response = requests.post(
    "http://localhost:8000/v1/tts",
    json={
        "text": "Bu ses referans sese benzeyecek.",
        "language": "tr",
        "speaker_wav_b64": ref_b64
    }
)

with open("cloned_output.wav", "wb") as f:
    f.write(response.content)
```

---

## 7. Reference Tables

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TTS_MODEL_TYPE` | piper, legacy, f5tts, cosyvoice, styletts2, chatterbox, kokoro, qwen3tts, vibevoice | - | Engine selection |
| `TTS_DEVICE` | cuda, cpu | cuda | Compute device |
| `TTS_MS_LOG_LEVEL` | 1, 2, 3, 4 | 2 | Verbosity (1=minimal, 4=debug) |
| `TTS_MS_RUNS_DIR` | path | `./runs` | Per-run log directory |
| `TTS_MS_SKIP_WARMUP` | 0, 1 | 0 | Skip warmup (testing only) |
| `TTS_MS_NO_COLOR` | 0, 1 | 0 | Disable colored console output |
| `TTS_HOME` | path | system default | Model cache directory |
| `TTS_MS_RESOURCES_ENABLED` | 0, 1 | 1 | Enable/disable resource monitoring |
| `TTS_MS_RESOURCES_PER_STAGE` | 0, 1 | 1 | Log per-stage resources (VERBOSE) |
| `TTS_MS_RESOURCES_SUMMARY` | 0, 1 | 1 | Log resource summary (NORMAL) |
| `TTS_MS_AUTO_INSTALL` | 0, 1 | 0 | Auto-install missing pip packages |
| `TTS_MS_SKIP_SETUP` | 0, 1 | 0 | Skip engine requirement checks |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tts` | POST | Native TTS synthesis |
| `/v1/audio/speech` | POST | OpenAI-compatible |
| `/v1/tts/stream` | POST | SSE streaming |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Request Schema

```json
{
  "text": "string (required, 1-4000 chars)",
  "language": "string (optional, e.g., 'tr', 'en')",
  "speaker": "string (optional, engine-specific)",
  "split_sentences": "boolean (optional)",
  "speaker_wav_b64": "string (optional, base64 audio for cloning)"
}
```

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-Id` | Unique request identifier |
| `X-Sample-Rate` | Audio sample rate (e.g., 22050) |
| `X-Bytes` | Audio size in bytes |

### Engine Comparison

| Engine | GPU | Voice Cloning | Best For |
|--------|-----|---------------|----------|
| piper | No | No | Fast, lightweight, CPU-only |
| legacy | Yes | Yes | General purpose, Turkish |
| f5tts | Yes | Yes (required) | High-quality cloning |
| cosyvoice | Yes | Yes | Natural prosody |
| styletts2 | Yes | No | Style control |
| chatterbox | Yes | Yes | Expressive speech |
| kokoro | No | No | CPU-only, preset voices (ONNX) |
| qwen3tts | Yes | Yes | Preset voices + voice cloning |
| vibevoice | Yes | Yes | Research, high-quality cloning |

### SSE Event Types

| Event | Data Fields | Description |
|-------|-------------|-------------|
| `meta` | request_id, sample_rate, chunks | Stream metadata |
| `chunk` | i, n, cache, t_synth, audio_wav_b64 | Audio chunk |
| `done` | chunks, seconds_total | Completion |
| `error` | code, message | Error occurred |

---

## Testing

```bash
# Run tests
pytest -q -m "not slow"

# Run specific test
pytest -q tests/test_09_logging_levels.py

# Lint
ruff check src/ tests/
```
