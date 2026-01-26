# TTS-MS Workflow Documentation

This document describes the complete workflow of the tts-ms (Text-to-Speech Microservice) project, from application startup to audio output, with references to the source code.

---

## Table of Contents

1. [Application Startup](#1-application-startup)
2. [Configuration System](#2-configuration-system)
3. [Engine Loading and Warmup](#3-engine-loading-and-warmup)
4. [Request Processing Flow](#4-request-processing-flow)
5. [Text Processing Pipeline](#5-text-processing-pipeline)
6. [Caching System](#6-caching-system)
7. [Synthesis Execution](#7-synthesis-execution)
8. [Concurrency and Batching](#8-concurrency-and-batching)
9. [Streaming Response](#9-streaming-response)
10. [Logging and Monitoring](#10-logging-and-monitoring)
11. [CLI Interface](#11-cli-interface)

---

## 1. Application Startup

The application entry point is defined in `src/tts_ms/main.py`. When the server starts (via uvicorn or gunicorn), it follows this initialization sequence:

### 1.1 FastAPI Application Creation

```
src/tts_ms/main.py:28-54 - create_app()
```

The `create_app()` function orchestrates the startup:

1. **Configure Logging** - Initializes the structured logging system based on `TTS_MS_LOG_LEVEL` environment variable
2. **Create FastAPI Instance** - Instantiates the ASGI application with metadata (title, version, description)
3. **Register Routers** - Attaches API route handlers:
   - Native TTS router from `api/routes.py`
   - OpenAI-compatible router from `api/openai_compat.py`
4. **Register Startup Handler** - Schedules engine warmup via `warmup_engine()` event handler

```
src/tts_ms/main.py:58
```

The global `app` instance is created for ASGI servers to import.

### 1.2 Dependency Injection Setup

```
src/tts_ms/api/dependencies.py:55-73 - get_settings()
src/tts_ms/api/dependencies.py:76-102 - get_tts_service()
```

The dependency injection system uses `@lru_cache(maxsize=1)` to ensure:
- **Settings singleton** - Configuration loaded once from `config/settings.yaml`
- **TTSService singleton** - Single service instance shared across all requests

This pattern prevents multiple engine loads and VRAM fragmentation.

---

## 2. Configuration System

Configuration is managed through a hierarchical system with clear precedence rules.

### 2.1 Configuration Hierarchy

```
src/tts_ms/core/config.py:48-123 - Defaults class
src/tts_ms/core/config.py:395-451 - Settings class
src/tts_ms/core/config.py:220-393 - TTSServiceConfig class
```

**Priority (highest to lowest):**
1. Environment variables (`TTS_MODEL_TYPE`, `TTS_MS_LOG_LEVEL`, etc.)
2. YAML configuration (`config/settings.yaml`)
3. Static defaults (`Defaults` class)

### 2.2 Configuration Classes

The `TTSServiceConfig` dataclass aggregates validated sub-configurations:

```
src/tts_ms/core/config.py:125-217
```

- `CacheConfig` - Memory cache settings (max_items, ttl_seconds)
- `StorageConfig` - Disk storage settings (base_dir, ttl_seconds)
- `ConcurrencyConfig` - Request limiting (max_concurrent, max_queue, timeout)
- `BatchingConfig` - Request batching (enabled, window_ms, max_batch_size)
- `ChunkingConfig` - Text splitting (use_breath_groups, first_chunk_max, rest_chunk_max)
- `LoggingConfig` - Logging settings (level, no_color)
- `ResourcesConfig` - Monitoring settings (enabled, per_stage, summary)

### 2.3 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TTS_MODEL_TYPE` | Engine selection (piper, f5tts, cosyvoice, styletts2, chatterbox, legacy) | settings.yaml |
| `TTS_DEVICE` | Compute device (cuda/cpu) | cuda |
| `TTS_MS_LOG_LEVEL` | 1=MINIMAL, 2=NORMAL, 3=VERBOSE, 4=DEBUG | 2 |
| `TTS_MS_SKIP_WARMUP` | Skip engine warmup (testing) | 0 |
| `TTS_HOME` | Model cache directory | system default |

---

## 3. Engine Loading and Warmup

### 3.1 Engine Factory Pattern

```
src/tts_ms/tts/engine.py:239-344 - Engine factory functions
```

The engine system uses a factory pattern for flexibility:

1. **Engine Resolution** - `_resolve_engine_type()` checks `TTS_MODEL_TYPE` env var first, then falls back to settings
2. **Alias Normalization** - `_normalize_engine_type()` maps aliases (e.g., "xtts" → "legacy")
3. **Engine Creation** - `_create_engine()` instantiates the appropriate engine class
4. **Singleton Access** - `get_engine()` returns a cached instance

### 3.2 Base Engine Interface

```
src/tts_ms/tts/engine.py:70-228 - BaseTTSEngine class
```

All engines inherit from `BaseTTSEngine` and implement:

- `load()` - Load model files into memory/VRAM
- `synthesize(text, speaker, language, speaker_wav, split_sentences)` → `SynthResult`
- `warmup()` - JIT compilation, CUDA initialization
- `cache_key()` - Generate deterministic cache keys

### 3.3 Engine Capabilities

```
src/tts_ms/tts/engine.py:42-67 - EngineCapabilities dataclass
```

Each engine declares its capabilities:
- `speaker` - Supports multiple speakers
- `speaker_reference_audio` - Supports voice cloning from audio
- `language` - Supports multiple languages
- `streaming` - Supports chunk-by-chunk streaming

### 3.4 Warmup Process

```
src/tts_ms/services/tts_service.py:371-406 - warmup()
src/tts_ms/api/dependencies.py:104-122 - warmup_service()
```

At startup, the warmup process runs in a background thread:

1. **Model Loading** - `engine.load()` loads model weights to GPU/CPU
2. **JIT Compilation** - `engine.warmup()` triggers CUDA graph compilation
3. **Ready Flag** - Sets `_warmed_up = True` to allow requests

Requests arriving before warmup completes receive HTTP 503 (Service Unavailable).

---

## 4. Request Processing Flow

### 4.1 Native TTS Endpoint

```
src/tts_ms/api/routes.py:102-199 - tts_v1()
```

The `/v1/tts` endpoint processes requests through these stages:

```
POST /v1/tts (TTSRequest)
    │
    ├─→ Generate unique request_id (UUID)
    ├─→ Set request context for distributed tracing
    ├─→ Validate service readiness (is_ready check)
    ├─→ Decode speaker_wav from base64 (if provided)
    ├─→ Create SynthesizeRequest dataclass
    ├─→ Call service.synthesize()
    ├─→ Build response headers (X-Request-Id, X-Sample-Rate, X-Bytes)
    └─→ Return Response (audio/wav)
```

### 4.2 OpenAI-Compatible Endpoint

```
src/tts_ms/api/openai_compat.py:1-120+ - audio_speech()
```

The `/v1/audio/speech` endpoint maps OpenAI's TTS API format:

- Maps `voice` parameter to internal speaker names
- Supports `response_format` (wav, mp3, opus, flac)
- Uses the same underlying TTSService

### 4.3 Request Schema

```
src/tts_ms/api/schemas.py:38-99 - TTSRequest
```

```python
class TTSRequest:
    text: str           # Required, 1-4000 chars
    speaker: str | None # Optional, engine-specific
    language: str | None # Optional, default from settings
    split_sentences: bool | None # Optional
    speaker_wav_b64: str | None  # Base64 audio for voice cloning
```

### 4.4 Input Validation

```
src/tts_ms/services/validators.py:64-100+
```

Validation rules applied before synthesis:
- Text: Required, max 4000 characters
- Speaker WAV: Max 10MB base64 (≈7.5MB decoded audio)
- Validation errors return HTTP 400 with structured error response

---

## 5. Text Processing Pipeline

### 5.1 Text Normalization

```
src/tts_ms/services/tts_service.py:621-625 - normalize stage
```

Turkish-specific text normalization:
- Number to word conversion
- Abbreviation expansion
- Punctuation normalization

### 5.2 Text Chunking

```
src/tts_ms/tts/chunker.py:84-100+ - chunk_text()
src/tts_ms/tts/chunker.py - chunk_text_breath_groups()
```

Text is split into synthesis-friendly chunks using two strategies:

**Simple Chunking:**
1. Split by sentence boundaries (. ! ? …)
2. If too long, split by clause boundaries (, ; :)
3. If still too long, hard split at max_chars

**Breath-Group Chunking (Recommended):**
Splits at natural pause points including Turkish conjunctions:
- Sentence endings (. ! ? …)
- Clause boundaries (, ; :)
- Turkish conjunctions (ve, ama, fakat, veya, çünkü, ancak, yani, dolayısıyla)

Configuration from `Defaults`:
- `CHUNKING_USE_BREATH_GROUPS = True`
- `CHUNKING_FIRST_CHUNK_MAX = 80` (faster time-to-first-audio)
- `CHUNKING_REST_CHUNK_MAX = 180`

---

## 6. Caching System

The caching system uses a two-tier architecture for optimal performance.

### 6.1 Cache Key Generation

```
src/tts_ms/tts/engine.py:140-180 - cache_key()
src/tts_ms/tts/storage.py - make_key()
```

Cache keys are SHA256 hashes incorporating:
- Normalized text
- Speaker and language parameters
- Engine type and model ID
- Reference audio hash (if provided)
- Normalization version

### 6.2 Memory Cache (Tier 1)

```
src/tts_ms/tts/cache.py:57-120+ - TinyLRUCache
```

Fast, per-process in-memory cache:
- **LRU Eviction** - Removes least recently used items when full
- **TTL Support** - Automatic expiration after configured seconds
- **Thread-Safe** - Protected by threading.Lock

```python
class TinyLRUCache:
    def get(key) → (CacheItem | None, timings)
    def set(key, CacheItem)
    def stats() → cache statistics
```

### 6.3 Disk Storage (Tier 2)

```
src/tts_ms/tts/storage.py:90-100+ - Storage functions
```

Persistent, sharded file storage:

```
{base_dir}/
    ab/                    # Shard from key prefix
        abc123def.wav      # Cached audio file
    cd/
        cd456jkl.wav
```

Features:
- **Sharding** - Files distributed across 256 directories (first 2 hex chars)
- **Atomic Writes** - Prevents corruption from interrupted saves
- **TTL Manager** - Background cleanup of expired files

### 6.4 Two-Tier Cache Flow

```
src/tts_ms/services/tts_service.py:429-504 - _check_cache()
```

```
_check_cache(key)
    │
    ├─→ 1. Check memory cache (fast)
    │   ├── Hit → Return (wav_bytes, sr, "mem")
    │   └── Miss → Continue
    │
    └─→ 2. Check disk storage (slower)
        ├── Hit → Promote to memory, return (wav_bytes, sr, "disk")
        └── Miss → Return (None, None, "miss")
```

On cache miss, synthesized audio is stored in both tiers.

---

## 7. Synthesis Execution

### 7.1 Main Synthesis Method

```
src/tts_ms/services/tts_service.py:581-757 - synthesize()
```

The complete synthesis pipeline:

```
synthesize(request, request_id)
    │
    ├─→ STAGE 1: Normalize text
    │   └── normalize_tr(text)
    │
    ├─→ STAGE 2: Chunk text
    │   └── chunk_text() or chunk_text_breath_groups()
    │
    ├─→ STAGE 3: Generate cache key
    │   └── engine.cache_key()
    │
    ├─→ STAGE 4: Cache lookup
    │   └── _check_cache(key) → "mem" | "disk" | "miss"
    │
    ├─→ STAGE 5: Synthesize (if cache miss)
    │   └── _do_synthesis() → SynthResult
    │
    ├─→ STAGE 6: Store in cache
    │   └── _store_cache(key, wav_bytes, sr)
    │
    └─→ STAGE 7: Log metrics and return
        └── SynthesizeResult
```

### 7.2 Engine Synthesis Call

```
src/tts_ms/services/tts_service.py:509-576 - _do_synthesis()
```

The actual engine call with concurrency control:

```python
def _do_synthesis():
    if batching_enabled:
        return batcher.submit(text, speaker, language, speaker_wav)

    with concurrency_controller.acquire_sync(timeout):
        return engine.synthesize(text, speaker, language, speaker_wav)
```

### 7.3 SynthResult Structure

```
src/tts_ms/tts/engine.py:28-40 - SynthResult
```

```python
@dataclass
class SynthResult:
    wav_bytes: bytes      # Raw WAV audio data
    sample_rate: int      # Sample rate (22050, 24000, etc.)
    duration_seconds: float
```

### 7.4 SynthesizeResult (Full Response)

```
src/tts_ms/services/tts_service.py:153-172 - SynthesizeResult
```

```python
@dataclass
class SynthesizeResult:
    wav_bytes: bytes
    sample_rate: int
    cache_status: str       # "mem" | "disk" | "miss"
    total_seconds: float    # Total processing time
    request_id: str
    timings: dict          # Per-stage timing breakdown
```

---

## 8. Concurrency and Batching

### 8.1 Concurrency Controller

```
src/tts_ms/tts/concurrency.py:80-120+ - ConcurrencyController
```

Prevents GPU memory exhaustion through slot-based limiting:

```python
class ConcurrencyController:
    def __init__(max_concurrent=2, max_queue=10):
        # Single unified counter for sync/async

    def acquire_sync(timeout) → context manager
    def acquire_async(timeout) → async context manager
```

**Backpressure Strategy:**
1. Slot available → Acquire immediately
2. Queue has space → Wait for slot (up to timeout)
3. Queue full → Reject with 503 Service Unavailable

### 8.2 Request Batching

```
src/tts_ms/tts/batcher.py:86-100+ - RequestBatcher
```

Groups multiple requests for efficient processing:

```
Timeline:
0ms   - Request A arrives, starts collection window (50ms)
20ms  - Request B arrives, joins batch
40ms  - Request C arrives, joins batch
50ms  - Window expires, batch [A, B, C] processed together
60ms  - All results ready
```

Configuration:
- `window_ms` - Collection window duration (default: 50ms)
- `max_batch_size` - Maximum requests per batch (default: 8)
- `max_workers` - Thread pool size (default: 4)

---

## 9. Streaming Response

### 9.1 SSE Streaming Endpoint

```
src/tts_ms/api/routes.py:201-305 - tts_stream()
```

The `/v1/tts/stream` endpoint uses Server-Sent Events:

```
POST /v1/tts/stream
    │
    ├─→ SSE "meta" event
    │   └── {request_id, sample_rate, chunks: -1}
    │
    ├─→ Loop through synthesize_stream():
    │   └── SSE "chunk" event
    │       └── {index, b64_audio, duration_ms, is_final}
    │
    ├─→ SSE "done" event
    │   └── {total_chunks, total_bytes, total_duration_ms}
    │
    └─→ On error: SSE "error" event
        └── {code, message}
```

### 9.2 Stream Generator

```
src/tts_ms/services/tts_service.py:763-838 - synthesize_stream()
```

```python
async def synthesize_stream(request, request_id):
    chunks = chunk_text(request.text)

    for i, chunk in enumerate(chunks):
        # Check cache
        # Synthesize if miss
        # Store in cache
        yield StreamChunk(
            index=i,
            wav_bytes=...,
            sample_rate=...,
            is_final=(i == len(chunks) - 1)
        )
```

---

## 10. Logging and Monitoring

### 10.1 Structured Logging

```
src/tts_ms/core/logging/__init__.py:89-100+
src/tts_ms/core/logging/levels.py - LogLevel enum
```

**Numeric Log Levels (1-4):**
- **1 (MINIMAL)** - Startup, shutdown, critical errors only
- **2 (NORMAL)** - Request lifecycle, cache status (default)
- **3 (VERBOSE)** - Per-stage timing, detailed flow
- **4 (DEBUG)** - Internal state, full tracing

**Logging API:**
```python
info(logger, event, **kwargs)
warn(logger, event, **kwargs)
error(logger, event, **kwargs)
verbose(logger, event, **kwargs)  # Level 3+
debug(logger, event, **kwargs)    # Level 4+
```

### 10.2 Output Formats

**Console (Colored):**
```
14:30:05 [ INFO  ] (abc123) request_started chars=150 language=tr
```

**JSONL File:**
```json
{"ts":"2024-01-15T14:30:05+03:00","level":2,"tag":"INFO","request_id":"abc123",...}
```

### 10.3 Prometheus Metrics

```
src/tts_ms/core/metrics.py:85-100+ - TTSMetrics
```

Exposed at `/metrics`:
- `tts_requests_total` - Counter by engine, status
- `tts_request_duration_seconds` - Histogram
- `tts_cache_hits_total` - By tier (mem/disk)
- `tts_cache_misses_total`
- `tts_queue_depth` - Gauge
- `tts_concurrent_requests` - Gauge

### 10.4 Resource Monitoring

```
src/tts_ms/core/resources.py:75-100+
```

Tracks physical resource consumption:
- **CPU** - Process CPU usage (%)
- **RAM** - Memory delta during operation (MB)
- **GPU** - Utilization (%) and VRAM delta (MB)

```python
with resourceit("synthesis", sampler) as ctx:
    result = engine.synthesize(...)
# ctx.resources contains ResourceDelta
```

Logging output:
- **Per-stage (VERBOSE):** `resources stage=synth cpu=45.2% ram_delta=+12.3MB`
- **Summary (NORMAL):** `resources_summary cpu_percent=52.1 ram_delta_mb=45.2`

---

## 11. CLI Interface

```
src/tts_ms/cli.py:47-78 - _parse_args()
src/tts_ms/cli.py:81-100+ - main()
```

### 11.1 CLI Modes

**Single Text Synthesis:**
```bash
tts-ms --text "Merhaba dünya" --out hello.wav
tts-ms "Merhaba dünya" --out hello.wav  # Positional
```

**Batch Processing:**
```bash
tts-ms --file inputs.txt --out output_dir/
```

**Dry-Run (No Synthesis):**
```bash
tts-ms --text "Test" --dry-run --json
```

### 11.2 CLI Options

| Option | Description |
|--------|-------------|
| `--text`, `-t` | Input text to synthesize |
| `--file`, `-f` | Input file (one text per line) |
| `--out`, `-o` | Output file or directory |
| `--speaker`, `-s` | Speaker name |
| `--language`, `-l` | Language code |
| `--device`, `-d` | Compute device (cuda/cpu) |
| `--dry-run` | Validate without synthesis |
| `--json` | Output as JSON |

### 11.3 Entry Point

Defined in `pyproject.toml`:
```toml
[project.scripts]
tts-ms = "tts_ms.cli:main"
```

---

## Complete Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       HTTP POST /v1/tts                             │
│                        (TTSRequest JSON)                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API Layer (routes.py)                            │
│  • Generate request_id                                              │
│  • Validate service readiness                                       │
│  • Decode speaker_wav_b64                                           │
│  • Create SynthesizeRequest                                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Service Layer (tts_service.py)                       │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Normalize  │───▶│   Chunk     │───▶│ Cache Key   │             │
│  │    Text     │    │   Text      │    │ Generation  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                               │                     │
│                                               ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Cache Lookup                              │   │
│  │  ┌─────────────┐         ┌─────────────┐                    │   │
│  │  │   Memory    │───miss──▶│    Disk     │                    │   │
│  │  │   Cache     │         │   Storage   │                    │   │
│  │  └──────┬──────┘         └──────┬──────┘                    │   │
│  │         │ hit                   │ hit/miss                   │   │
│  │         ▼                       ▼                            │   │
│  │     RETURN                 RETURN/CONTINUE                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                               │                     │
│                                               ▼ (on miss)           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Synthesis Pipeline                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │ Concurrency │───▶│   Engine    │───▶│   Store     │      │   │
│  │  │  Control    │    │ synthesize()│    │   Cache     │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                               │                     │
│                                               ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SynthesizeResult: wav_bytes, sample_rate, cache_status,    │   │
│  │                    total_seconds, timings                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API Layer (routes.py)                            │
│  • Build response headers                                           │
│  • Record metrics                                                   │
│  • Log completion                                                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              HTTP Response (audio/wav)                              │
│  Headers: X-Request-Id, X-Sample-Rate, X-Bytes                      │
│  Body: WAV audio bytes                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Source Files Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| App Entry | `src/tts_ms/main.py` | 28-54 |
| Native API | `src/tts_ms/api/routes.py` | 102-199 |
| OpenAI API | `src/tts_ms/api/openai_compat.py` | 1-120+ |
| Schemas | `src/tts_ms/api/schemas.py` | 38-99 |
| Dependencies | `src/tts_ms/api/dependencies.py` | 55-122 |
| Config | `src/tts_ms/core/config.py` | 48-451 |
| TTS Service | `src/tts_ms/services/tts_service.py` | 201-838 |
| Validators | `src/tts_ms/services/validators.py` | 64-100+ |
| Engine Base | `src/tts_ms/tts/engine.py` | 70-344 |
| Chunker | `src/tts_ms/tts/chunker.py` | 84-100+ |
| Cache | `src/tts_ms/tts/cache.py` | 57-120+ |
| Storage | `src/tts_ms/tts/storage.py` | 90-100+ |
| Concurrency | `src/tts_ms/tts/concurrency.py` | 80-120+ |
| Batcher | `src/tts_ms/tts/batcher.py` | 86-100+ |
| Logging | `src/tts_ms/core/logging/__init__.py` | 89-100+ |
| Metrics | `src/tts_ms/core/metrics.py` | 85-100+ |
| Resources | `src/tts_ms/core/resources.py` | 75-100+ |
| CLI | `src/tts_ms/cli.py` | 47-100+ |
