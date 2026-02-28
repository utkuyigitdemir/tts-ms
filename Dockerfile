# tts-ms Dockerfile
# Multi-stage build supporting multiple TTS engines
#
# Build examples:
#   docker build --build-arg TTS_MODEL_TYPE=piper -t tts-ms:piper .
#   docker build --build-arg TTS_MODEL_TYPE=legacy -t tts-ms:legacy .
#   docker build --build-arg TTS_MODEL_TYPE=f5tts -t tts-ms:f5tts .
#   docker build --build-arg TTS_MODEL_TYPE=styletts2 -t tts-ms:styletts2 .
#   docker build --build-arg TTS_MODEL_TYPE=chatterbox -t tts-ms:chatterbox .
#   docker build --build-arg TTS_MODEL_TYPE=kokoro -t tts-ms:kokoro .
#   docker build --build-arg TTS_MODEL_TYPE=qwen3tts -t tts-ms:qwen3tts .
#   docker build --build-arg TTS_MODEL_TYPE=vibevoice -t tts-ms:vibevoice .
#
# Run examples:
#   docker run -p 8000:8000 tts-ms:piper
#   docker run -p 8000:8000 --gpus all tts-ms:legacy
#   docker run -p 8000:8000 --gpus all -e TTS_MS_LOG_LEVEL=4 tts-ms:f5tts
#
# Dev mode (code changes without rebuild):
#   docker run -p 8000:8000 \
#     -v ./src:/app/src -v ./config:/app/config \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     tts-ms:qwen3tts

ARG TTS_MODEL_TYPE=piper

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
FROM python:3.12-slim AS base-3.12
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM python:3.11-slim AS base-3.11
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM python:3.10-slim AS base-3.10
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# =============================================================================
# Stage 2: Engine-specific builds
#
# Layer order optimized for cache: pyproject.toml + stub -> deps -> src/
# Only the final COPY src/ layer is invalidated on code changes.
# =============================================================================

# -----------------------------------------------------------------------------
# Piper (Python 3.12, CPU-only)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-piper
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir piper-tts
# Download Piper models (Turkish + English)
COPY scripts/download_models.sh ./
RUN chmod +x download_models.sh && ./download_models.sh piper
COPY src/ ./src/

# -----------------------------------------------------------------------------
# Legacy XTTS v2 (Python 3.12, GPU recommended)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-legacy
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir coqui-tts
COPY src/ ./src/

# -----------------------------------------------------------------------------
# F5-TTS (Python 3.12, GPU recommended)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-f5tts
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir f5-tts edge-tts
COPY src/ ./src/

# -----------------------------------------------------------------------------
# StyleTTS2 (Python 3.12, GPU recommended, requires espeak-ng)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-styletts2
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir styletts2
RUN python -c "import nltk; nltk.download('punkt_tab')"
COPY src/ ./src/

# -----------------------------------------------------------------------------
# Chatterbox (Python 3.11 required, GPU recommended)
# -----------------------------------------------------------------------------
FROM base-3.11 AS engine-chatterbox
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir chatterbox-tts
COPY src/ ./src/

# -----------------------------------------------------------------------------
# CosyVoice (Python 3.10 required, GPU recommended)
# -----------------------------------------------------------------------------
FROM base-3.10 AS engine-cosyvoice
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
# CosyVoice requires git clone + deps. GPU-only and build-failing pkgs stripped from requirements.txt
RUN pip install --no-cache-dir setuptools
RUN git clone --depth 1 --recurse-submodules https://github.com/FunAudioLLM/CosyVoice /opt/CosyVoice && \
    cd /opt/CosyVoice && \
    sed -i '/tensorrt/d; /deepspeed/d; /onnxruntime-gpu/d; /extra-index-url/d; s/torch==2.3.1/torch/; s/torchaudio==2.3.1/torchaudio/; /openai-whisper/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt
# Ensure critical deps are installed (ruamel.yaml pinned for HyperPyYAML compat)
RUN pip install --no-cache-dir "ruamel.yaml<0.19" onnxruntime==1.18.0
RUN pip install --no-cache-dir openai-whisper==20231117 || pip install --no-cache-dir openai-whisper
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS"
COPY src/ ./src/

# -----------------------------------------------------------------------------
# Kokoro (Python 3.12, CPU-only, ONNX)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-kokoro
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir kokoro-onnx
# Download Kokoro model files from GitHub releases (not HuggingFace)
COPY scripts/download_models.sh ./
RUN chmod +x download_models.sh && ./download_models.sh kokoro
COPY src/ ./src/

# -----------------------------------------------------------------------------
# Qwen3-TTS (Python 3.12, GPU recommended)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-qwen3tts
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
# Install qwen-tts (pulls transformers<5 which it needs), then restore CUDA torch
RUN pip install --no-cache-dir qwen-tts && \
    (pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || true)
COPY src/ ./src/

# -----------------------------------------------------------------------------
# VibeVoice (Python 3.12, GPU required)
# -----------------------------------------------------------------------------
FROM base-3.12 AS engine-vibevoice
COPY pyproject.toml ./
RUN mkdir -p src/tts_ms && touch src/tts_ms/__init__.py
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
# Install vibevoice (pulls transformers<5 which it needs), then restore CUDA torch
RUN pip install --no-cache-dir vibevoice && \
    (pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || true)
COPY src/ ./src/

# =============================================================================
# Stage 3: Runtime image
# =============================================================================
FROM engine-${TTS_MODEL_TYPE} AS runtime

ARG TTS_MODEL_TYPE
ENV TTS_MODEL_TYPE=${TTS_MODEL_TYPE}

# Copy config
COPY config/ ./config/

# Create directories for logs and storage
RUN mkdir -p /app/logs /app/storage

# Set runtime environment
ENV TZ=Europe/Istanbul \
    TTS_MS_LOG_LEVEL=2 \
    TTS_MS_NO_COLOR=1 \
    TTS_MS_RESOURCES_ENABLED=1 \
    TTS_MS_SKIP_SETUP=1

# Declare volumes for persistence
VOLUME ["/app/logs", "/app/storage"]

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["tts_ms.main:app", "--host", "0.0.0.0", "--port", "8000"]
