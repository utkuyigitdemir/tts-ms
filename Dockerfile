# tts-ms Dockerfile
# Multi-stage build supporting multiple TTS engines
#
# Build examples:
#   docker build --build-arg TTS_MODEL_TYPE=piper -t tts-ms:piper .
#   docker build --build-arg TTS_MODEL_TYPE=legacy -t tts-ms:legacy .
#   docker build --build-arg TTS_MODEL_TYPE=chatterbox -t tts-ms:chatterbox .
#
# Run examples:
#   docker run -p 8000:8000 tts-ms:piper
#   docker run -p 8000:8000 --gpus all tts-ms:legacy
#   docker run -p 8000:8000 --gpus all tts-ms:chatterbox

ARG PYTHON_VERSION=3.12
ARG TTS_MODEL_TYPE=piper

# =============================================================================
# Stage 1: Base image with Python
# =============================================================================
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# Stage 2: Builder - install Python dependencies
# =============================================================================
FROM base as builder

# Copy dependency files
COPY pyproject.toml requirements.txt ./
COPY requirements_*.txt ./

# Install base dependencies
RUN pip install --no-cache-dir -e . || pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pyyaml \
    numpy \
    soundfile \
    httpx \
    nest-asyncio \
    python-multipart

# =============================================================================
# Stage 3: Engine-specific dependencies
# =============================================================================
FROM builder as engine-piper
RUN pip install --no-cache-dir piper-tts || echo "Piper installation skipped"
# Download Turkish piper model
RUN mkdir -p /app/models/piper && \
    pip install --no-cache-dir requests && \
    python -c "import requests; \
    r = requests.get('https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx'); \
    open('/app/models/piper/tr_TR.onnx', 'wb').write(r.content); \
    r = requests.get('https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json'); \
    open('/app/models/piper/tr_TR.json', 'wb').write(r.content)"

FROM builder as engine-legacy
RUN pip install --no-cache-dir coqui-tts || echo "Coqui-TTS installation may need manual setup"

FROM builder as engine-cosyvoice
# CosyVoice requires manual installation from source
RUN echo "CosyVoice requires manual installation"

FROM builder as engine-styletts2
# StyleTTS2 requires manual installation from source
RUN echo "StyleTTS2 requires manual installation"

FROM builder as engine-f5tts
# F5-TTS requires manual installation from source
RUN echo "F5-TTS requires manual installation"

FROM builder as engine-chatterbox
# Chatterbox TTS - requires GPU for best performance
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch torchaudio
RUN pip install --no-cache-dir chatterbox-tts || echo "Chatterbox installation may need manual setup"

# =============================================================================
# Stage 4: Runtime image
# =============================================================================
FROM engine-${TTS_MODEL_TYPE} as runtime

ARG TTS_MODEL_TYPE
ENV TTS_MODEL_TYPE=${TTS_MODEL_TYPE}

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for logs and storage
RUN mkdir -p /app/logs /app/storage

# Set runtime environment
ENV TZ=Europe/Istanbul \
    TTS_MS_LOG_LEVEL=2 \
    TTS_MS_LOG_DIR=/app/logs \
    TTS_MS_NO_COLOR=1

# Declare volumes for persistence
VOLUME ["/app/logs", "/app/storage"]

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); exit(0 if r.status_code == 200 else 1)"

# Run the server
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["tts_ms.main:app", "--host", "0.0.0.0", "--port", "8000"]
