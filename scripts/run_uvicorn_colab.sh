#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Persist model cache on Drive
export TTS_HOME="/content/drive/MyDrive/tts-ms/.tts_cache"

# Make sure our package is importable
export PYTHONPATH="src"

# Default host/port for Colab
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting tts-ms on http://${HOST}:${PORT}"
echo "TTS_HOME=$TTS_HOME"
echo "PYTHONPATH=$PYTHONPATH"

python -c "from tts_ms.main import app; print('APP_IMPORT_OK')"

uvicorn tts_ms.main:app --host "$HOST" --port "$PORT" --log-level info
