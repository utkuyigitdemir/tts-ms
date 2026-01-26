#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Persist Coqui TTS cache on Google Drive
export TTS_HOME="/content/drive/MyDrive/tts-ms/.tts_cache"


echo "==[1/4]== System deps (ffmpeg) (skip if exists)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg
else
  echo "ffmpeg already installed"
fi

echo "==[2/4]== Python deps (per-runtime marker in /tmp)"
MARKER="/tmp/tts_ms_bootstrap_done"
if [ ! -f "$MARKER" ]; then
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  if [ -n "${TTS_MODEL_TYPE:-}" ]; then
    case "$TTS_MODEL_TYPE" in
      cosyvoice|styletts2|f5tts|piper)
        REQ_FILE="requirements_${TTS_MODEL_TYPE}.txt"
        if [ -f "$REQ_FILE" ]; then
          python -m pip install -r "$REQ_FILE"
        fi
        ;;
    esac
  fi
  touch "$MARKER"
else
  echo "deps already installed for this runtime"
fi

echo "==[3/4]== Quick sanity imports"
PYTHONPATH=src python -c "import tts_ms; print('IMPORT_OK')"

echo "==[4/4]== Pytest"
if [[ "${SKIP_SLOW:-0}" == "1" ]]; then
  PYTHONPATH=src pytest -q -m "not slow"
else
  PYTHONPATH=src pytest -q
fi
