#!/bin/bash
set -e

ENGINE=$1
echo "Downloading models for engine: $ENGINE"

download_with_retry() {
    local url=$1
    local dest=$2
    echo "=> Downloading $url to $dest"
    # Added retry logic for better Docker build stability
    curl -fL --retry 5 --retry-delay 5 --retry-max-time 120 -o "$dest" "$url"
}

if [ "$ENGINE" = "piper" ]; then
    mkdir -p /app/models/piper
    download_with_retry "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx" "/app/models/piper/tr_TR-dfki-medium.onnx"
    download_with_retry "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json" "/app/models/piper/tr_TR-dfki-medium.onnx.json"
    download_with_retry "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx" "/app/models/piper/en_US-lessac-medium.onnx"
    download_with_retry "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" "/app/models/piper/en_US-lessac-medium.onnx.json"
elif [ "$ENGINE" = "kokoro" ]; then
    mkdir -p /app/models/kokoro
    download_with_retry "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" "/app/models/kokoro/kokoro-v1.0.onnx"
    download_with_retry "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" "/app/models/kokoro/voices-v1.0.bin"
else
    echo "No direct downloads configured for engine: $ENGINE"
fi
