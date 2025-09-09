#!/bin/bash
# Startup script for Voxtral 3B streaming service on RunPod

set -e

echo "🚀 Starting Voxtral 3B Real-Time Streaming Service..."

# Resolve application directory (supports both /app and /workspace/... layouts)
APP_DIR_CANDIDATES=(
    "${APP_DIR:-}"
    "/app"
    "/workspace/voxtralwithlabs"
    "/workspace"
)

for d in "${APP_DIR_CANDIDATES[@]}"; do
    if [ -n "$d" ] && [ -d "$d/src" ]; then
        APP_DIR="$d"
        break
    fi
done

if [ -z "$APP_DIR" ]; then
    echo "❌ Could not find application directory containing src/. Set APP_DIR and retry."
    exit 1
fi

# Set environment variables
export PYTHONPATH="$APP_DIR/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"}
export MODEL_NAME=${MODEL_NAME:-"mistralai/Voxtral-Mini-3B-2507"}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}
export ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-true}

# Function to check if GPU is available
check_gpu() {
    echo "🔍 Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "✅ GPU check completed"
    else
        echo "⚠️ nvidia-smi not found"
    fi
}

# Function to pre-download model
preload_model() {
    echo "📦 Pre-loading Voxtral model..."
    python3 -c "
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
print('Loading processor...')
processor = AutoProcessor.from_pretrained('$MODEL_NAME')
print('Loading model...')
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    '$MODEL_NAME',
    torch_dtype=torch.float16,
    device_map='auto' if torch.cuda.is_available() else None
)
print('✅ Model pre-loaded successfully')
"
}

# Function to setup directories
setup_directories() {
    echo "📁 Setting up directories..."
    mkdir -p /app/logs /app/data /app/cache
    chmod 755 /app/logs /app/data /app/cache
    echo "✅ Directories setup completed"
}

# Function to start the service
start_service() {
    echo "🎬 Starting the streaming service..."
    cd "$APP_DIR"

    # Determine uvicorn binary
    UVICORN_BIN=${UVICORN_BIN:-/app/venv/bin/uvicorn}
    if [ ! -x "$UVICORN_BIN" ]; then
        UVICORN_BIN=$(python3 -c 'import shutil,sys; p=shutil.which("uvicorn"); print(p or "")' || true)
    fi
    if [ -z "$UVICORN_BIN" ]; then
        echo "❌ uvicorn not found. Ensure dependencies are installed and/or set UVICORN_BIN."
        exit 1
    fi

    # Start with uvicorn for production; use --app-dir to avoid import path issues
    exec "$UVICORN_BIN" --app-dir ./src main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --log-level info \
        --access-log \
        --loop asyncio \
        --timeout-keep-alive 65 \
        --timeout-graceful-shutdown 30
}

# Function to monitor service health
health_check() {
    echo "🔍 Starting health monitoring..."
    while true; do
        sleep 30
        if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "⚠️ Health check failed at $(date)"
        fi
    done &
}

# Main execution
main() {
    echo "🌟 Voxtral 3B Streaming Service Startup"
    echo "========================================"

    # System checks
    check_gpu
    setup_directories

    # Pre-load model (optional, for faster first request)
    if [[ "${PRELOAD_MODEL:-false}" == "true" ]]; then
        preload_model
    fi

    # Start health monitoring in background
    health_check

    # Start the main service
    start_service
}

# Handle signals gracefully
trap 'echo "🛑 Shutting down..."; kill $(jobs -p); exit 0' SIGTERM SIGINT

# Run main function
main "$@"
