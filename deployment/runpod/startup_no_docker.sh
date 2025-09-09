#!/bin/bash
set -e

echo "🚀 Starting Voxtral 3B Real-Time Streaming Service (No Docker)..."

# Install system dependencies
apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

# Create a virtual environment
python3.11 -m venv /app/venv
. /app/venv/bin/activate

# Upgrade pip (best-effort; pip may already be available)
pip install --upgrade pip || true

# Install Python dependencies
pip install --no-cache-dir -r "$APP_DIR/requirements.txt"

# Set environment variables
export PYTHONPATH="$APP_DIR/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"}
export MODEL_NAME=${MODEL_NAME:-"mistralai/Voxtral-Mini-3B-2507"}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}
export ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-true}

# Start the service
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

exec "$UVICORN_BIN" --app-dir ./src main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --loop asyncio \
    --timeout-keep-alive 65 \
    --timeout-graceful-shutdown 30
