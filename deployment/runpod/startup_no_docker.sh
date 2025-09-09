#!/bin/bash
set -e

echo "🚀 Starting Voxtral 3B Real-Time Streaming Service (No Docker)..."

# Install system dependencies
apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
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

# Create a virtual environment
python3.11 -m venv /app/venv
source /app/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Set environment variables
export PYTHONPATH="/app/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"}
export MODEL_NAME=${MODEL_NAME:-"mistralai/Voxtral-Mini-3B-2507"}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}
export ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-true}

# Start the service
echo "🎬 Starting the streaming service..."
cd /app
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --loop asyncio \
    --timeout-keep-alive 65 \
    --timeout-graceful-shutdown 30
