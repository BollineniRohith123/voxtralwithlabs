# Voxtral 3B Real-Time Streaming Implementation

A production-ready, real-time conversational AI system using Voxtral-Mini-3B-2507 with ultra-low latency WebRTC audio streaming and advanced VAD processing.

## 🌟 Features

### Core Capabilities
- **Real-time Audio Processing**: Ultra-low latency (<200ms end-to-end)
- **WebRTC Streaming**: Bidirectional audio streaming with WebSocket fallback
- **Advanced VAD**: Multiple VAD implementations (Silero, WebRTC, Energy-based)
- **Production Ready**: Docker containerization with GPU support
- **Scalable Architecture**: Async processing with concurrent session management
- **Monitoring & Metrics**: Prometheus metrics and Grafana dashboards

### Technical Optimizations
- **torch.compile()**: JIT compilation for 30%+ speedup
- **Response Caching**: Intelligent caching with LRU eviction
- **Batch Processing**: Dynamic batching for improved throughput
- **Memory Management**: GPU memory optimization and monitoring
- **Error Handling**: Comprehensive error recovery and reconnection logic

### Audio Processing
- **Multi-VAD Support**: Silero VAD, WebRTC VAD, and energy-based fallback
- **Noise Reduction**: Real-time noise suppression and filtering
- **Auto Gain Control**: Dynamic level adjustment
- **Chunked Processing**: Overlapping chunks with configurable parameters
- **Format Support**: 16-bit PCM, multiple sample rates (8kHz-48kHz)

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- RunPod account (for cloud deployment)
- Python 3.11+ (for local development)

### Local Development

1. **Clone and Setup**
```bash
git clone <repository>
cd voxtral-streaming
pip install -r requirements.txt
```

2. **Start Services**
```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Or run locally
cd src
python main.py
```

3. **Access the Client**
```bash
# Open web browser
http://localhost:8000/client
```

### RunPod Deployment

1. **Deploy to RunPod**
```bash
python deployment/runpod/runpod_deploy.py \
    --api-key YOUR_RUNPOD_API_KEY \
    --gpu-type "NVIDIA GeForce RTX 3090" \
    --name voxtral-streaming
```

2. **Monitor Deployment**
```bash
# Check deployment status
curl https://your-pod-url/health
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WebRTC Client │───▶│  FastAPI Server  │───▶│  Voxtral 3B     │
│   (Browser)     │◀───│  WebSocket Mgr   │◀───│  Inference      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │ Audio Processor │
                       │ VAD + Streaming │
                       └─────────────────┘
```

### Key Modules

1. **main.py**: FastAPI application with WebSocket endpoints
2. **voxtral_inference.py**: Optimized model inference engine
3. **audio_processor.py**: Real-time audio processing pipeline
4. **websocket_manager.py**: Connection and message management
5. **config.py**: Environment-based configuration system

### Data Flow

1. **Audio Capture**: Client captures microphone input via WebRTC
2. **WebSocket Streaming**: Real-time binary audio transmission
3. **VAD Processing**: Voice activity detection and buffering
4. **Model Inference**: Voxtral 3B processes audio chunks
5. **Response Delivery**: Text responses sent back to client

## ⚙️ Configuration

### Environment Variables

```bash
# Model Configuration
MODEL_NAME=mistralai/Voxtral-Mini-3B-2507
DEVICE=cuda
ENABLE_TORCH_COMPILE=true
MAX_BATCH_SIZE=4
TEMPERATURE=0.2

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
VAD_AGGRESSIVENESS=2
ENABLE_NOISE_REDUCTION=true
ENABLE_AUTO_GAIN=true

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0
```

### Advanced Settings

```python
# In config.py
model_config = ModelConfig(
    model_id="mistralai/Voxtral-Mini-3B-2507",
    use_compile=True,
    compile_mode="reduce-overhead",  # "default", "max-autotune"
    max_batch_size=4,
    cache_size=1000
)

audio_config = AudioConfig(
    sample_rate=16000,
    vad_aggressiveness=2,
    chunk_size_ms=160,
    overlap_ms=80,
    enable_noise_reduction=True
)
```

## 🔧 Development

### Project Structure

```
voxtral-streaming/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   └── voxtral_inference.py # Model inference engine
│   ├── audio/
│   │   ├── processor.py        # Audio processing pipeline
│   │   ├── vad.py             # Voice activity detection
│   │   └── webrtc_handler.py  # WebRTC audio handling
│   ├── streaming/
│   │   ├── websocket_manager.py # WebSocket management
│   │   ├── session_manager.py  # Session state management
│   │   └── queue_manager.py    # Processing queue management
│   └── utils/
│       ├── config.py          # Configuration management
│       ├── logging_config.py  # Logging setup
│       └── metrics.py         # Monitoring metrics
├── frontend/
│   ├── index.html             # WebRTC client interface
│   └── js/
│       ├── webrtc-client.js   # WebRTC implementation
│       └── audio-processor.js # Client-side processing
├── docker/
│   ├── Dockerfile             # Production container
│   ├── requirements.txt       # Python dependencies
│   └── docker-compose.yml     # Multi-service setup
├── deployment/
│   └── runpod/
│       ├── deploy.py          # RunPod deployment script
│       └── startup.sh         # Container startup
└── tests/
    ├── test_audio_processing.py
    ├── test_websocket.py
    └── test_integration.py
```

### Testing

```bash
# Run unit tests
pytest tests/

# Test audio processing
python tests/test_audio_processing.py

# Test WebSocket connections
python tests/test_websocket.py

# Integration tests
python tests/test_integration.py
```

### Performance Tuning

1. **torch.compile() Optimization**
```python
# Enable compilation for 30%+ speedup
model = torch.compile(
    model,
    mode="reduce-overhead",  # or "max-autotune" for best performance
    fullgraph=True
)
```

2. **Batch Processing**
```python
# Process multiple audio chunks together
responses = await voxtral_engine.process_batch(
    audio_chunks, session_ids
)
```

3. **Caching Strategy**
```python
# Configure response caching
inference_config = InferenceConfig(
    enable_caching=True,
    cache_size=1000
)
```

## 📊 Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/health | jq .
```

### Prometheus Metrics

```bash
# Access metrics endpoint
curl http://localhost:8000/metrics

# Key metrics to monitor:
# - websocket_connections_active
# - audio_processing_duration_seconds
# - model_inference_duration_seconds
# - gpu_memory_allocated_bytes
```

### Grafana Dashboard

Pre-configured dashboards available at `http://localhost:3000`:
- Real-time connection monitoring
- Audio processing latency
- GPU utilization and memory
- Error rates and system health

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
```bash
# Reduce batch size
export MAX_BATCH_SIZE=2

# Enable gradient checkpointing
export ENABLE_GRADIENT_CHECKPOINTING=true
```

2. **WebSocket Connection Failures**
```bash
# Check firewall settings
# Ensure ports 8000-8001 are open
# Verify WebSocket proxy configuration
```

3. **Audio Quality Issues**
```bash
# Adjust VAD sensitivity
export VAD_AGGRESSIVENESS=1  # Less aggressive

# Enable noise reduction
export ENABLE_NOISE_REDUCTION=true
```

4. **Model Loading Errors**
```bash
# Pre-download model
export PRELOAD_MODEL=true

# Check available disk space
df -h /app/cache
```

### Performance Optimization

1. **Reduce Latency**
   - Use `compile_mode="max-autotune"`
   - Increase `MAX_BATCH_SIZE` if GPU memory allows
   - Enable response caching
   - Use faster GPU (RTX 4090, A100, etc.)

2. **Improve Quality**
   - Adjust VAD sensitivity for environment
   - Enable noise reduction and auto-gain
   - Increase audio sample rate to 22kHz or 44kHz
   - Fine-tune chunk processing parameters

3. **Scale Performance**
   - Deploy multiple instances with load balancer
   - Use Redis for session state sharing
   - Enable horizontal pod autoscaling
   - Optimize Docker image size

## 📚 API Documentation

### WebSocket API

#### Audio Streaming Endpoint
```
WS /ws/audio/{session_id}
```

**Binary Messages**: Raw audio data (16-bit PCM)
**Text Messages**: JSON control messages

```javascript
// Send text input
{
  "type": "text_input",
  "text": "Hello, how are you?",
  "timestamp": 1640995200.0
}

// Receive text response
{
  "type": "text_response",
  "text": "I'm doing well, thank you!",
  "timestamp": 1640995201.0
}
```

### REST API

#### Create Session
```http
POST /session/create
```

Response:
```json
{
  "session_id": "uuid-string",
  "status": "created",
  "websocket_url": "/ws/audio/uuid-string"
}
```

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 1,
  "active_sessions": 5,
  "system": {
    "cpu_percent": 25.0,
    "memory_percent": 60.0,
    "gpu_memory_allocated": 8.5
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow Python PEP 8 style guide
- Add type hints to all functions
- Include docstrings for public methods
- Write unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Voxtral model
- **Silero Team** for VAD implementation
- **Google** for WebRTC VAD
- **PyTorch Team** for torch.compile optimization
- **FastAPI** for the web framework

## 🔗 Links

- [Voxtral Model on Hugging Face](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [RunPod Documentation](https://docs.runpod.io/)
- [WebRTC Documentation](https://webrtc.org/getting-started/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for real-time AI conversations**
"# voxtralwithlabs" 
