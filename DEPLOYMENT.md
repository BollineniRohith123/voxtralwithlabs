# Voxtral 3B Deployment Guide

This guide covers deploying the Voxtral 3B Real-Time Streaming service across different environments, with detailed instructions for RunPod, local development, and production deployments.

## 🎯 Deployment Options

### 1. RunPod Cloud Deployment (Recommended)
- **Best for**: Production workloads, scalable GPU access
- **GPU Options**: RTX 3090, RTX 4090, A40, A100
- **Estimated Cost**: $0.20-0.80/hour depending on GPU

### 2. Local Development Setup
- **Best for**: Development, testing, prototyping
- **Requirements**: NVIDIA GPU with 8GB+ VRAM
- **Cost**: Free (using own hardware)

### 3. Docker Compose Deployment
- **Best for**: Multi-service local deployments
- **Includes**: Redis, Prometheus, Grafana
- **Cost**: Free (infrastructure costs only)

### 4. Kubernetes Deployment
- **Best for**: Large-scale production deployments
- **Features**: Auto-scaling, load balancing, rolling updates
- **Cost**: Variable based on cluster size

## 🚀 RunPod Deployment

### Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **API Key**: Generate from RunPod dashboard
3. **Docker Image**: Build or use pre-built image

### Step 1: Prepare Docker Image

Build and push the Docker image:

```bash
# Clone repository
git clone <your-repository>
cd voxtral-streaming

# Build Docker image
docker build -t your-username/voxtral-streaming:latest .

# Push to Docker Hub
docker push your-username/voxtral-streaming:latest
```

### Step 2: Automated Deployment

Use the provided deployment script:

```bash
# Install dependencies
pip install requests

# Deploy to RunPod
python deployment/runpod/runpod_deploy.py \
    --api-key YOUR_RUNPOD_API_KEY \
    --name voxtral-streaming-prod \
    --gpu-type "NVIDIA GeForce RTX 3090" \
    --cloud-type COMMUNITY \
    --volume-size 50 \
    --max-batch-size 4 \
    --enable-torch-compile
```

### Step 3: Manual Deployment (Alternative)

If you prefer manual deployment through RunPod UI:

1. **Create New Pod**:
   - Go to RunPod dashboard → Pods → Deploy
   - Select GPU type (RTX 3090 recommended)
   - Choose Community Cloud for cost savings

2. **Configure Container**:
   - Image: `your-username/voxtral-streaming:latest`
   - Ports: `8000/http, 8001/http`
   - Container Disk: 20 GB
   - Volume: 50 GB (optional, for model caching)

3. **Environment Variables**:
   ```bash
   CUDA_VISIBLE_DEVICES=0
   TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0
   MODEL_NAME=mistralai/Voxtral-Mini-3B-2507
   ENABLE_TORCH_COMPILE=true
   MAX_BATCH_SIZE=4
   ```

4. **Deploy and Wait**:
   - Click "Deploy" and wait for pod to start
   - Monitor logs for initialization progress

### Step 4: Access Your Deployment

After successful deployment:

```bash
# Get pod endpoints from deployment script output or RunPod dashboard
export API_URL="https://your-pod-id-8000.proxy.runpod.net"

# Test health endpoint
curl $API_URL/health

# Access web client
open $API_URL/client
```

### RunPod Configuration Options

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `gpu_type` | GPU model to use | `"NVIDIA GeForce RTX 3090"` |
| `cloud_type` | `COMMUNITY` or `SECURE` | `COMMUNITY` (cheaper) |
| `volume_size` | Persistent storage (GB) | `50` |
| `container_disk` | Container disk (GB) | `20` |
| `max_batch_size` | Max concurrent processing | `4` |
| `ports` | Exposed ports | `"8000/http,8001/http"` |

## 🏠 Local Development Setup

### Prerequisites

```bash
# System requirements
- NVIDIA GPU (8GB+ VRAM recommended)
- CUDA 12.1+ installed
- Python 3.11+
- Docker (optional)
```

### Step 1: Environment Setup

```bash
# Clone repository
git clone <your-repository>
cd voxtral-streaming

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Configuration

Create environment file:

```bash
# Create .env file
cat > .env << EOF
MODEL_NAME=mistralai/Voxtral-Mini-3B-2507
DEVICE=cuda
ENABLE_TORCH_COMPILE=true
MAX_BATCH_SIZE=2  # Reduce for local development
TEMPERATURE=0.2
VAD_AGGRESSIVENESS=2
AUDIO_SAMPLE_RATE=16000
HOST=localhost
PORT=8000
LOG_LEVEL=debug
EOF
```

### Step 3: Run Development Server

```bash
# Start the server
cd src
python main.py

# Or use uvicorn directly
uvicorn main:app --host localhost --port 8000 --reload
```

### Step 4: Access Development Interface

```bash
# Open web browser
http://localhost:8000/client

# Test API
curl http://localhost:8000/health
```

### Development Tips

1. **Reduce Memory Usage**:
   ```bash
   export MAX_BATCH_SIZE=1
   export ENABLE_GRADIENT_CHECKPOINTING=true
   ```

2. **Enable Debug Logging**:
   ```bash
   export LOG_LEVEL=debug
   ```

3. **Use CPU for Testing** (no GPU):
   ```bash
   export DEVICE=cpu
   export ENABLE_TORCH_COMPILE=false
   ```

## 🐳 Docker Compose Deployment

For full-stack local deployment with monitoring:

### Step 1: Docker Compose Setup

```bash
# Ensure Docker and docker-compose are installed
docker --version
docker-compose --version

# Clone repository
git clone <your-repository>
cd voxtral-streaming
```

### Step 2: Configure Services

Edit `docker-compose.yml` if needed:

```yaml
version: '3.8'

services:
  voxtral-streaming:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NAME=mistralai/Voxtral-Mini-3B-2507
      - MAX_BATCH_SIZE=4
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Step 3: Start All Services

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f voxtral-streaming
```

### Step 4: Access Services

- **Main API**: http://localhost:8000
- **Web Client**: http://localhost:8000/client
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379

### Monitoring Setup

1. **Grafana Dashboards**:
   - Import pre-configured dashboards from `monitoring/grafana/`
   - Monitor real-time metrics, GPU usage, and connection health

2. **Prometheus Metrics**:
   - View raw metrics at http://localhost:9090
   - Set up alerting rules for production monitoring

## ☸️ Kubernetes Deployment

For production-scale deployments:

### Step 1: Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: voxtral-streaming
```

### Step 2: Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voxtral-streaming
  namespace: voxtral-streaming
spec:
  replicas: 2
  selector:
    matchLabels:
      app: voxtral-streaming
  template:
    metadata:
      labels:
        app: voxtral-streaming
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80  # or your GPU type
      containers:
      - name: voxtral-streaming
        image: your-username/voxtral-streaming:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        env:
        - name: MODEL_NAME
          value: "mistralai/Voxtral-Mini-3B-2507"
        - name: MAX_BATCH_SIZE
          value: "4"
        - name: ENABLE_TORCH_COMPILE
          value: "true"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Step 3: Create Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: voxtral-streaming-service
  namespace: voxtral-streaming
spec:
  selector:
    app: voxtral-streaming
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 8001
    targetPort: 8001
  type: LoadBalancer
```

### Step 4: Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get pods -n voxtral-streaming
kubectl get services -n voxtral-streaming

# View logs
kubectl logs -f deployment/voxtral-streaming -n voxtral-streaming
```

### Auto-Scaling Configuration

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voxtral-streaming-hpa
  namespace: voxtral-streaming
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voxtral-streaming
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔧 Production Configuration

### Environment Variables

Essential production environment variables:

```bash
# Model Configuration
MODEL_NAME=mistralai/Voxtral-Mini-3B-2507
DEVICE=cuda
ENABLE_TORCH_COMPILE=true
TORCH_COMPILE_MODE=max-autotune  # Best performance
MAX_BATCH_SIZE=8  # Adjust based on GPU memory

# Performance Tuning
ENABLE_CACHING=true
CACHE_SIZE=2000
PRELOAD_MODEL=true
WARMUP_ITERATIONS=5

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1  # GPU sharing limitation
LOG_LEVEL=info
ACCESS_LOG=true

# Security
CORS_ORIGINS=https://yourdomain.com
API_KEY_REQUIRED=true  # If implementing auth
RATE_LIMIT_ENABLED=true

# Monitoring
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=8001
ENABLE_HEALTH_CHECKS=true
LOG_DIR=/app/logs

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
VAD_AGGRESSIVENESS=2
ENABLE_NOISE_REDUCTION=true
SILENCE_TIMEOUT_MS=2000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0
NVIDIA_VISIBLE_DEVICES=all
```

### Performance Tuning

#### GPU Optimization

```bash
# For RTX 3090 (24GB VRAM)
MAX_BATCH_SIZE=8
TORCH_COMPILE_MODE=max-autotune
ENABLE_GRADIENT_CHECKPOINTING=false

# For RTX 4090 (24GB VRAM) 
MAX_BATCH_SIZE=12
TORCH_COMPILE_MODE=max-autotune
ENABLE_MIXED_PRECISION=true

# For A100 (40GB/80GB VRAM)
MAX_BATCH_SIZE=16
TORCH_COMPILE_MODE=max-autotune
ENABLE_FLASH_ATTENTION=true
```

#### Memory Management

```bash
# Conservative settings (8GB VRAM)
MAX_BATCH_SIZE=2
CACHE_SIZE=500
ENABLE_GRADIENT_CHECKPOINTING=true

# Aggressive settings (24GB+ VRAM)
MAX_BATCH_SIZE=16
CACHE_SIZE=5000
PRELOAD_MODEL=true
```

### Security Hardening

1. **Enable HTTPS/TLS**:
   ```bash
   # Use reverse proxy (nginx/traefik) for TLS termination
   # Or configure uvicorn with certificates
   uvicorn main:app --host 0.0.0.0 --port 8000 \
       --ssl-keyfile /path/to/private.key \
       --ssl-certfile /path/to/cert.pem
   ```

2. **Implement Rate Limiting**:
   ```python
   # Add to FastAPI app
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

   @app.post("/session/create")
   @limiter.limit("5/minute")
   async def create_session(request: Request):
       # Session creation logic
   ```

3. **Input Validation**:
   ```python
   from pydantic import BaseModel, validator

   class TextInput(BaseModel):
       text: str

       @validator('text')
       def validate_text(cls, v):
           if len(v) > 1000:
               raise ValueError('Text too long')
           return v.strip()
   ```

### Monitoring and Alerting

#### Prometheus Rules

Create alerting rules in `prometheus.yml`:

```yaml
groups:
- name: voxtral-streaming
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(audio_processing_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High audio processing latency detected

  - alert: GPUMemoryHigh
    expr: gpu_memory_allocated_bytes / gpu_memory_total_bytes > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: GPU memory usage is critically high

  - alert: ServiceDown
    expr: up{job="voxtral-streaming"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Voxtral streaming service is down
```

#### Health Check Monitoring

```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
HEALTH_URL="${API_URL:-http://localhost:8000}/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")

if [ "$RESPONSE" != "200" ]; then
    echo "Health check failed with status: $RESPONSE"
    exit 1
else
    echo "Health check passed"
    exit 0
fi
EOF

chmod +x health_check.sh

# Run as cron job every minute
echo "* * * * * /path/to/health_check.sh" | crontab -
```

## 🐛 Troubleshooting

### Common Deployment Issues

#### 1. CUDA Out of Memory

```bash
# Symptoms
RuntimeError: CUDA out of memory

# Solutions
export MAX_BATCH_SIZE=1
export ENABLE_GRADIENT_CHECKPOINTING=true
export TORCH_COMPILE_MODE=reduce-overhead
```

#### 2. Model Loading Failures

```bash
# Symptoms
OSError: Can't load model

# Solutions
# Pre-download model
export PRELOAD_MODEL=true

# Check disk space
df -h /app/cache

# Verify network connectivity
curl -I https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
```

#### 3. WebSocket Connection Issues

```bash
# Symptoms
WebSocket connection failed

# Check firewall
sudo ufw status
sudo ufw allow 8000
sudo ufw allow 8001

# Check proxy configuration
# For nginx:
location /ws/ {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

#### 4. Audio Processing Errors

```bash
# Symptoms
Audio format not supported

# Solutions
# Check audio configuration
export AUDIO_SAMPLE_RATE=16000
export AUDIO_CHANNELS=1

# Verify VAD libraries
pip install webrtcvad silero-vad
```

### Performance Debugging

#### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU metrics
nvidia-ml-py3
```

#### Profile Application

```python
# Add profiling to main.py
import torch.profiler
import cProfile
import pstats

# Enable PyTorch profiler
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your inference code here
    prof.step()
```

#### Check System Resources

```bash
# Monitor system resources
htop
iotop
nethogs

# Check memory leaks
valgrind --tool=memcheck python main.py
```

### Log Analysis

#### Enable Detailed Logging

```python
# In logging_config.py
import structlog

logger = structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

#### Centralized Logging

```yaml
# fluentd or filebeat configuration
inputs:
  - type: container
    paths:
      - '/var/lib/docker/containers/*/*.log'

outputs:
  - type: elasticsearch
    hosts: ["elasticsearch:9200"]
  - type: console
```

## 📊 Deployment Checklist

### Pre-Deployment

- [ ] Docker image built and tested
- [ ] Environment variables configured
- [ ] GPU drivers installed and verified
- [ ] Network ports opened (8000, 8001)
- [ ] SSL certificates configured (production)
- [ ] Monitoring setup (Prometheus/Grafana)
- [ ] Backup and recovery plan
- [ ] Load testing completed
- [ ] Security review completed

### Post-Deployment

- [ ] Health checks passing
- [ ] WebSocket connectivity verified
- [ ] Audio processing tested
- [ ] Performance metrics baseline established
- [ ] Alerting rules configured
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Incident response procedures defined

### Production Readiness

- [ ] Auto-scaling configured
- [ ] Load balancer setup
- [ ] Database/Redis clustering
- [ ] Log aggregation enabled
- [ ] Backup automation
- [ ] Disaster recovery tested
- [ ] Security scanning completed
- [ ] Performance optimization verified

---

For additional support, check the [API Documentation](API.md) or [GitHub Issues](https://github.com/your-repo/voxtral-streaming/issues).
