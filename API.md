# Voxtral 3B Streaming API Documentation

## Overview

The Voxtral 3B Real-Time Streaming API provides WebSocket-based audio streaming capabilities for conversational AI interactions. The API supports both binary audio data and JSON text messages for maximum flexibility.

## Base URL

```
Production: https://your-runpod-instance.proxy.runpod.net
Development: http://localhost:8000
WebSocket: wss://your-runpod-instance.proxy.runpod.net (or ws://localhost:8000)
```

## Authentication

Currently, the API operates without authentication. In production deployments, consider implementing:
- API key authentication
- JWT tokens
- IP whitelisting
- Rate limiting

## REST Endpoints

### Health Check

#### GET /health

Returns the current health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200.0,
  "system": {
    "cpu_percent": 25.0,
    "memory_percent": 60.0,
    "disk_usage": 45.0
  },
  "gpu": {
    "gpu_count": 1,
    "gpu_memory": 24576000000,
    "gpu_memory_allocated": 8589934592,
    "gpu_memory_cached": 1073741824
  },
  "active_sessions": 3,
  "queue_size": 0
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service unavailable

### Create Session

#### POST /session/create

Creates a new conversation session.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created",
  "websocket_url": "/ws/audio/550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- `200`: Session created successfully
- `500`: Failed to create session

### Delete Session

#### DELETE /session/{session_id}

Deletes an existing conversation session.

**Parameters:**
- `session_id` (path): UUID of the session to delete

**Response:**
```json
{
  "status": "deleted",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- `200`: Session deleted successfully
- `404`: Session not found
- `500`: Failed to delete session

### Metrics

#### GET /metrics

Returns Prometheus-formatted metrics for monitoring.

**Response:**
```
# HELP websocket_connections_total Total WebSocket connections
# TYPE websocket_connections_total counter
websocket_connections_total 127

# HELP websocket_connections_active Active WebSocket connections
# TYPE websocket_connections_active gauge
websocket_connections_active 5

# HELP audio_processing_duration_seconds Audio processing time
# TYPE audio_processing_duration_seconds histogram
audio_processing_duration_seconds_bucket{le="0.005"} 0
audio_processing_duration_seconds_bucket{le="0.01"} 2
audio_processing_duration_seconds_bucket{le="0.025"} 8
audio_processing_duration_seconds_bucket{le="0.05"} 15
```

## WebSocket API

### Audio Streaming

#### WS /ws/audio/{session_id}

Real-time audio streaming endpoint for bidirectional communication.

**Connection Parameters:**
- `session_id` (path): UUID of the session (created via REST API)

### Message Types

The WebSocket endpoint supports both binary and text messages:

#### Binary Messages (Audio Data)

Send raw audio data as binary WebSocket frames.

**Format:**
- **Encoding**: 16-bit PCM
- **Sample Rate**: 16kHz (configurable)
- **Channels**: Mono (1 channel)
- **Byte Order**: Little-endian

**Example (JavaScript):**
```javascript
// Convert audio to Int16Array
const int16Array = new Int16Array(audioBuffer.length);
for (let i = 0; i < audioBuffer.length; i++) {
    int16Array[i] = Math.max(-32768, Math.min(32767, audioBuffer[i] * 32767));
}

// Send binary data
websocket.send(int16Array.buffer);
```

#### Text Messages (JSON)

Send JSON messages for text input and control commands.

### Outbound Message Types (Client → Server)

#### Text Input

Send text for processing by the Voxtral model.

```json
{
  "type": "text_input",
  "text": "Hello, how are you today?",
  "timestamp": 1640995200.0
}
```

#### Control Messages

Send control commands to manage the session.

```json
{
  "type": "control",
  "control_type": "ping",
  "timestamp": 1640995200.0
}
```

**Control Types:**
- `ping`: Request pong response for latency measurement
- `get_stats`: Request session statistics

### Inbound Message Types (Server → Client)

#### Status Messages

Receive status updates and connection information.

```json
{
  "type": "status",
  "status": "connected",
  "client_id": "client_12345",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "server_time": 1640995200.0
}
```

#### Text Responses

Receive processed text responses from the Voxtral model.

```json
{
  "type": "text_response",
  "text": "I'm doing very well, thank you for asking! How can I help you today?",
  "timestamp": 1640995201.5
}
```

#### Error Messages

Receive error information when processing fails.

```json
{
  "type": "error",
  "error": "Audio processing failed: Invalid format",
  "timestamp": 1640995201.0
}
```

#### Heartbeat Messages

Receive periodic heartbeat messages for connection monitoring.

```json
{
  "type": "heartbeat",
  "timestamp": 1640995202.0
}
```

#### Control Responses

Receive responses to control messages.

```json
{
  "type": "control",
  "control_type": "pong",
  "timestamp": 1640995202.1
}
```

```json
{
  "type": "status",
  "stats": {
    "client_id": "client_12345",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "connection_duration": 125.5,
    "message_count": 48,
    "error_count": 0,
    "queue_size": 2
  }
}
```

### Control WebSocket

#### WS /ws/control/{session_id}

Separate endpoint for control messages and text-only communication.

**Text Input:**
```json
{
  "type": "text_input",
  "text": "What is the weather like today?"
}
```

**Configuration Update:**
```json
{
  "type": "config_update",
  "config": {
    "vad_sensitivity": 3,
    "temperature": 0.3,
    "max_length": 256
  }
}
```

## Client Implementation Examples

### JavaScript WebRTC Client

```javascript
class VoxtralClient {
    constructor(serverUrl) {
        this.serverUrl = serverUrl;
        this.websocket = null;
        this.audioContext = null;
        this.sessionId = null;
    }

    async createSession() {
        const response = await fetch(`${this.serverUrl}/session/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();
        this.sessionId = data.session_id;
        return data;
    }

    async connect() {
        const wsUrl = `${this.serverUrl.replace('http', 'ws')}/ws/audio/${this.sessionId}`;
        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('Connected to Voxtral streaming service');
        };

        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        return new Promise((resolve) => {
            this.websocket.onopen = resolve;
        });
    }

    async startAudioStreaming() {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(stream);

        // Create audio processor
        const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (event) => {
            const audioData = event.inputBuffer.getChannelData(0);
            this.sendAudioData(audioData);
        };

        source.connect(processor);
        processor.connect(this.audioContext.destination);
    }

    sendAudioData(audioBuffer) {
        if (this.websocket.readyState === WebSocket.OPEN) {
            // Convert to Int16Array
            const int16Array = new Int16Array(audioBuffer.length);
            for (let i = 0; i < audioBuffer.length; i++) {
                int16Array[i] = Math.max(-32768, Math.min(32767, audioBuffer[i] * 32767));
            }

            this.websocket.send(int16Array.buffer);
        }
    }

    sendTextMessage(text) {
        if (this.websocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'text_input',
                text: text,
                timestamp: Date.now() / 1000
            };

            this.websocket.send(JSON.stringify(message));
        }
    }

    handleMessage(message) {
        switch (message.type) {
            case 'text_response':
                console.log('Response:', message.text);
                break;
            case 'error':
                console.error('Error:', message.error);
                break;
            case 'heartbeat':
                // Calculate latency
                const latency = Date.now() - (message.timestamp * 1000);
                console.log(`Latency: ${latency}ms`);
                break;
        }
    }
}

// Usage
const client = new VoxtralClient('http://localhost:8000');
await client.createSession();
await client.connect();
await client.startAudioStreaming();
```

### Python Client Example

```python
import asyncio
import websockets
import json
import pyaudio
import numpy as np

class VoxtralPythonClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.websocket = None
        self.session_id = None

    async def create_session(self):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.server_url}/session/create") as response:
                data = await response.json()
                self.session_id = data['session_id']
                return data

    async def connect(self):
        ws_url = f"{self.server_url.replace('http', 'ws')}/ws/audio/{self.session_id}"
        self.websocket = await websockets.connect(ws_url)

        # Start message handler
        asyncio.create_task(self.message_handler())

    async def message_handler(self):
        async for message in self.websocket:
            try:
                data = json.loads(message)
                await self.handle_message(data)
            except json.JSONDecodeError:
                # Binary message (audio response)
                pass

    async def handle_message(self, message):
        if message['type'] == 'text_response':
            print(f"Response: {message['text']}")
        elif message['type'] == 'error':
            print(f"Error: {message['error']}")

    async def send_text(self, text):
        if self.websocket:
            message = {
                'type': 'text_input',
                'text': text,
                'timestamp': time.time()
            }
            await self.websocket.send(json.dumps(message))

    async def stream_audio(self):
        # PyAudio setup
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        try:
            while True:
                audio_data = stream.read(1024)
                if self.websocket:
                    await self.websocket.send(audio_data)
                await asyncio.sleep(0.01)  # 10ms chunks
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

# Usage
async def main():
    client = VoxtralPythonClient('http://localhost:8000')
    await client.create_session()
    await client.connect()

    # Start audio streaming in background
    audio_task = asyncio.create_task(client.stream_audio())

    # Send text message
    await client.send_text("Hello, how are you?")

    # Keep running
    await audio_task

asyncio.run(main())
```

## Error Handling

### Common Error Codes

| Error | Description | Resolution |
|-------|-------------|------------|
| `CONNECTION_FAILED` | WebSocket connection failed | Check network connectivity and server status |
| `SESSION_NOT_FOUND` | Session ID not found | Create a new session |
| `AUDIO_FORMAT_ERROR` | Invalid audio format | Ensure 16-bit PCM, 16kHz, mono |
| `PROCESSING_ERROR` | Model processing failed | Check audio quality and retry |
| `RATE_LIMITED` | Too many requests | Implement exponential backoff |
| `GPU_OOM` | GPU out of memory | Reduce batch size or audio chunk length |

### Retry Logic

Implement exponential backoff for connection failures:

```javascript
async function connectWithRetry(maxRetries = 5) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            await client.connect();
            return;
        } catch (error) {
            const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
            console.log(`Connection attempt ${attempt} failed, retrying in ${delay}ms`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw new Error('Failed to connect after maximum retries');
}
```

## Rate Limiting

The API implements the following rate limits:

- **WebSocket Connections**: 10 per minute per IP
- **Session Creation**: 5 per minute per IP
- **Audio Data**: 1MB per second per session
- **Text Messages**: 60 per minute per session

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995260
```

## Performance Considerations

### Latency Optimization

1. **Use Regional Deployment**: Deploy close to users
2. **Enable torch.compile**: Set `ENABLE_TORCH_COMPILE=true`
3. **Optimize Audio Parameters**: Use 16kHz sample rate, 160ms chunks
4. **Enable Caching**: Response caching for repeated queries

### Bandwidth Optimization

1. **Audio Compression**: Use Opus codec for WebRTC
2. **Chunked Processing**: Send optimal chunk sizes (160-320ms)
3. **VAD Filtering**: Only send audio during speech

### Scaling Guidelines

- **Concurrent Sessions**: Up to 50 per GPU (RTX 3090)
- **Memory Usage**: ~200MB per active session
- **CPU Usage**: 1-2 cores per 10 sessions
- **Network Bandwidth**: ~64kbps per audio stream

## Monitoring and Observability

### Key Metrics to Monitor

1. **Connection Metrics**:
   - Active WebSocket connections
   - Connection success rate
   - Average connection duration

2. **Processing Metrics**:
   - Audio processing latency (p50, p95, p99)
   - Model inference time
   - Queue depth and wait times

3. **System Metrics**:
   - GPU utilization and memory
   - CPU usage and memory consumption
   - Network I/O and error rates

4. **Business Metrics**:
   - Messages processed per hour
   - Session success rate
   - User engagement duration

### Health Check Endpoints

Monitor these endpoints for service health:

- `GET /health`: Basic health status
- `GET /metrics`: Detailed Prometheus metrics
- `WS /ws/audio/{session_id}`: WebSocket connectivity test

## Security Considerations

### Production Deployment

1. **Enable HTTPS/WSS**: Use TLS encryption for all connections
2. **Input Validation**: Sanitize all text inputs
3. **Rate Limiting**: Implement per-user rate limits
4. **Authentication**: Add API key or JWT authentication
5. **CORS Configuration**: Restrict allowed origins
6. **Monitoring**: Log all API access and errors

### Audio Data Privacy

1. **No Persistent Storage**: Audio data is processed in memory only
2. **Session Isolation**: Each session is completely isolated
3. **Automatic Cleanup**: Sessions expire after inactivity
4. **Encryption**: All data transmission is encrypted

---

For more information, see the [GitHub repository](https://github.com/your-repo/voxtral-streaming) or contact support.
