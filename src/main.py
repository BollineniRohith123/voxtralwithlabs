"""
Production-ready FastAPI application for Voxtral 3B real-time streaming
Supports WebRTC audio streaming with ultra-low latency (<200ms)
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import torch
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from models.voxtral_inference import VoxtralInferenceEngine
from streaming.websocket_manager import WebSocketManager
from streaming.session_manager import SessionManager
from streaming.queue_manager import QueueManager
from audio.audio_processor import AudioProcessor
from utils.config import Config
from utils.logging_config import setup_logging
from utils.metrics import MetricsCollector

# Initialize logging
logger = setup_logging()
# Metrics
connection_counter = Counter('websocket_connections_total', 'Total WebSocket connections')
active_connections = Gauge('websocket_connections_active', 'Active WebSocket connections')
processing_time = Histogram('audio_processing_duration_seconds', 'Audio processing time')
model_inference_time = Histogram('model_inference_duration_seconds', 'Model inference time')

# Global managers
websocket_manager: Optional[WebSocketManager] = None
session_manager: Optional[SessionManager] = None
queue_manager: Optional[QueueManager] = None
audio_processor: Optional[AudioProcessor] = None
voxtral_engine: Optional[VoxtralInferenceEngine] = None
metrics_collector: Optional[MetricsCollector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global websocket_manager, session_manager, queue_manager, audio_processor, voxtral_engine, metrics_collector

    logger.info("🚀 Starting Voxtral 3B Real-Time Streaming Server...")

    # Initialize configuration
    config = Config()

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Initialize Voxtral inference engine
    logger.info("📦 Loading Voxtral 3B model...")
    voxtral_engine = VoxtralInferenceEngine(config)
    await voxtral_engine.initialize()

    # Initialize audio processor
    audio_processor = AudioProcessor(config)
    await audio_processor.initialize()

    # Initialize managers
    queue_manager = QueueManager(config)
    session_manager = SessionManager(config)
    websocket_manager = WebSocketManager(config)

    logger.info("✅ Server initialization complete")

    yield

    # Cleanup
    logger.info("🛑 Shutting down server...")
    if voxtral_engine:
        await voxtral_engine.cleanup()
    if audio_processor:
        await audio_processor.cleanup()
    logger.info("✅ Server shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Voxtral 3B Real-Time Streaming API",
    description="Production-ready real-time conversational AI with WebRTC streaming",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Voxtral 3B Streaming API",
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            "gpu_memory_cached": torch.cuda.memory_reserved(0)
        }

    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "gpu": gpu_info,
        "active_sessions": len(session_manager.active_sessions) if session_manager else 0,
        "queue_size": queue_manager.get_queue_size() if queue_manager else 0
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/client", response_class=HTMLResponse)
async def get_client():
    """Serve WebRTC client interface"""
    with open("frontend/index.html", "r") as f:
        return f.read()

@app.post("/session/create")
async def create_session():
    """Create a new conversation session"""
    try:
        session_id = await session_manager.create_session()
        return {
            "session_id": session_id,
            "status": "created",
            "websocket_url": f"/ws/audio/{session_id}"
        }
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    try:
        await session_manager.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    client_id = f"{session_id}_{id(websocket)}"

    try:
        # Accept WebSocket connection
        await websocket.accept()
        connection_counter.inc()
        active_connections.inc()

        logger.info(f"🔌 WebSocket connected: {client_id}")

        # Register client with websocket manager
        await websocket_manager.connect(client_id, websocket, session_id)

        # Start processing loop
        await websocket_manager.handle_client(client_id)

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for {client_id}: {e}")
    finally:
        # Cleanup
        active_connections.dec()
        if websocket_manager:
            await websocket_manager.disconnect(client_id)

@app.websocket("/ws/control/{session_id}")
async def websocket_control_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for control messages and text input"""
    try:
        await websocket.accept()
        logger.info(f"🎛️ Control WebSocket connected: {session_id}")

        while True:
            # Receive control messages
            message = await websocket.receive_json()

            if message.get("type") == "text_input":
                # Handle text input
                text = message.get("text", "")
                if text:
                    # Process text through Voxtral
                    response = await voxtral_engine.process_text(text, session_id)
                    await websocket.send_json({
                        "type": "text_response",
                        "text": response
                    })

            elif message.get("type") == "config_update":
                # Handle configuration updates
                config_update = message.get("config", {})
                await session_manager.update_session_config(session_id, config_update)
                await websocket.send_json({
                    "type": "config_updated",
                    "status": "success"
                })

    except WebSocketDisconnect:
        logger.info(f"🎛️ Control WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ Control WebSocket error: {e}")

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        loop="asyncio",
        workers=1  # Single worker for GPU sharing
    )
