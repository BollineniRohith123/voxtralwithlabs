"""
WebSocket connection manager for real-time audio streaming
Handles connection lifecycle, message routing, and error recovery
"""
import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Callable
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_RESPONSE = "audio_response"
    TEXT_INPUT = "text_input"
    TEXT_RESPONSE = "text_response"
    CONTROL = "control"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class ClientInfo:
    client_id: str
    websocket: WebSocket
    session_id: str
    connected_at: float
    last_activity: float
    message_count: int = 0
    error_count: int = 0
    is_active: bool = True

class WebSocketManager:
    """Manages WebSocket connections and message routing"""

    def __init__(self, config):
        self.config = config

        # Connection management
        self.clients: Dict[str, ClientInfo] = {}
        self.session_clients: Dict[str, Set[str]] = defaultdict(set)

        # Message queues
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}

        # Performance tracking
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_processed": 0,
            "errors_encountered": 0
        }

        # Heartbeat management
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_task = None

        # Thread safety
        self.clients_lock = threading.RLock()

        logger.info("🔌 WebSocket manager initialized")

    async def connect(self, client_id: str, websocket: WebSocket, session_id: str):
        """Register a new WebSocket connection"""
        try:
            with self.clients_lock:
                # Create client info
                client_info = ClientInfo(
                    client_id=client_id,
                    websocket=websocket,
                    session_id=session_id,
                    connected_at=time.time(),
                    last_activity=time.time()
                )

                # Register client
                self.clients[client_id] = client_info
                self.session_clients[session_id].add(client_id)

                # Create message queue for client
                self.message_queues[client_id] = asyncio.Queue(maxsize=1000)

                # Update stats
                self.connection_stats["total_connections"] += 1
                self.connection_stats["active_connections"] += 1

                logger.info(f"🔌 Client connected: {client_id} (session: {session_id})")

                # Send welcome message
                await self.send_message(client_id, {
                    "type": MessageType.STATUS.value,
                    "status": "connected",
                    "client_id": client_id,
                    "session_id": session_id,
                    "server_time": time.time()
                })

                # Start heartbeat if first client
                if len(self.clients) == 1 and not self.heartbeat_task:
                    self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        except Exception as e:
            logger.error(f"❌ Failed to connect client {client_id}: {e}")
            raise

    async def disconnect(self, client_id: str):
        """Unregister a WebSocket connection"""
        try:
            with self.clients_lock:
                if client_id not in self.clients:
                    return

                client_info = self.clients[client_id]
                session_id = client_info.session_id

                # Remove from session clients
                if session_id in self.session_clients:
                    self.session_clients[session_id].discard(client_id)
                    if not self.session_clients[session_id]:
                        del self.session_clients[session_id]

                # Cancel processing task
                if client_id in self.processing_tasks:
                    task = self.processing_tasks[client_id]
                    if not task.done():
                        task.cancel()
                    del self.processing_tasks[client_id]

                # Clear message queue
                if client_id in self.message_queues:
                    del self.message_queues[client_id]

                # Remove client
                del self.clients[client_id]

                # Update stats
                self.connection_stats["active_connections"] -= 1

                logger.info(f"🔌 Client disconnected: {client_id}")

                # Stop heartbeat if no clients
                if not self.clients and self.heartbeat_task:
                    self.heartbeat_task.cancel()
                    self.heartbeat_task = None

        except Exception as e:
            logger.error(f"❌ Failed to disconnect client {client_id}: {e}")

    async def handle_client(self, client_id: str):
        """Main message handling loop for a client"""
        if client_id not in self.clients:
            logger.error(f"❌ Client not found: {client_id}")
            return

        client_info = self.clients[client_id]
        websocket = client_info.websocket

        # Start processing task
        self.processing_tasks[client_id] = asyncio.create_task(
            self._process_client_messages(client_id)
        )

        try:
            while client_info.is_active:
                # Receive message from client
                try:
                    # Handle both text and binary messages
                    if hasattr(websocket, 'receive'):
                        message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                    else:
                        break

                    # Update activity
                    client_info.last_activity = time.time()
                    client_info.message_count += 1

                    # Process message based on type
                    if "bytes" in message:
                        # Binary audio data
                        await self._handle_audio_message(client_id, message["bytes"])
                    elif "text" in message:
                        # Text/JSON message
                        await self._handle_text_message(client_id, message["text"])

                except asyncio.TimeoutError:
                    # Timeout waiting for message
                    logger.debug(f"⏱️ Message timeout for client {client_id}")
                    continue
                except WebSocketDisconnect:
                    logger.info(f"🔌 Client disconnected: {client_id}")
                    break
                except Exception as e:
                    logger.error(f"❌ Message handling error for {client_id}: {e}")
                    client_info.error_count += 1

                    # Disconnect if too many errors
                    if client_info.error_count > 10:
                        logger.error(f"❌ Too many errors for client {client_id}, disconnecting")
                        break

                    await self.send_error(client_id, f"Message processing error: {str(e)}")

        except Exception as e:
            logger.error(f"❌ Client handling error: {e}")
        finally:
            client_info.is_active = False
            await self.disconnect(client_id)

    async def _handle_audio_message(self, client_id: str, audio_data: bytes):
        """Handle incoming audio data"""
        try:
            # Add to processing queue
            message = {
                "type": MessageType.AUDIO_CHUNK.value,
                "client_id": client_id,
                "data": audio_data,
                "timestamp": time.time()
            }

            # Queue for processing (non-blocking)
            if client_id in self.message_queues:
                try:
                    self.message_queues[client_id].put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(f"⚠️ Message queue full for client {client_id}")
                    # Remove oldest message and add new one
                    try:
                        self.message_queues[client_id].get_nowait()
                        self.message_queues[client_id].put_nowait(message)
                    except asyncio.QueueEmpty:
                        pass

            self.connection_stats["messages_processed"] += 1

        except Exception as e:
            logger.error(f"❌ Audio message handling error: {e}")
            await self.send_error(client_id, f"Audio processing error: {str(e)}")

    async def _handle_text_message(self, client_id: str, text_data: str):
        """Handle incoming text/JSON messages"""
        try:
            # Parse JSON message
            try:
                message_data = json.loads(text_data)
            except json.JSONDecodeError:
                # Treat as plain text
                message_data = {
                    "type": MessageType.TEXT_INPUT.value,
                    "text": text_data
                }

            # Add metadata
            message_data["client_id"] = client_id
            message_data["timestamp"] = time.time()

            # Queue for processing
            if client_id in self.message_queues:
                try:
                    self.message_queues[client_id].put_nowait(message_data)
                except asyncio.QueueFull:
                    logger.warning(f"⚠️ Message queue full for client {client_id}")

            self.connection_stats["messages_processed"] += 1

        except Exception as e:
            logger.error(f"❌ Text message handling error: {e}")
            await self.send_error(client_id, f"Text processing error: {str(e)}")

    async def _process_client_messages(self, client_id: str):
        """Background task to process client messages"""
        from models.voxtral_inference import VoxtralInferenceEngine
        from audio.processor import AudioProcessor

        # Get global processors (injected via main app)
        voxtral_engine = None
        audio_processor = None

        # Import globals from main module
        import sys
        if 'main' in sys.modules:
            main_module = sys.modules['main']
            voxtral_engine = getattr(main_module, 'voxtral_engine', None)
            audio_processor = getattr(main_module, 'audio_processor', None)

        if not voxtral_engine or not audio_processor:
            logger.error(f"❌ Missing processors for client {client_id}")
            return

        try:
            while client_id in self.clients and self.clients[client_id].is_active:
                try:
                    # Get message from queue
                    message = await asyncio.wait_for(
                        self.message_queues[client_id].get(),
                        timeout=1.0
                    )

                    message_type = message.get("type")

                    if message_type == MessageType.AUDIO_CHUNK.value:
                        # Process audio
                        audio_data = message.get("data")
                        if audio_data:
                            session_id = self.clients[client_id].session_id

                            # Process audio through pipeline
                            processed_audio = await audio_processor.process_audio_chunk(
                                audio_data, session_id
                            )

                            if processed_audio is not None:
                                # Run through Voxtral
                                response = await voxtral_engine.process_audio_streaming(
                                    processed_audio, session_id
                                )

                                # Send response back
                                await self.send_message(client_id, {
                                    "type": MessageType.TEXT_RESPONSE.value,
                                    "text": response,
                                    "timestamp": time.time()
                                })

                    elif message_type == MessageType.TEXT_INPUT.value:
                        # Process text input
                        text = message.get("text")
                        if text:
                            session_id = self.clients[client_id].session_id

                            # Process through Voxtral
                            response = await voxtral_engine.process_text(text, session_id)

                            # Send response
                            await self.send_message(client_id, {
                                "type": MessageType.TEXT_RESPONSE.value,
                                "text": response,
                                "timestamp": time.time()
                            })

                    elif message_type == MessageType.CONTROL.value:
                        # Handle control messages
                        await self._handle_control_message(client_id, message)

                except asyncio.TimeoutError:
                    # No messages to process
                    continue
                except Exception as e:
                    logger.error(f"❌ Message processing error for {client_id}: {e}")
                    self.connection_stats["errors_encountered"] += 1

        except Exception as e:
            logger.error(f"❌ Client processing task error: {e}")
        finally:
            logger.info(f"🔄 Processing task ended for client {client_id}")

    async def _handle_control_message(self, client_id: str, message: Dict):
        """Handle control messages"""
        try:
            control_type = message.get("control_type")

            if control_type == "ping":
                await self.send_message(client_id, {
                    "type": MessageType.CONTROL.value,
                    "control_type": "pong",
                    "timestamp": time.time()
                })
            elif control_type == "get_stats":
                stats = self.get_client_stats(client_id)
                await self.send_message(client_id, {
                    "type": MessageType.STATUS.value,
                    "stats": stats
                })

        except Exception as e:
            logger.error(f"❌ Control message error: {e}")

    async def send_message(self, client_id: str, message: Dict):
        """Send message to specific client"""
        if client_id not in self.clients:
            return False

        client_info = self.clients[client_id]
        if not client_info.is_active:
            return False

        try:
            # Send JSON message
            await client_info.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send message to {client_id}: {e}")
            client_info.error_count += 1
            return False

    async def send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self.send_message(client_id, {
            "type": MessageType.ERROR.value,
            "error": error_message,
            "timestamp": time.time()
        })

    async def broadcast_to_session(self, session_id: str, message: Dict):
        """Broadcast message to all clients in a session"""
        if session_id not in self.session_clients:
            return

        client_ids = list(self.session_clients[session_id])

        for client_id in client_ids:
            await self.send_message(client_id, message)

    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        try:
            while self.clients:
                await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat to all clients
                current_time = time.time()
                disconnected_clients = []

                for client_id, client_info in list(self.clients.items()):
                    # Check if client is still active
                    if current_time - client_info.last_activity > 60:  # 1 minute timeout
                        logger.warning(f"⏱️ Client {client_id} inactive, marking for disconnect")
                        disconnected_clients.append(client_id)
                    else:
                        # Send heartbeat
                        await self.send_message(client_id, {
                            "type": MessageType.HEARTBEAT.value,
                            "timestamp": current_time
                        })

                # Disconnect inactive clients
                for client_id in disconnected_clients:
                    await self.disconnect(client_id)

        except asyncio.CancelledError:
            logger.info("💓 Heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"❌ Heartbeat loop error: {e}")

    def get_client_stats(self, client_id: str) -> Dict:
        """Get statistics for a specific client"""
        if client_id not in self.clients:
            return {}

        client_info = self.clients[client_id]

        return {
            "client_id": client_id,
            "session_id": client_info.session_id,
            "connected_at": client_info.connected_at,
            "last_activity": client_info.last_activity,
            "connection_duration": time.time() - client_info.connected_at,
            "message_count": client_info.message_count,
            "error_count": client_info.error_count,
            "queue_size": self.message_queues[client_id].qsize() if client_id in self.message_queues else 0
        }

    def get_global_stats(self) -> Dict:
        """Get global WebSocket statistics"""
        return {
            **self.connection_stats,
            "active_sessions": len(self.session_clients),
            "total_queued_messages": sum(
                queue.qsize() for queue in self.message_queues.values()
            ),
            "uptime": time.time() - (
                min(client.connected_at for client in self.clients.values())
                if self.clients else time.time()
            )
        }
