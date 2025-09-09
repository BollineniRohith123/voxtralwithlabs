"""
Real-time audio processing pipeline with streaming capabilities
Includes VAD, chunking, noise reduction, and format conversion
"""
import asyncio
import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Union
import io
import threading
from collections import deque
from dataclasses import dataclass
import time

# VAD implementations
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    logging.warning("webrtcvad not available, falling back to energy-based VAD")

try:
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    logging.warning("Silero VAD not available")

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size_ms: int = 160  # 10ms chunks for WebRTC VAD
    overlap_ms: int = 80      # 50% overlap
    buffer_size_ms: int = 1000  # 1 second buffer
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    vad_frame_duration_ms: int = 30  # 10, 20, or 30ms
    energy_threshold: float = 0.01
    silence_timeout_ms: int = 1000  # 1 second of silence
    enable_noise_reduction: bool = True
    enable_auto_gain: bool = True

class VADProcessor:
    """Voice Activity Detection processor with multiple VAD engines"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.webrtc_vad = None
        self.silero_vad_model = None
        self.vad_type = "energy"  # fallback

        # Initialize VAD engines
        self._initialize_vad()

        # State tracking
        self.speech_frames = deque(maxlen=10)
        self.silence_duration = 0

    def _initialize_vad(self):
        """Initialize available VAD engines"""
        # Try Silero VAD first (most accurate)
        if SILERO_VAD_AVAILABLE:
            try:
                self.silero_vad_model = load_silero_vad()
                self.vad_type = "silero"
                logger.info("✅ Initialized Silero VAD")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Silero VAD: {e}")

        # Try WebRTC VAD
        if WEBRTC_VAD_AVAILABLE:
            try:
                self.webrtc_vad = webrtcvad.Vad(self.config.vad_aggressiveness)
                self.vad_type = "webrtc"
                logger.info("✅ Initialized WebRTC VAD")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC VAD: {e}")

        # Fallback to energy-based VAD
        self.vad_type = "energy"
        logger.info("✅ Using energy-based VAD (fallback)")

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process audio frame and return True if speech detected"""
        try:
            if self.vad_type == "silero":
                return self._silero_vad(audio_frame)
            elif self.vad_type == "webrtc":
                return self._webrtc_vad(audio_frame)
            else:
                return self._energy_vad(audio_frame)
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            return self._energy_vad(audio_frame)

    def _silero_vad(self, audio_frame: np.ndarray) -> bool:
        """Silero VAD processing"""
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_frame)

            # Get speech probability
            speech_prob = self.silero_vad_model(audio_tensor, self.config.sample_rate).item()

            # Threshold for speech detection
            is_speech = speech_prob > 0.5

            self.speech_frames.append(is_speech)
            return is_speech

        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return self._energy_vad(audio_frame)

    def _webrtc_vad(self, audio_frame: np.ndarray) -> bool:
        """WebRTC VAD processing"""
        try:
            # Convert to 16-bit PCM
            audio_pcm = (audio_frame * 32767).astype(np.int16)

            # WebRTC VAD requires specific frame sizes
            frame_size = int(self.config.sample_rate * self.config.vad_frame_duration_ms / 1000)

            # Pad or truncate to required size
            if len(audio_pcm) < frame_size:
                audio_pcm = np.pad(audio_pcm, (0, frame_size - len(audio_pcm)))
            elif len(audio_pcm) > frame_size:
                audio_pcm = audio_pcm[:frame_size]

            # Convert to bytes
            audio_bytes = audio_pcm.tobytes()

            # VAD processing
            is_speech = self.webrtc_vad.is_speech(audio_bytes, self.config.sample_rate)

            self.speech_frames.append(is_speech)
            return is_speech

        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return self._energy_vad(audio_frame)

    def _energy_vad(self, audio_frame: np.ndarray) -> bool:
        """Energy-based VAD (fallback)"""
        try:
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_frame ** 2))

            # Simple threshold-based detection
            is_speech = energy > self.config.energy_threshold

            self.speech_frames.append(is_speech)
            return is_speech

        except Exception as e:
            logger.error(f"Energy VAD error: {e}")
            return False

    def is_speaking(self) -> bool:
        """Check if currently speaking based on recent frames"""
        if not self.speech_frames:
            return False

        # Consider speaking if majority of recent frames contain speech
        speech_ratio = sum(self.speech_frames) / len(self.speech_frames)
        return speech_ratio > 0.3

    def reset(self):
        """Reset VAD state"""
        self.speech_frames.clear()
        self.silence_duration = 0

class AudioProcessor:
    """Real-time audio processing pipeline"""

    def __init__(self, config):
        self.config = config
        self.audio_config = AudioConfig()
        self.vad_processor = VADProcessor(self.audio_config)

        # Audio buffers
        self.audio_buffer = deque()
        self.processed_chunks = deque()

        # Processing state
        self.is_processing = False
        self.processing_lock = threading.Lock()

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.chunks_processed = 0

        logger.info("🎵 Audio processor initialized")

    async def initialize(self):
        """Initialize audio processor"""
        try:
            # Initialize audio processing components
            if torch.cuda.is_available():
                logger.info("🚀 CUDA available for audio processing")

            logger.info("✅ Audio processor initialization complete")

        except Exception as e:
            logger.error(f"❌ Audio processor initialization failed: {e}")
            raise

    async def process_audio_chunk(self, audio_data: bytes, session_id: str) -> Optional[np.ndarray]:
        """Process incoming audio chunk"""
        start_time = time.time()

        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_numpy(audio_data)

            # Normalize audio
            audio_array = self._normalize_audio(audio_array)

            # Apply noise reduction if enabled
            if self.audio_config.enable_noise_reduction:
                audio_array = self._reduce_noise(audio_array)

            # Apply auto gain if enabled
            if self.audio_config.enable_auto_gain:
                audio_array = self._apply_auto_gain(audio_array)

            # VAD processing
            is_speech = self.vad_processor.process_frame(audio_array)

            # Buffer management
            self.audio_buffer.append({
                "audio": audio_array,
                "timestamp": time.time(),
                "is_speech": is_speech,
                "session_id": session_id
            })

            # Process buffered audio if speech detected
            if is_speech or self.vad_processor.is_speaking():
                processed_audio = await self._process_buffered_audio()

                # Track performance
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.chunks_processed += 1

                return processed_audio

            return None

        except Exception as e:
            logger.error(f"❌ Audio chunk processing failed: {e}")
            return None

    def _bytes_to_numpy(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Assume 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32) / 32767.0

            return audio_array

        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return np.array([], dtype=np.float32)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels"""
        try:
            # Remove DC offset
            audio = audio - np.mean(audio)

            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            return audio

        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return audio

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Simple noise reduction using spectral subtraction"""
        try:
            # Simple high-pass filter to remove low-frequency noise
            if len(audio) > 1:
                # First-order high-pass filter
                filtered = np.zeros_like(audio)
                filtered[0] = audio[0]
                for i in range(1, len(audio)):
                    filtered[i] = 0.95 * filtered[i-1] + 0.95 * (audio[i] - audio[i-1])
                return filtered

            return audio

        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return audio

    def _apply_auto_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply automatic gain control"""
        try:
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio ** 2))

            # Target RMS level
            target_rms = 0.1

            # Apply gain if signal is too quiet
            if rms > 0 and rms < target_rms:
                gain = min(target_rms / rms, 10.0)  # Limit max gain
                audio = audio * gain

            return audio

        except Exception as e:
            logger.error(f"Auto gain error: {e}")
            return audio

    async def _process_buffered_audio(self) -> Optional[np.ndarray]:
        """Process buffered audio chunks"""
        try:
            if not self.audio_buffer:
                return None

            # Combine recent audio chunks
            audio_chunks = []
            current_time = time.time()

            # Get audio from last second
            while self.audio_buffer:
                chunk = self.audio_buffer[0]
                if current_time - chunk["timestamp"] > 1.0:  # Remove old chunks
                    self.audio_buffer.popleft()
                else:
                    break

            # Collect speech chunks
            for chunk in list(self.audio_buffer):
                if chunk["is_speech"]:
                    audio_chunks.append(chunk["audio"])

            if not audio_chunks:
                return None

            # Concatenate audio chunks
            combined_audio = np.concatenate(audio_chunks)

            # Ensure minimum length for processing
            min_samples = int(0.1 * self.audio_config.sample_rate)  # 100ms
            if len(combined_audio) < min_samples:
                return None

            return combined_audio

        except Exception as e:
            logger.error(f"Buffer processing error: {e}")
            return None

    def create_overlapping_chunks(self, audio: np.ndarray, chunk_size_ms: int = 500, overlap_ms: int = 250) -> List[np.ndarray]:
        """Create overlapping audio chunks for processing"""
        try:
            chunk_samples = int(chunk_size_ms * self.audio_config.sample_rate / 1000)
            overlap_samples = int(overlap_ms * self.audio_config.sample_rate / 1000)
            hop_samples = chunk_samples - overlap_samples

            chunks = []
            start = 0

            while start + chunk_samples <= len(audio):
                chunk = audio[start:start + chunk_samples]
                chunks.append(chunk)
                start += hop_samples

            # Add final chunk if there's remaining audio
            if start < len(audio):
                final_chunk = audio[start:]
                # Pad to required size
                padding = chunk_samples - len(final_chunk)
                if padding > 0:
                    final_chunk = np.pad(final_chunk, (0, padding), mode='constant')
                chunks.append(final_chunk)

            return chunks

        except Exception as e:
            logger.error(f"Chunk creation error: {e}")
            return [audio]  # Return original audio as single chunk

    async def process_streaming_audio(self, audio_stream: AsyncGenerator[bytes, None], session_id: str) -> AsyncGenerator[np.ndarray, None]:
        """Process streaming audio data"""
        audio_buffer = np.array([], dtype=np.float32)

        async for audio_chunk in audio_stream:
            try:
                # Convert and append to buffer
                chunk_array = self._bytes_to_numpy(audio_chunk)
                audio_buffer = np.concatenate([audio_buffer, chunk_array])

                # Process when buffer is large enough
                min_buffer_size = int(0.5 * self.audio_config.sample_rate)  # 500ms

                while len(audio_buffer) >= min_buffer_size:
                    # Extract chunk for processing
                    process_chunk = audio_buffer[:min_buffer_size]
                    audio_buffer = audio_buffer[min_buffer_size//2:]  # 50% overlap

                    # Process chunk
                    processed = await self.process_audio_chunk(process_chunk.tobytes(), session_id)

                    if processed is not None:
                        yield processed

            except Exception as e:
                logger.error(f"Streaming audio processing error: {e}")
                continue

    def get_performance_stats(self) -> Dict:
        """Get audio processing performance statistics"""
        if not self.processing_times:
            return {"status": "no_data"}

        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times)
        min_time = np.min(self.processing_times)

        return {
            "average_processing_time": avg_time,
            "max_processing_time": max_time,
            "min_processing_time": min_time,
            "chunks_processed": self.chunks_processed,
            "buffer_size": len(self.audio_buffer),
            "vad_type": self.vad_processor.vad_type,
            "sample_rate": self.audio_config.sample_rate,
            "chunk_size_ms": self.audio_config.chunk_size_ms
        }

    async def cleanup(self):
        """Clean up audio processor resources"""
        logger.info("🧹 Cleaning up audio processor...")

        self.audio_buffer.clear()
        self.processed_chunks.clear()
        self.processing_times.clear()

        logger.info("✅ Audio processor cleanup completed")
