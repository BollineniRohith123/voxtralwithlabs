"""
Configuration management for Voxtral 3B streaming service
Handles environment variables, settings validation, and defaults
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    device: str = "cuda"
    torch_dtype: str = "float16"
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    max_batch_size: int = 4
    max_length: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    enable_caching: bool = True
    cache_size: int = 1000
    warmup_iterations: int = 3

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size_ms: int = 160
    overlap_ms: int = 80
    buffer_size_ms: int = 1000
    vad_aggressiveness: int = 2
    vad_frame_duration_ms: int = 30
    energy_threshold: float = 0.01
    silence_timeout_ms: int = 1000
    enable_noise_reduction: bool = True
    enable_auto_gain: bool = True

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    access_log: bool = True
    timeout_keep_alive: int = 65
    timeout_graceful_shutdown: int = 30
    max_connections: int = 1000
    websocket_timeout: int = 300
    heartbeat_interval: int = 30

@dataclass
class RedisConfig:
    """Redis configuration for session management"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 8001
    enable_logging: bool = True
    log_dir: str = "/app/logs"
    log_rotation: str = "daily"
    log_retention_days: int = 7
    enable_health_checks: bool = True

class Config:
    """Main configuration class"""

    def __init__(self):
        # Load configuration from environment
        self.model = self._load_model_config()
        self.audio = self._load_audio_config()
        self.server = self._load_server_config()
        self.redis = self._load_redis_config()
        self.monitoring = self._load_monitoring_config()

        # Validate configuration
        self._validate_config()

        logger.info("✅ Configuration loaded successfully")

    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment"""
        return ModelConfig(
            model_id=os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-3B-2507"),
            device=os.getenv("DEVICE", "cuda" if self._is_cuda_available() else "cpu"),
            torch_dtype=os.getenv("TORCH_DTYPE", "float16"),
            use_compile=os.getenv("ENABLE_TORCH_COMPILE", "true").lower() == "true",
            compile_mode=os.getenv("TORCH_COMPILE_MODE", "reduce-overhead"),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "4")),
            max_length=int(os.getenv("MAX_LENGTH", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            top_p=float(os.getenv("TOP_P", "0.95")),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            cache_size=int(os.getenv("CACHE_SIZE", "1000")),
            warmup_iterations=int(os.getenv("WARMUP_ITERATIONS", "3"))
        )

    def _load_audio_config(self) -> AudioConfig:
        """Load audio configuration from environment"""
        return AudioConfig(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("AUDIO_CHANNELS", "1")),
            chunk_size_ms=int(os.getenv("AUDIO_CHUNK_SIZE_MS", "160")),
            overlap_ms=int(os.getenv("AUDIO_OVERLAP_MS", "80")),
            buffer_size_ms=int(os.getenv("AUDIO_BUFFER_SIZE_MS", "1000")),
            vad_aggressiveness=int(os.getenv("VAD_AGGRESSIVENESS", "2")),
            vad_frame_duration_ms=int(os.getenv("VAD_FRAME_DURATION_MS", "30")),
            energy_threshold=float(os.getenv("VAD_ENERGY_THRESHOLD", "0.01")),
            silence_timeout_ms=int(os.getenv("SILENCE_TIMEOUT_MS", "1000")),
            enable_noise_reduction=os.getenv("ENABLE_NOISE_REDUCTION", "true").lower() == "true",
            enable_auto_gain=os.getenv("ENABLE_AUTO_GAIN", "true").lower() == "true"
        )

    def _load_server_config(self) -> ServerConfig:
        """Load server configuration from environment"""
        return ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "1")),
            log_level=os.getenv("LOG_LEVEL", "info"),
            access_log=os.getenv("ACCESS_LOG", "true").lower() == "true",
            timeout_keep_alive=int(os.getenv("TIMEOUT_KEEP_ALIVE", "65")),
            timeout_graceful_shutdown=int(os.getenv("TIMEOUT_GRACEFUL_SHUTDOWN", "30")),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "1000")),
            websocket_timeout=int(os.getenv("WEBSOCKET_TIMEOUT", "300")),
            heartbeat_interval=int(os.getenv("HEARTBEAT_INTERVAL", "30"))
        )

    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment"""
        return RedisConfig(
            enabled=os.getenv("REDIS_ENABLED", "false").lower() == "true",
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        )

    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration from environment"""
        return MonitoringConfig(
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8001")),
            enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
            log_dir=os.getenv("LOG_DIR", "/app/logs"),
            log_rotation=os.getenv("LOG_ROTATION", "daily"),
            log_retention_days=int(os.getenv("LOG_RETENTION_DAYS", "7")),
            enable_health_checks=os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
        )

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Model validation
        if self.model.max_batch_size < 1:
            errors.append("max_batch_size must be >= 1")

        if not (0.0 <= self.model.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.model.top_p <= 1.0):
            errors.append("top_p must be between 0.0 and 1.0")

        # Audio validation
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            errors.append("sample_rate must be one of: 8000, 16000, 22050, 44100, 48000")

        if not (0 <= self.audio.vad_aggressiveness <= 3):
            errors.append("vad_aggressiveness must be between 0 and 3")

        if self.audio.vad_frame_duration_ms not in [10, 20, 30]:
            errors.append("vad_frame_duration_ms must be 10, 20, or 30")

        # Server validation
        if not (1 <= self.server.port <= 65535):
            errors.append("port must be between 1 and 65535")

        if self.server.workers < 1:
            errors.append("workers must be >= 1")

        # Redis validation
        if self.redis.enabled:
            if not (1 <= self.redis.port <= 65535):
                errors.append("redis_port must be between 1 and 65535")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary"""
        return {
            "model_id": self.model.model_id,
            "device": self.model.device,
            "torch_dtype": self.model.torch_dtype,
            "use_compile": self.model.use_compile,
            "compile_mode": self.model.compile_mode,
            "max_batch_size": self.model.max_batch_size,
            "max_length": self.model.max_length,
            "temperature": self.model.temperature,
            "top_p": self.model.top_p,
            "enable_caching": self.model.enable_caching,
            "cache_size": self.model.cache_size
        }

    def get_audio_config_dict(self) -> Dict[str, Any]:
        """Get audio configuration as dictionary"""
        return {
            "sample_rate": self.audio.sample_rate,
            "channels": self.audio.channels,
            "chunk_size_ms": self.audio.chunk_size_ms,
            "overlap_ms": self.audio.overlap_ms,
            "buffer_size_ms": self.audio.buffer_size_ms,
            "vad_aggressiveness": self.audio.vad_aggressiveness,
            "vad_frame_duration_ms": self.audio.vad_frame_duration_ms,
            "energy_threshold": self.audio.energy_threshold,
            "silence_timeout_ms": self.audio.silence_timeout_ms,
            "enable_noise_reduction": self.audio.enable_noise_reduction,
            "enable_auto_gain": self.audio.enable_auto_gain
        }

    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*50)
        print("🔧 VOXTRAL 3B STREAMING CONFIGURATION")
        print("="*50)

        print("\n📦 Model Configuration:")
        print(f"  Model ID: {self.model.model_id}")
        print(f"  Device: {self.model.device}")
        print(f"  Torch Compile: {self.model.use_compile}")
        print(f"  Batch Size: {self.model.max_batch_size}")
        print(f"  Temperature: {self.model.temperature}")
        print(f"  Top P: {self.model.top_p}")

        print("\n🎵 Audio Configuration:")
        print(f"  Sample Rate: {self.audio.sample_rate} Hz")
        print(f"  VAD Aggressiveness: {self.audio.vad_aggressiveness}")
        print(f"  Noise Reduction: {self.audio.enable_noise_reduction}")
        print(f"  Auto Gain: {self.audio.enable_auto_gain}")

        print("\n🌐 Server Configuration:")
        print(f"  Host: {self.server.host}")
        print(f"  Port: {self.server.port}")
        print(f"  Workers: {self.server.workers}")
        print(f"  Log Level: {self.server.log_level}")

        print("\n📊 Monitoring Configuration:")
        print(f"  Prometheus: {self.monitoring.enable_prometheus}")
        print(f"  Logging: {self.monitoring.enable_logging}")
        print(f"  Health Checks: {self.monitoring.enable_health_checks}")

        print("="*50 + "\n")
