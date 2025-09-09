"""
Optimized Voxtral 3B inference engine for real-time streaming
Includes torch.compile optimization, memory management, and batching
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    batch_size: int = 4
    max_length: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    sample_rate: int = 16000
    chunk_length_s: float = 30.0
    return_timestamps: bool = False
    enable_caching: bool = True
    cache_size: int = 1000

class VoxtralInferenceEngine:
    """Production-ready Voxtral 3B inference engine with streaming optimization"""

    def __init__(self, config):
        self.config = config
        self.inference_config = InferenceConfig()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = torch.device(self.inference_config.device)

        # Threading for async inference
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Response cache
        self.response_cache = {}
        self.cache_access_times = {}

        # Model warm-up status
        self.is_warmed_up = False

        # Performance metrics
        self.inference_times = []
        self.batch_processing_enabled = True

        logger.info(f"🧠 Initializing Voxtral inference engine on {self.device}")

    async def initialize(self):
        """Initialize the model and optimize for inference"""
        try:
            # Load processor and tokenizer
            logger.info("📦 Loading Voxtral processor and tokenizer...")
            self.processor = AutoProcessor.from_pretrained(
                self.inference_config.model_id,
                torch_dtype=self.inference_config.torch_dtype
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.inference_config.model_id
            )

            # Load model with optimizations
            logger.info("🚀 Loading Voxtral 3B model...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.inference_config.model_id,
                torch_dtype=self.inference_config.torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                use_cache=self.inference_config.enable_caching,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )

            # Move to device and optimize
            self.model = self.model.to(self.device)
            self.model.eval()

            # Apply torch.compile optimization
            if self.inference_config.use_compile and hasattr(torch, 'compile'):
                logger.info(f"⚡ Applying torch.compile with mode: {self.inference_config.compile_mode}")
                self.model = torch.compile(
                    self.model,
                    mode=self.inference_config.compile_mode,
                    fullgraph=True,
                    dynamic=False
                )

            # Warm up the model
            await self._warm_up_model()

            logger.info("✅ Voxtral inference engine initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Voxtral engine: {e}")
            raise

    async def _warm_up_model(self):
        """Warm up the model with dummy inputs to optimize compilation"""
        logger.info("🔥 Warming up model...")

        try:
            # Create dummy audio input (1 second of silence)
            dummy_audio = np.zeros((self.inference_config.sample_rate,), dtype=np.float32)

            # Run inference multiple times to warm up
            for i in range(3):
                logger.info(f"  Warm-up pass {i+1}/3")
                start_time = time.time()

                # Process dummy audio
                inputs = self.processor(
                    dummy_audio,
                    sampling_rate=self.inference_config.sample_rate,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                # Run inference
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=self.inference_config.max_length,
                        temperature=self.inference_config.temperature,
                        top_p=self.inference_config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                warm_up_time = time.time() - start_time
                logger.info(f"    Warm-up time: {warm_up_time:.3f}s")

            self.is_warmed_up = True
            logger.info("✅ Model warm-up completed")

        except Exception as e:
            logger.error(f"❌ Model warm-up failed: {e}")
            raise

    async def process_audio_streaming(self, audio_chunk: np.ndarray, session_id: str) -> str:
        """Process audio chunk for real-time streaming"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(audio_chunk, session_id)
            if cache_key in self.response_cache:
                self.cache_access_times[cache_key] = time.time()
                return self.response_cache[cache_key]

            # Preprocess audio
            inputs = self.processor(
                audio_chunk,
                sampling_rate=self.inference_config.sample_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.inference_config.chunk_length_s * self.inference_config.sample_rate)
            ).to(self.device)

            # Run inference asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                inputs
            )

            # Cache response
            if self.inference_config.enable_caching:
                self._cache_response(cache_key, response)

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Keep only last 100 times for rolling average
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            logger.debug(f"🎯 Audio inference completed in {inference_time:.3f}s")
            return response

        except Exception as e:
            logger.error(f"❌ Audio inference failed: {e}")
            return f"Error processing audio: {str(e)}"

    def _run_inference(self, inputs: Dict[str, torch.Tensor]) -> str:
        """Run model inference synchronously"""
        try:
            with torch.no_grad():
                # Generate response
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.inference_config.max_length,
                    temperature=self.inference_config.temperature,
                    top_p=self.inference_config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_timestamps=self.inference_config.return_timestamps
                )

                # Decode response
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]

                return response.strip()

        except Exception as e:
            logger.error(f"❌ Model inference failed: {e}")
            raise

    async def process_text(self, text: str, session_id: str) -> str:
        """Process text input for conversation"""
        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.inference_config.max_length
            ).to(self.device)

            # Run inference asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._run_text_inference,
                inputs
            )

            inference_time = time.time() - start_time
            logger.debug(f"💬 Text inference completed in {inference_time:.3f}s")

            return response

        except Exception as e:
            logger.error(f"❌ Text inference failed: {e}")
            return f"Error processing text: {str(e)}"

    def _run_text_inference(self, inputs: Dict[str, torch.Tensor]) -> str:
        """Run text inference synchronously"""
        try:
            with torch.no_grad():
                # Generate response
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.inference_config.max_length,
                    temperature=self.inference_config.temperature,
                    top_p=self.inference_config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode response
                response = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )

                return response.strip()

        except Exception as e:
            logger.error(f"❌ Text inference failed: {e}")
            raise

    def _get_cache_key(self, audio_chunk: np.ndarray, session_id: str) -> str:
        """Generate cache key for audio chunk"""
        audio_hash = hash(audio_chunk.tobytes())
        return f"{session_id}_{audio_hash}_{len(audio_chunk)}"

    def _cache_response(self, cache_key: str, response: str):
        """Cache response with size management"""
        # Remove old entries if cache is full
        if len(self.response_cache) >= self.inference_config.cache_size:
            # Remove least recently accessed
            oldest_key = min(
                self.cache_access_times.keys(),
                key=lambda k: self.cache_access_times[k]
            )
            del self.response_cache[oldest_key]
            del self.cache_access_times[oldest_key]

        self.response_cache[cache_key] = response
        self.cache_access_times[cache_key] = time.time()

    async def process_batch(self, audio_chunks: List[np.ndarray], session_ids: List[str]) -> List[str]:
        """Process multiple audio chunks in batch for better throughput"""
        if not self.batch_processing_enabled or len(audio_chunks) == 1:
            # Process individually
            results = []
            for chunk, session_id in zip(audio_chunks, session_ids):
                result = await self.process_audio_streaming(chunk, session_id)
                results.append(result)
            return results

        start_time = time.time()

        try:
            # Prepare batch inputs
            batch_inputs = []
            for chunk in audio_chunks:
                inputs = self.processor(
                    chunk,
                    sampling_rate=self.inference_config.sample_rate,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                batch_inputs.append(inputs)

            # Pad and batch
            input_features = torch.cat([inp["input_features"] for inp in batch_inputs], dim=0)

            # Run batch inference
            loop = asyncio.get_event_loop()
            responses = await loop.run_in_executor(
                self.executor,
                self._run_batch_inference,
                {"input_features": input_features}
            )

            batch_time = time.time() - start_time
            logger.debug(f"🔄 Batch inference ({len(audio_chunks)} items) completed in {batch_time:.3f}s")

            return responses

        except Exception as e:
            logger.error(f"❌ Batch inference failed: {e}")
            # Fallback to individual processing
            results = []
            for chunk, session_id in zip(audio_chunks, session_ids):
                result = await self.process_audio_streaming(chunk, session_id)
                results.append(result)
            return results

    def _run_batch_inference(self, batch_inputs: Dict[str, torch.Tensor]) -> List[str]:
        """Run batch inference synchronously"""
        try:
            with torch.no_grad():
                # Generate responses for batch
                generated_ids = self.model.generate(
                    **batch_inputs,
                    max_length=self.inference_config.max_length,
                    temperature=self.inference_config.temperature,
                    top_p=self.inference_config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode all responses
                responses = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                return [resp.strip() for resp in responses]

        except Exception as e:
            logger.error(f"❌ Batch inference failed: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {"status": "no_data"}

        avg_time = np.mean(self.inference_times)
        min_time = np.min(self.inference_times)
        max_time = np.max(self.inference_times)
        p95_time = np.percentile(self.inference_times, 95)

        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            }

        return {
            "average_inference_time": avg_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "p95_inference_time": p95_time,
            "total_inferences": len(self.inference_times),
            "cache_size": len(self.response_cache),
            "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            "is_warmed_up": self.is_warmed_up,
            **gpu_stats
        }

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)"""
        # This is a simplified implementation
        # In production, you'd track hits vs misses
        return len(self.response_cache) / max(len(self.inference_times), 1)

    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up Voxtral inference engine...")

        if self.executor:
            self.executor.shutdown(wait=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.response_cache.clear()
        self.cache_access_times.clear()

        logger.info("✅ Voxtral inference engine cleanup completed")
