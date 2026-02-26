"""Coqui TTS integration adapter.

Wraps the Coqui TTS library (TTS package) to implement TTSEngineProtocol.
Handles model loading, device selection, and audio format conversion.
"""

import asyncio
import io
import os
import tempfile
from functools import cached_property
from typing import Any

import soundfile as sf

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)


class CoquiTTSEngine:
    """Coqui TTS text-to-speech synthesis adapter.

    Wraps TTS library models for neural text-to-speech synthesis.
    Supports CPU and GPU inference. Audio is returned as bytes in the
    specified output format.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize with service settings.

        Args:
            settings: Audio engine settings with TTS model configuration.
        """
        self._settings = settings
        self._tts: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load Coqui TTS model into memory.

        Downloads model if not cached locally. Should be called once on
        service startup for GPU deployments to avoid cold-start latency.
        """
        if self._initialized:
            return

        logger.info(
            "Initializing Coqui TTS engine",
            model=self._settings.tts_model,
            gpu_enabled=self._settings.gpu_enabled,
        )

        # Run model loading in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self._initialized = True
        logger.info("Coqui TTS engine initialized successfully", model=self._settings.tts_model)

    def _load_model(self) -> None:
        """Load TTS model synchronously (runs in thread pool)."""
        try:
            from TTS.api import TTS  # type: ignore[import-untyped]

            use_gpu = self._settings.gpu_enabled and self._settings.tts_use_cuda

            self._tts = TTS(
                model_name=self._settings.tts_model,
                progress_bar=False,
                gpu=use_gpu,
            )
            logger.info(
                "TTS model loaded",
                model=self._settings.tts_model,
                gpu=use_gpu,
            )
        except ImportError:
            logger.error("Coqui TTS package not installed. Install with: pip install TTS>=0.22.0")
            raise
        except Exception as exc:
            logger.error("Failed to load TTS model", model=self._settings.tts_model, error=str(exc))
            raise

    async def synthesize(
        self,
        text: str,
        voice_style_config: dict,
        output_format: str,
        sample_rate: int,
    ) -> bytes:
        """Synthesize speech from text using Coqui TTS.

        Args:
            text: Input text to synthesize.
            voice_style_config: Voice parameters (speaker, speed, pitch modifiers).
            output_format: Target audio format.
            sample_rate: Output sample rate in Hz.

        Returns:
            Raw audio bytes in the specified format.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Starting TTS synthesis",
            text_length=len(text),
            output_format=output_format,
            sample_rate=sample_rate,
        )

        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            voice_style_config,
            output_format,
            sample_rate,
        )

        logger.info(
            "TTS synthesis complete",
            output_bytes=len(audio_bytes),
            output_format=output_format,
        )
        return audio_bytes

    def _synthesize_sync(
        self,
        text: str,
        voice_style_config: dict,
        output_format: str,
        sample_rate: int,
    ) -> bytes:
        """Synchronous TTS synthesis (runs in thread pool).

        Args:
            text: Text to synthesize.
            voice_style_config: Voice configuration dict.
            output_format: Output format.
            sample_rate: Sample rate.

        Returns:
            Audio bytes.
        """
        if self._tts is None:
            raise RuntimeError("TTS model not initialized — call initialize() first")

        # Use a temp file since Coqui TTS writes to file
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=self._settings.temp_dir, delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            speaker = voice_style_config.get("speaker")
            language = voice_style_config.get("language", "en")
            speed = float(voice_style_config.get("speed", 1.0))

            kwargs: dict[str, Any] = {"text": text, "file_path": tmp_path}
            if speaker:
                kwargs["speaker"] = speaker
            if language and hasattr(self._tts, "is_multi_lingual") and self._tts.is_multi_lingual:
                kwargs["language"] = language

            self._tts.tts_to_file(**kwargs)

            # Read and convert to desired format
            audio_data, source_sample_rate = sf.read(tmp_path)

            # Resample if needed
            if source_sample_rate != sample_rate:
                import librosa  # type: ignore[import-untyped]

                audio_data = librosa.resample(audio_data, orig_sr=source_sample_rate, target_sr=sample_rate)

            # Apply speed modification if needed
            if speed != 1.0:
                import librosa

                audio_data = librosa.effects.time_stretch(audio_data, rate=speed)

            # Encode to target format
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio_data, sample_rate, format=output_format.upper())
            return output_buffer.getvalue()

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def get_available_voices(self) -> list[dict]:
        """Return list of available pre-built voice configurations.

        Returns:
            List of voice config dicts with 'id', 'name', 'style_config' keys.
        """
        if not self._initialized:
            await self.initialize()

        if self._tts is None:
            return []

        voices = []
        if hasattr(self._tts, "speakers") and self._tts.speakers:
            for speaker_name in self._tts.speakers:
                voices.append({
                    "id": speaker_name,
                    "name": speaker_name,
                    "style_config": {"speaker": speaker_name},
                })
        else:
            # Single-speaker model — return default voice
            voices.append({
                "id": "default",
                "name": "Default Voice",
                "style_config": {},
            })

        return voices

    async def health_check(self) -> bool:
        """Return True if the TTS engine is healthy and ready."""
        try:
            if not self._initialized:
                await self.initialize()
            return self._tts is not None
        except Exception:
            return False
