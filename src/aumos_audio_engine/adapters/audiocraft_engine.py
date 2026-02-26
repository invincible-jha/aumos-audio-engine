"""AudioCraft environmental sound synthesis adapter.

Wraps Meta's AudioCraft library (audiocraft package) for generating
environmental and ambient audio without any biometric voice content.
Used for background sound synthesis, soundscape generation, and
non-voice audio creation.
"""

import asyncio
import io
from typing import Any

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)


class AudioCraftEngine:
    """Meta AudioCraft environmental audio synthesis adapter.

    Provides text-conditioned environmental sound generation using
    AudioCraft's MusicGen and AudioGen models. All generated audio
    is purely synthetic with no biometric content.
    """

    def __init__(self, settings: Settings, model_type: str = "audiogen") -> None:
        """Initialize AudioCraft engine.

        Args:
            settings: Audio engine settings.
            model_type: 'audiogen' for environmental sounds, 'musicgen' for music.
        """
        self._settings = settings
        self._model_type = model_type
        self._model: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load AudioCraft model into memory.

        Uses GPU if AUMOS_AUDIO_GPU_ENABLED is True and CUDA is available.
        """
        if self._initialized:
            return

        logger.info("Initializing AudioCraft engine", model_type=self._model_type, gpu=self._settings.gpu_enabled)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self._initialized = True
        logger.info("AudioCraft engine initialized", model_type=self._model_type)

    def _load_model(self) -> None:
        """Load AudioCraft model synchronously."""
        try:
            if self._model_type == "audiogen":
                from audiocraft.models import AudioGen  # type: ignore[import-untyped]

                device = "cuda" if self._settings.gpu_enabled else "cpu"
                self._model = AudioGen.get_pretrained("facebook/audiogen-medium", device=device)
            elif self._model_type == "musicgen":
                from audiocraft.models import MusicGen  # type: ignore[import-untyped]

                device = "cuda" if self._settings.gpu_enabled else "cpu"
                self._model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
            else:
                raise ValueError(f"Unknown AudioCraft model type: {self._model_type}")
        except ImportError:
            logger.error("AudioCraft not installed. Install with: pip install audiocraft>=1.2.0")
            raise

    async def generate(
        self,
        description: str,
        duration_seconds: float,
        output_format: str = "wav",
        sample_rate: int = 16000,
    ) -> bytes:
        """Generate environmental audio from text description.

        Args:
            description: Text description of the sound to generate.
            duration_seconds: Desired output duration in seconds.
            output_format: Output audio format.
            sample_rate: Output sample rate.

        Returns:
            Generated audio bytes in specified format.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Generating AudioCraft audio",
            description=description[:100],
            duration=duration_seconds,
            output_format=output_format,
        )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            description,
            duration_seconds,
            output_format,
            sample_rate,
        )

    def _generate_sync(
        self,
        description: str,
        duration_seconds: float,
        output_format: str,
        sample_rate: int,
    ) -> bytes:
        """Synchronous audio generation (runs in thread pool)."""
        import soundfile as sf
        import torch  # type: ignore[import-untyped]

        if self._model is None:
            raise RuntimeError("AudioCraft model not initialized")

        self._model.set_generation_params(duration=duration_seconds)
        wav_tensors = self._model.generate([description])

        # wav_tensors shape: (batch, channels, samples)
        wav_numpy = wav_tensors[0, 0].cpu().numpy()

        # Resample to desired sample rate if needed
        model_sample_rate = self._model.sample_rate
        if model_sample_rate != sample_rate:
            import librosa  # type: ignore[import-untyped]

            wav_numpy = librosa.resample(wav_numpy, orig_sr=model_sample_rate, target_sr=sample_rate)

        output_buffer = io.BytesIO()
        sf.write(output_buffer, wav_numpy, sample_rate, format=output_format.upper())
        return output_buffer.getvalue()

    async def health_check(self) -> bool:
        """Return True if AudioCraft engine is operational."""
        try:
            if not self._initialized:
                await self.initialize()
            return self._model is not None
        except Exception:
            return False
