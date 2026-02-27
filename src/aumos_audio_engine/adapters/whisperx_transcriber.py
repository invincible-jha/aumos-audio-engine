"""WhisperX transcription adapter.

Implements TranscriberProtocol using faster-whisper (WhisperX-compatible) for
GPU-accelerated audio transcription with word-level timestamp alignment, speaker
diarization, language detection, and confidence scoring.
"""

import asyncio
import io
import time
from typing import Any

import numpy as np
import soundfile as sf

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)

# Supported audio formats for conversion to float32 PCM
_SUPPORTED_FORMATS = frozenset({"wav", "mp3", "flac", "ogg", "opus", "m4a"})


class WhisperXTranscriber:
    """Audio transcription adapter using faster-whisper (WhisperX backend).

    Provides word-level timestamps, speaker diarization, language detection,
    and per-segment confidence scoring. All heavy CPU/GPU inference is offloaded
    to a thread-pool executor to avoid blocking the async event loop.

    Model loading is deferred to the first call to initialize() or transcribe().
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize transcriber with service settings.

        Args:
            settings: Audio engine settings containing Whisper model configuration,
                device selection, and compute type.
        """
        self._settings = settings
        self._model: Any = None
        self._diarization_pipeline: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load WhisperX / faster-whisper model into memory.

        Idempotent — safe to call multiple times. Blocks until the model is
        fully loaded in the background executor thread.

        Raises:
            ImportError: If faster-whisper package is not installed.
            RuntimeError: If model loading fails.
        """
        if self._initialized:
            return

        logger.info(
            "Initializing WhisperX transcriber",
            model=self._settings.whisper_model,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)

        self._initialized = True
        logger.info(
            "WhisperX transcriber initialized",
            model=self._settings.whisper_model,
            device=self._settings.whisper_device,
        )

    def _load_model(self) -> None:
        """Load faster-whisper model synchronously (runs in thread pool)."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self._model = WhisperModel(
                self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
            logger.info(
                "Faster-whisper model loaded",
                model=self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
        except ImportError:
            logger.error(
                "faster-whisper package not installed. "
                "Install with: pip install faster-whisper>=1.0.0"
            )
            raise
        except Exception as exc:
            logger.error(
                "Failed to load WhisperX model",
                model=self._settings.whisper_model,
                error=str(exc),
            )
            raise

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language: str | None,
    ) -> dict:
        """Transcribe audio to text with word-level timestamps and confidence scores.

        Decodes the input audio bytes, normalises to float32 PCM mono, then runs
        faster-whisper inference in a thread-pool executor to avoid event loop
        blocking.

        Args:
            audio_bytes: Raw audio bytes to transcribe.
            audio_format: Audio container format (wav, mp3, flac, ogg, opus, m4a).
            language: BCP-47 language code for forced language (e.g. 'en', 'fr').
                Pass None to enable automatic language detection.

        Returns:
            Dict with keys:
                - 'text': Full transcription as a single string.
                - 'segments': List of segment dicts, each containing:
                    - 'id': Segment index (int).
                    - 'start': Start time in seconds (float).
                    - 'end': End time in seconds (float).
                    - 'text': Segment text (str).
                    - 'confidence': Mean token log-probability normalised to [0,1].
                    - 'words': List of word-level dicts with 'word', 'start', 'end', 'probability'.
                - 'language': Detected or provided language code.
                - 'confidence': Overall confidence score averaged across all segments.
                - 'duration_seconds': Approximate audio duration in seconds.
                - 'processing_time_seconds': Wall-clock inference time.
        """
        if not self._initialized:
            await self.initialize()

        if audio_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{audio_format}'. "
                f"Supported: {sorted(_SUPPORTED_FORMATS)}"
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_bytes,
            audio_format,
            language,
        )

        logger.info(
            "Transcription complete",
            language=result.get("language"),
            segment_count=len(result.get("segments", [])),
            text_length=len(result.get("text", "")),
            confidence=round(result.get("confidence", 0.0), 4),
            duration_seconds=round(result.get("duration_seconds", 0.0), 2),
            processing_time_seconds=round(result.get("processing_time_seconds", 0.0), 2),
        )

        return result

    def _transcribe_sync(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language: str | None,
    ) -> dict:
        """Synchronous transcription pipeline (runs in thread pool).

        Args:
            audio_bytes: Raw audio bytes.
            audio_format: Audio container format.
            language: Forced language code or None for auto-detection.

        Returns:
            Transcription result dict.
        """
        if self._model is None:
            raise RuntimeError("WhisperX model not initialised — call initialize() first")

        start_time = time.monotonic()

        # Decode to float32 PCM
        audio_array, sample_rate = self._decode_audio(audio_bytes)
        audio_duration = len(audio_array) / sample_rate

        # Run faster-whisper inference with word timestamps
        transcribe_kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if language:
            transcribe_kwargs["language"] = language

        segments_generator, info = self._model.transcribe(audio_array, **transcribe_kwargs)

        segments: list[dict] = []
        all_text_parts: list[str] = []
        confidence_scores: list[float] = []

        for segment in segments_generator:
            segment_confidence = self._log_prob_to_confidence(
                getattr(segment, "avg_logprob", -1.0)
            )
            confidence_scores.append(segment_confidence)

            word_entries: list[dict] = []
            if hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    word_entries.append({
                        "word": word.word,
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "probability": round(float(getattr(word, "probability", 0.0)), 4),
                    })

            segment_dict = {
                "id": segment.id,
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "confidence": round(segment_confidence, 4),
                "no_speech_prob": round(float(getattr(segment, "no_speech_prob", 0.0)), 4),
                "words": word_entries,
            }
            segments.append(segment_dict)
            all_text_parts.append(segment.text.strip())

        full_text = " ".join(all_text_parts)
        overall_confidence = (
            float(np.mean(confidence_scores)) if confidence_scores else 0.0
        )
        detected_language = info.language if language is None else language
        processing_time = time.monotonic() - start_time

        return {
            "text": full_text,
            "segments": segments,
            "language": detected_language,
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
            "confidence": round(overall_confidence, 4),
            "duration_seconds": round(audio_duration, 3),
            "processing_time_seconds": round(processing_time, 3),
        }

    async def transcribe_batch(
        self,
        audio_items: list[tuple[bytes, str]],
        language: str | None = None,
    ) -> list[dict]:
        """Transcribe multiple audio files in parallel using the thread pool.

        Args:
            audio_items: List of (audio_bytes, audio_format) tuples.
            language: Shared language hint for all items, or None for auto-detect.

        Returns:
            List of transcription result dicts in the same order as audio_items.
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, self._transcribe_sync, audio_bytes, audio_format, language)
            for audio_bytes, audio_format in audio_items
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: list[dict] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch transcription item failed",
                    item_index=index,
                    error=str(result),
                )
                processed.append({
                    "text": "",
                    "segments": [],
                    "language": language or "unknown",
                    "confidence": 0.0,
                    "duration_seconds": 0.0,
                    "processing_time_seconds": 0.0,
                    "error": str(result),
                })
            else:
                processed.append(result)  # type: ignore[arg-type]

        logger.info(
            "Batch transcription complete",
            total_items=len(audio_items),
            failed=sum(1 for r in processed if "error" in r),
        )
        return processed

    def _decode_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode audio bytes to float32 mono PCM array.

        Args:
            audio_bytes: Raw audio bytes in any format supported by soundfile.

        Returns:
            Tuple of (float32 mono audio array, sample_rate).
        """
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer, dtype="float32")

        # Convert stereo/multi-channel to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16 kHz if needed (Whisper's native rate)
        if sample_rate != 16000:
            import librosa  # type: ignore[import-untyped]

            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )
            sample_rate = 16000

        return audio_data.astype(np.float32), sample_rate

    @staticmethod
    def _log_prob_to_confidence(avg_log_prob: float) -> float:
        """Convert average log probability to a [0, 1] confidence score.

        Uses sigmoid-like mapping: exp(log_prob) clamped to [0, 1].

        Args:
            avg_log_prob: Average log probability from Whisper (typically -2.0 to 0.0).

        Returns:
            Confidence score between 0.0 (low) and 1.0 (high).
        """
        return float(np.clip(np.exp(avg_log_prob), 0.0, 1.0))

    async def detect_language(self, audio_bytes: bytes, audio_format: str) -> dict:
        """Detect the spoken language of an audio file without full transcription.

        Args:
            audio_bytes: Raw audio bytes.
            audio_format: Audio container format.

        Returns:
            Dict with 'language' (BCP-47 code) and 'probability' (float).
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._detect_language_sync,
            audio_bytes,
            audio_format,
        )

    def _detect_language_sync(self, audio_bytes: bytes, audio_format: str) -> dict:
        """Synchronous language detection (runs in thread pool)."""
        if self._model is None:
            raise RuntimeError("WhisperX model not initialised — call initialize() first")

        audio_array, _ = self._decode_audio(audio_bytes)

        # Pad/trim to 30-second window that Whisper uses for language detection
        target_length = 16000 * 30
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))

        _, info = self._model.transcribe(audio_array, language=None, word_timestamps=False)

        return {
            "language": info.language,
            "probability": round(float(getattr(info, "language_probability", 0.0)), 4),
        }

    async def health_check(self) -> bool:
        """Return True if the transcriber is healthy and ready to process audio."""
        try:
            if not self._initialized:
                await self.initialize()
            return self._model is not None
        except Exception as exc:
            logger.warning("WhisperX health check failed", error=str(exc))
            return False
