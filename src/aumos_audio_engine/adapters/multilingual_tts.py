"""Multi-language TTS adapter — GAP-84 competitive gap implementation.

Extends the core TTS capability with multi-language synthesis using
XTTS-v2 (Coqui) for 17 languages and MMS-TTS (Meta) as fallback for
100+ languages. Supports zero-shot voice cloning across languages.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Supported language codes with display names
XTTS_SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "hu": "Hungarian",
}

MMS_SUPPORTED_LANGUAGE_COUNT = 1107  # Meta MMS covers 1107 languages


class MultilingualTTSAdapter:
    """Multi-language text-to-speech adapter using XTTS-v2 and MMS-TTS.

    Provides high-quality synthesis in 17 languages via XTTS-v2 with
    automatic fallback to Meta MMS-TTS for less common languages.
    Supports zero-shot voice cloning: synthesize in any language using
    a reference audio clip from a different language.

    Args:
        device: Torch device for inference ("cuda", "cpu", "mps").
        cache_dir: Directory for downloaded model weights.
        xtts_model_id: HuggingFace model ID for XTTS-v2.
        enable_mms_fallback: Whether to use MMS-TTS for unsupported languages.
        default_language: Default language code when not specified.
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: str = "/tmp/model-cache",
        xtts_model_id: str = "coqui/XTTS-v2",
        enable_mms_fallback: bool = True,
        default_language: str = "en",
    ) -> None:
        self._device = device
        self._cache_dir = cache_dir
        self._xtts_model_id = xtts_model_id
        self._enable_mms_fallback = enable_mms_fallback
        self._default_language = default_language
        self._xtts_model: Any = None
        self._mms_models: dict[str, Any] = {}
        self._log = logger.bind(adapter="multilingual_tts")

    async def initialize(self) -> None:
        """Load XTTS-v2 model weights into memory.

        Downloads weights on first call (cached for subsequent calls).
        Runs in thread pool to avoid blocking the event loop.
        """
        self._log.info("multilingual_tts.initialize.start", device=self._device)
        await asyncio.to_thread(self._load_xtts_sync)
        self._log.info("multilingual_tts.initialize.complete")

    def _load_xtts_sync(self) -> None:
        """Synchronous XTTS-v2 model loading — called from thread pool."""
        try:
            from TTS.api import TTS  # type: ignore[import]

            self._xtts_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            )
            if self._device != "cpu":
                self._xtts_model = self._xtts_model.to(self._device)
            self._log.info("multilingual_tts.xtts_loaded")
        except Exception as exc:
            self._log.warning("multilingual_tts.xtts_load_failed", error=str(exc))
            self._xtts_model = None

    async def synthesize(
        self,
        text: str,
        language: str,
        output_format: str = "wav",
        sample_rate: int = 22050,
        reference_audio_bytes: bytes | None = None,
        speaker_name: str | None = None,
    ) -> bytes:
        """Synthesize speech from text in the specified language.

        Uses XTTS-v2 for supported languages. Falls back to MMS-TTS for
        unsupported languages if enable_mms_fallback is True.

        Args:
            text: Input text to synthesize.
            language: BCP-47 language code (e.g., "en", "fr", "zh-cn").
            output_format: Output audio format ("wav", "mp3", "flac").
            sample_rate: Output sample rate in Hz.
            reference_audio_bytes: Optional reference audio for zero-shot voice cloning.
                When provided, the synthesized voice will match the reference speaker.
            speaker_name: Named speaker preset (alternative to reference audio).

        Returns:
            Raw audio bytes in the specified format.

        Raises:
            ValueError: If the language is not supported and MMS fallback is disabled.
        """
        effective_language = language.lower() if language else self._default_language
        self._log.info(
            "multilingual_tts.synthesize",
            language=effective_language,
            text_length=len(text),
            has_reference=reference_audio_bytes is not None,
        )

        if effective_language in XTTS_SUPPORTED_LANGUAGES and self._xtts_model is not None:
            return await asyncio.to_thread(
                self._synthesize_xtts_sync,
                text,
                effective_language,
                output_format,
                sample_rate,
                reference_audio_bytes,
                speaker_name,
            )
        elif self._enable_mms_fallback:
            return await self._synthesize_mms(text, effective_language, output_format, sample_rate)
        else:
            supported = ", ".join(sorted(XTTS_SUPPORTED_LANGUAGES.keys()))
            raise ValueError(
                f"Language '{effective_language}' not supported. "
                f"Supported languages: {supported}"
            )

    def _synthesize_xtts_sync(
        self,
        text: str,
        language: str,
        output_format: str,
        sample_rate: int,
        reference_audio_bytes: bytes | None,
        speaker_name: str | None,
    ) -> bytes:
        """Synchronous XTTS-v2 synthesis — called from thread pool."""
        import io
        import tempfile

        import soundfile as sf  # type: ignore[import]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name

        kwargs: dict[str, Any] = {
            "text": text,
            "language": language,
            "file_path": output_path,
        }

        if reference_audio_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
                ref_file.write(reference_audio_bytes)
                kwargs["speaker_wav"] = ref_file.name
        elif speaker_name is not None:
            kwargs["speaker"] = speaker_name

        self._xtts_model.tts_to_file(**kwargs)

        # Read the output file and convert format if needed
        audio_data, sr = sf.read(output_path)
        buf = io.BytesIO()
        sf.write(buf, audio_data, samplerate=sample_rate, format=output_format.upper())
        return buf.getvalue()

    async def _synthesize_mms(
        self,
        text: str,
        language: str,
        output_format: str,
        sample_rate: int,
    ) -> bytes:
        """Synthesize via Meta MMS-TTS for languages not supported by XTTS-v2.

        MMS-TTS covers 1107 languages using Facebook's Massively Multilingual
        Speech model family.

        Args:
            text: Input text.
            language: BCP-47 language code.
            output_format: Output audio format.
            sample_rate: Output sample rate in Hz.

        Returns:
            Raw audio bytes.
        """
        self._log.info("multilingual_tts.mms_fallback", language=language)

        return await asyncio.to_thread(
            self._synthesize_mms_sync,
            text,
            language,
            output_format,
            sample_rate,
        )

    def _synthesize_mms_sync(
        self,
        text: str,
        language: str,
        output_format: str,
        sample_rate: int,
    ) -> bytes:
        """Synchronous MMS-TTS synthesis — called from thread pool."""
        import io

        import soundfile as sf  # type: ignore[import]
        from transformers import VitsModel, AutoTokenizer  # type: ignore[import]
        import torch

        model_id = f"facebook/mms-tts-{language}"
        if model_id not in self._mms_models:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self._cache_dir)
            model = VitsModel.from_pretrained(model_id, cache_dir=self._cache_dir)
            if self._device != "cpu":
                model = model.to(self._device)
            self._mms_models[model_id] = (tokenizer, model)

        tokenizer, model = self._mms_models[model_id]
        inputs = tokenizer(text, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.squeeze().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, waveform, samplerate=model.config.sampling_rate, format=output_format.upper())
        return buf.getvalue()

    def get_supported_languages(self) -> dict[str, Any]:
        """Return information about supported languages.

        Returns:
            Dict with xtts_languages (high-quality, 17 languages) and
            mms_language_count (MMS fallback coverage).
        """
        return {
            "xtts_languages": XTTS_SUPPORTED_LANGUAGES,
            "mms_fallback_enabled": self._enable_mms_fallback,
            "mms_language_count": MMS_SUPPORTED_LANGUAGE_COUNT if self._enable_mms_fallback else 0,
            "total_supported": (
                len(XTTS_SUPPORTED_LANGUAGES) + (MMS_SUPPORTED_LANGUAGE_COUNT if self._enable_mms_fallback else 0)
            ),
        }

    @property
    def is_ready(self) -> bool:
        """True if XTTS-v2 model is loaded and ready."""
        return self._xtts_model is not None
