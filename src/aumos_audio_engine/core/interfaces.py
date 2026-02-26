"""Abstract interfaces (Protocol classes) for aumos-audio-engine adapters.

Defines contracts for all external integrations. Services depend on these
protocols, not on concrete implementations (dependency inversion principle).
"""

from typing import Protocol, runtime_checkable

from aumos_audio_engine.api.schemas import (
    AudioDeidentifyRequest,
    AudioSynthesizeRequest,
    AudioTranscribeRequest,
    BatchAudioRequest,
    MNPIScanRequest,
    StyleTransferRequest,
)


@runtime_checkable
class TTSEngineProtocol(Protocol):
    """Protocol for text-to-speech synthesis engines."""

    async def initialize(self) -> None:
        """Initialize the TTS engine (load models, warm up CUDA, etc.).

        Must be called before synthesize(). Idempotent.
        """
        ...

    async def synthesize(
        self,
        text: str,
        voice_style_config: dict,
        output_format: str,
        sample_rate: int,
    ) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            voice_style_config: Voice style parameters from VoiceProfile.style_config.
            output_format: Target audio format (wav, mp3, flac, ogg).
            sample_rate: Output sample rate in Hz.

        Returns:
            Raw audio bytes in the specified format.
        """
        ...

    async def get_available_voices(self) -> list[dict]:
        """Return list of available pre-built voice configurations.

        Returns:
            List of dicts with 'id', 'name', and 'style_config' keys.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the TTS engine is healthy and ready."""
        ...


@runtime_checkable
class SpeakerDeidentifierProtocol(Protocol):
    """Protocol for speaker de-identification processors."""

    async def deidentify(
        self,
        audio_bytes: bytes,
        input_format: str,
        output_format: str,
        target_similarity_threshold: float,
    ) -> tuple[bytes, dict]:
        """Remove speaker biometric identity from audio.

        Applies pitch shifting, formant modification, and temporal perturbation
        to eliminate biometric identifiability while preserving semantic content.

        Args:
            audio_bytes: Input audio bytes containing speaker voice.
            input_format: Format of the input audio (wav, mp3, etc.).
            output_format: Desired output format.
            target_similarity_threshold: Maximum allowable cosine similarity to
                original speaker embedding (default 0.85). Job fails if exceeded.

        Returns:
            Tuple of (processed_audio_bytes, metadata_dict). Metadata includes
            achieved similarity score and transformation parameters applied.
        """
        ...

    async def measure_speaker_similarity(
        self,
        audio_bytes_a: bytes,
        audio_bytes_b: bytes,
        format_a: str,
        format_b: str,
    ) -> float:
        """Measure cosine similarity between two audio samples' speaker embeddings.

        Args:
            audio_bytes_a: First audio sample.
            audio_bytes_b: Second audio sample.
            format_a: Format of first sample.
            format_b: Format of second sample.

        Returns:
            Cosine similarity score between 0.0 (different) and 1.0 (identical).
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the de-identifier is operational."""
        ...


@runtime_checkable
class TranscriberProtocol(Protocol):
    """Protocol for audio transcription engines."""

    async def initialize(self) -> None:
        """Initialize transcription model. Idempotent."""
        ...

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language: str | None,
    ) -> dict:
        """Transcribe audio to text.

        Args:
            audio_bytes: Input audio to transcribe.
            audio_format: Audio format (wav, mp3, flac, etc.).
            language: BCP-47 language code (e.g., 'en', 'fr'). None for auto-detect.

        Returns:
            Dict with keys:
                - 'text': Full transcription string
                - 'segments': List of timed segment dicts
                - 'language': Detected or provided language code
                - 'confidence': Average confidence score
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the transcriber is healthy and ready."""
        ...


@runtime_checkable
class MNPIDetectorProtocol(Protocol):
    """Protocol for MNPI (Material Non-Public Information) detection."""

    async def scan(
        self,
        text: str,
        tenant_id: str,
        context_metadata: dict | None,
    ) -> dict:
        """Scan text for MNPI content.

        Combines keyword matching with contextual analysis to detect material
        non-public information in financial communications.

        Args:
            text: Text content to scan (typically audio transcript).
            tenant_id: Tenant context for jurisdiction-specific ruleset selection.
            context_metadata: Optional metadata (speaker roles, meeting type, etc.)

        Returns:
            Dict with keys:
                - 'mnpi_detected': bool
                - 'confidence': float (0.0–1.0)
                - 'matched_keywords': List of matched MNPI keyword strings
                - 'flagged_segments': List of dicts with 'text', 'reason', 'confidence'
                - 'risk_level': 'none' | 'low' | 'medium' | 'high' | 'critical'
        """
        ...

    async def load_keywords(self, keywords_path: str) -> int:
        """Load MNPI keyword list from JSON file.

        Args:
            keywords_path: Path to JSON file containing keyword patterns.

        Returns:
            Number of keywords loaded.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the detector is operational."""
        ...


@runtime_checkable
class StyleTransferProtocol(Protocol):
    """Protocol for voice style transfer without biometric identity retention."""

    async def transfer_style(
        self,
        source_audio_bytes: bytes,
        source_format: str,
        target_style_config: dict,
        output_format: str,
        preserve_semantics: bool,
    ) -> tuple[bytes, dict]:
        """Transfer voice style while removing biometric identity.

        Extracts prosodic and stylistic features from source audio and applies
        them to synthesized output. The source speaker's biometric identity
        is NOT retained in the output.

        Args:
            source_audio_bytes: Source audio from which to extract style.
            source_format: Format of source audio.
            target_style_config: Target style overrides (can be empty for pure extraction).
            output_format: Output audio format.
            preserve_semantics: If True, verify transcript matches after transfer.

        Returns:
            Tuple of (output_audio_bytes, style_metadata). Style metadata includes
            the extracted style parameters (safe to log/store — no biometrics).
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the style transfer engine is operational."""
        ...


@runtime_checkable
class AudioStorageProtocol(Protocol):
    """Protocol for audio file storage (MinIO/S3)."""

    async def upload(
        self,
        tenant_id: str,
        job_id: str,
        audio_bytes: bytes,
        audio_format: str,
    ) -> str:
        """Upload audio bytes to object storage.

        Args:
            tenant_id: Tenant ID for namespacing the storage path.
            job_id: Job ID for unique file naming.
            audio_bytes: Audio data to store.
            audio_format: File extension for the stored object.

        Returns:
            Storage URI (e.g., 's3://bucket/tenant/job/output.wav').
        """
        ...

    async def download(self, storage_uri: str) -> bytes:
        """Download audio bytes from object storage.

        Args:
            storage_uri: URI returned by upload().

        Returns:
            Raw audio bytes.
        """
        ...

    async def delete(self, storage_uri: str) -> None:
        """Delete an audio object from storage."""
        ...

    async def health_check(self) -> bool:
        """Return True if storage backend is reachable."""
        ...


@runtime_checkable
class PrivacyClientProtocol(Protocol):
    """Protocol for privacy engine validation calls."""

    async def validate_audio_job(
        self,
        job_id: str,
        tenant_id: str,
        job_type: str,
        deidentification_applied: bool,
        similarity_score: float | None,
    ) -> dict:
        """Validate an audio job against privacy compliance rules.

        Args:
            job_id: Job identifier.
            tenant_id: Tenant context.
            job_type: Type of processing ('synthesize', 'deidentify', etc.).
            deidentification_applied: Whether de-identification was performed.
            similarity_score: Achieved speaker similarity score (None if not applicable).

        Returns:
            Dict with keys:
                - 'approved': bool
                - 'reason': str (explanation if not approved)
                - 'risk_flags': List of compliance risk flag strings
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the privacy engine is reachable."""
        ...
