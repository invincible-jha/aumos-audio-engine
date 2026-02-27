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


@runtime_checkable
class AudioQualityEvaluatorProtocol(Protocol):
    """Protocol for objective audio quality evaluation."""

    async def evaluate(
        self,
        synthesised_audio: bytes,
        reference_audio: bytes,
        synthesised_format: str,
        reference_format: str,
    ) -> dict:
        """Compute multi-dimensional quality metrics comparing synthesised to reference audio.

        Args:
            synthesised_audio: Synthesised audio bytes to evaluate.
            reference_audio: Reference audio bytes for comparison.
            synthesised_format: Container format of synthesised audio.
            reference_format: Container format of reference audio.

        Returns:
            Dict with keys:
                - 'mos_estimate': float [1.0–5.0] — MOS proxy score.
                - 'speaker_similarity': float [0.0–1.0].
                - 'pitch_contour_dtw': float [0.0–1.0] — higher = better match.
                - 'prosody_match': float [0.0–1.0].
                - 'snr_comparison': float [0.0–1.0].
                - 'fidelity_score': float [0.0–1.0] — weighted aggregate.
                - 'metric_details': dict — raw numeric values before normalisation.
        """
        ...

    async def evaluate_standalone(self, audio_bytes: bytes, audio_format: str) -> dict:
        """Compute standalone quality metrics without a reference audio sample.

        Args:
            audio_bytes: Audio bytes to evaluate.
            audio_format: Container format.

        Returns:
            Dict with SNR, spectral centroid, RMS energy, voiced ratio, and duration.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the quality evaluator is operational."""
        ...


@runtime_checkable
class AudioExportHandlerProtocol(Protocol):
    """Protocol for multi-format audio export and object storage operations."""

    async def export(
        self,
        audio_bytes: bytes,
        output_format: str,
        export_options: dict | None,
        metadata: dict | None,
    ) -> bytes:
        """Export audio bytes to the specified format.

        Args:
            audio_bytes: Source audio bytes (any soundfile-supported format).
            output_format: Target format: 'wav', 'mp3', 'ogg', 'flac'.
            export_options: Format-specific options (sample_rate, bitrate_kbps, etc.).
            metadata: Optional metadata for embedding (title, artist, copyright).

        Returns:
            Encoded audio bytes in the target format.
        """
        ...

    async def upload(
        self,
        tenant_id: str,
        job_id: str,
        audio_bytes: bytes,
        audio_format: str,
    ) -> str:
        """Upload audio bytes to MinIO/S3 object storage.

        Args:
            tenant_id: Tenant namespace for storage path isolation.
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
        """Delete an audio object from storage.

        Args:
            storage_uri: URI of the object to delete.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if storage backend is reachable."""
        ...


@runtime_checkable
class AudioBatchProcessorProtocol(Protocol):
    """Protocol for distributed audio batch processing job management."""

    async def start(self) -> None:
        """Start the worker pool. Idempotent."""
        ...

    async def stop(self, timeout_seconds: float) -> None:
        """Gracefully stop the worker pool.

        Args:
            timeout_seconds: Maximum time to wait before force-cancelling workers.
        """
        ...

    async def submit(
        self,
        payload: dict,
        handler_key: str,
        job_id: str | None,
        max_attempts: int | None,
        resource_hint: str,
    ) -> str:
        """Submit a single job for asynchronous processing.

        Args:
            payload: Job payload dict passed to the handler.
            handler_key: Registered handler key for this job type.
            job_id: Optional explicit job ID. Auto-generated if None.
            max_attempts: Retry limit override (None = use default).
            resource_hint: 'cpu' or 'gpu'.

        Returns:
            Job ID string for status polling.
        """
        ...

    async def submit_batch(
        self,
        items: list[dict],
        handler_key: str,
        max_attempts: int | None,
        resource_hint: str,
    ) -> list[str]:
        """Submit multiple jobs and return their job IDs.

        Args:
            items: List of payload dicts, one per job.
            handler_key: Handler key applied to all items.
            max_attempts: Retry limit override for all items.
            resource_hint: Resource hint for all items.

        Returns:
            List of job ID strings in the same order as items.
        """
        ...

    async def get_job_status(self, job_id: str) -> dict:
        """Get status and result for a single job.

        Args:
            job_id: Job ID to query.

        Returns:
            Dict with 'job_id', 'status', 'result', 'error', 'attempt', and timing fields.
        """
        ...

    async def wait_for_jobs(
        self,
        job_ids: list[str],
        poll_interval_seconds: float,
        timeout_seconds: float | None,
    ) -> dict:
        """Wait for a list of jobs to complete and return aggregated results.

        Args:
            job_ids: List of job IDs to await.
            poll_interval_seconds: Status polling interval.
            timeout_seconds: Maximum wait time. None = indefinite.

        Returns:
            BatchResult-compatible dict with total, completed, failed, results, errors.
        """
        ...

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancelled, False if not in a cancellable state.
        """
        ...

    def get_stats(self) -> dict:
        """Return aggregate runtime statistics.

        Returns:
            Dict with queue_depth, active_workers, total_processed, total_failed.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the processor is running with active workers."""
        ...
