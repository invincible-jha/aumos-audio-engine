"""Business logic services for aumos-audio-engine.

All services operate on domain models and depend only on Protocol interfaces,
not on concrete adapter implementations (dependency inversion).
"""

import uuid
from decimal import Decimal

from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_audio_engine.api.schemas import (
    AudioDeidentifyRequest,
    AudioSynthesizeRequest,
    AudioTranscribeRequest,
    BatchAudioRequest,
    MNPIScanRequest,
    StyleTransferRequest,
)
from aumos_audio_engine.core.interfaces import (
    AudioStorageProtocol,
    MNPIDetectorProtocol,
    PrivacyClientProtocol,
    SpeakerDeidentifierProtocol,
    StyleTransferProtocol,
    TranscriberProtocol,
    TTSEngineProtocol,
)
from aumos_audio_engine.core.models import AudioSynthesisJob, JobStatus, JobType, VoiceProfile

logger = get_logger(__name__)


class SynthesisService:
    """Orchestrates text-to-speech synthesis jobs.

    Coordinates voice profile selection, TTS generation, privacy validation,
    and audio storage. Publishes job lifecycle events to Kafka.
    """

    def __init__(
        self,
        tts_engine: TTSEngineProtocol,
        storage: AudioStorageProtocol,
        privacy_client: PrivacyClientProtocol,
        publisher: EventPublisher,
        job_repository: "AudioJobRepository",
        voice_profile_repository: "VoiceProfileRepository",
        default_sample_rate: int = 22050,
    ) -> None:
        """Initialize synthesis service with injected dependencies.

        Args:
            tts_engine: TTS synthesis engine (Coqui or compatible).
            storage: Audio file storage adapter.
            privacy_client: Privacy engine compliance validator.
            publisher: Kafka event publisher.
            job_repository: Repository for AudioSynthesisJob persistence.
            voice_profile_repository: Repository for VoiceProfile lookup.
            default_sample_rate: Default output sample rate when not specified.
        """
        self._tts = tts_engine
        self._storage = storage
        self._privacy = privacy_client
        self._publisher = publisher
        self._job_repo = job_repository
        self._voice_repo = voice_profile_repository
        self._default_sample_rate = default_sample_rate

    async def synthesize(
        self,
        request: AudioSynthesizeRequest,
        tenant_id: str,
        correlation_id: str,
    ) -> AudioSynthesisJob:
        """Run a text-to-speech synthesis job end-to-end.

        Creates a job record, calls TTS engine, validates privacy compliance,
        uploads output to storage, and publishes completion event.

        Args:
            request: Synthesis request parameters.
            tenant_id: Tenant context for isolation.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            Completed AudioSynthesisJob with output_uri populated.

        Raises:
            NotFoundError: If specified voice_profile_id does not exist.
            ValidationError: If privacy engine rejects the output.
        """
        # Resolve voice style config
        style_config: dict = {}
        voice_profile_id: uuid.UUID | None = None

        if request.voice_profile_id:
            profile = await self._voice_repo.get_by_id(request.voice_profile_id, tenant_id=tenant_id)
            if profile is None:
                raise NotFoundError(f"VoiceProfile {request.voice_profile_id} not found")
            style_config = profile.style_config
            voice_profile_id = profile.id
        elif request.voice_style_config:
            style_config = request.voice_style_config

        sample_rate = request.sample_rate or self._default_sample_rate
        output_format = request.output_format or "wav"

        # Create job record
        job = await self._job_repo.create(
            tenant_id=tenant_id,
            job_type=JobType.SYNTHESIZE,
            status=JobStatus.PENDING,
            input_config={
                "text": request.text,
                "voice_profile_id": str(voice_profile_id) if voice_profile_id else None,
                "style_config": style_config,
                "output_format": output_format,
                "sample_rate": sample_rate,
            },
            voice_profile_id=voice_profile_id,
            sample_rate=sample_rate,
            output_format=output_format,
        )

        logger.info(
            "Synthesis job started",
            job_id=str(job.id),
            tenant_id=tenant_id,
            text_length=len(request.text),
            output_format=output_format,
            correlation_id=correlation_id,
        )

        try:
            # Update status to processing
            job = await self._job_repo.update_status(job.id, JobStatus.PROCESSING)

            # Run TTS synthesis
            audio_bytes = await self._tts.synthesize(
                text=request.text,
                voice_style_config=style_config,
                output_format=output_format,
                sample_rate=sample_rate,
            )

            # Estimate duration from byte count (rough: bytes / (sample_rate * 2))
            estimated_duration = Decimal(len(audio_bytes)) / Decimal(sample_rate * 2)

            # Privacy engine validation (synthetic audio — no biometric risk)
            privacy_result = await self._privacy.validate_audio_job(
                job_id=str(job.id),
                tenant_id=tenant_id,
                job_type="synthesize",
                deidentification_applied=False,
                similarity_score=None,
            )

            if not privacy_result["approved"]:
                raise ValidationError(
                    f"Privacy engine rejected synthesis job: {privacy_result['reason']}"
                )

            # Upload to storage
            output_uri = await self._storage.upload(
                tenant_id=tenant_id,
                job_id=str(job.id),
                audio_bytes=audio_bytes,
                audio_format=output_format,
            )

            # Mark complete
            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=output_uri,
                duration_seconds=estimated_duration,
                privacy_engine_validated=True,
            )

            # Publish completion event
            await self._publisher.publish(
                Topics.AUDIO_JOB_LIFECYCLE,
                {
                    "event_type": "synthesis_completed",
                    "job_id": str(job.id),
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "output_uri": output_uri,
                },
            )

            logger.info(
                "Synthesis job completed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                output_uri=output_uri,
                duration_seconds=float(estimated_duration),
            )
            return job

        except Exception as exc:
            logger.error(
                "Synthesis job failed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                error=str(exc),
            )
            await self._job_repo.update(
                job.id,
                status=JobStatus.FAILED,
                error_message=str(exc),
            )
            raise


class DeidentificationService:
    """Orchestrates speaker de-identification jobs.

    Removes biometric voice identifiers while preserving semantic content.
    Validates de-identification quality before storing output.
    """

    def __init__(
        self,
        deidentifier: SpeakerDeidentifierProtocol,
        storage: AudioStorageProtocol,
        privacy_client: PrivacyClientProtocol,
        publisher: EventPublisher,
        job_repository: "AudioJobRepository",
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize de-identification service.

        Args:
            deidentifier: Speaker de-identification adapter.
            storage: Audio file storage adapter.
            privacy_client: Privacy engine compliance validator.
            publisher: Kafka event publisher.
            job_repository: Repository for job persistence.
            similarity_threshold: Maximum speaker similarity score allowed (reject if exceeded).
        """
        self._deidentifier = deidentifier
        self._storage = storage
        self._privacy = privacy_client
        self._publisher = publisher
        self._job_repo = job_repository
        self._threshold = similarity_threshold

    async def deidentify(
        self,
        request: AudioDeidentifyRequest,
        input_audio: bytes,
        tenant_id: str,
        correlation_id: str,
    ) -> AudioSynthesisJob:
        """Run speaker de-identification end-to-end.

        Downloads input audio, applies de-identification, validates similarity
        threshold is met, then stores and returns the processed audio.

        Args:
            request: De-identification request parameters.
            input_audio: Raw input audio bytes.
            tenant_id: Tenant context.
            correlation_id: Distributed tracing ID.

        Returns:
            Completed job with de-identified audio at output_uri.

        Raises:
            ValidationError: If achieved similarity score exceeds threshold.
        """
        output_format = request.output_format or "wav"

        job = await self._job_repo.create(
            tenant_id=tenant_id,
            job_type=JobType.DEIDENTIFY,
            status=JobStatus.PENDING,
            input_config={
                "input_format": request.input_format,
                "output_format": output_format,
                "threshold": request.similarity_threshold or self._threshold,
            },
            output_format=output_format,
        )

        logger.info(
            "De-identification job started",
            job_id=str(job.id),
            tenant_id=tenant_id,
            input_format=request.input_format,
            correlation_id=correlation_id,
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.PROCESSING)

            target_threshold = request.similarity_threshold or self._threshold

            # Apply de-identification
            processed_audio, deident_metadata = await self._deidentifier.deidentify(
                audio_bytes=input_audio,
                input_format=request.input_format,
                output_format=output_format,
                target_similarity_threshold=target_threshold,
            )

            achieved_similarity = float(deident_metadata.get("achieved_similarity", 0.0))

            # Enforce similarity threshold
            if achieved_similarity > target_threshold:
                raise ValidationError(
                    f"De-identification failed: achieved similarity {achieved_similarity:.3f} "
                    f"exceeds threshold {target_threshold:.3f}. "
                    "Biometric identity may still be recoverable."
                )

            # Privacy engine validation
            privacy_result = await self._privacy.validate_audio_job(
                job_id=str(job.id),
                tenant_id=tenant_id,
                job_type="deidentify",
                deidentification_applied=True,
                similarity_score=achieved_similarity,
            )

            if not privacy_result["approved"]:
                raise ValidationError(
                    f"Privacy engine rejected de-identification: {privacy_result['reason']}"
                )

            # Upload processed audio
            output_uri = await self._storage.upload(
                tenant_id=tenant_id,
                job_id=str(job.id),
                audio_bytes=processed_audio,
                audio_format=output_format,
            )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=output_uri,
                deidentification_applied=True,
                privacy_engine_validated=True,
                input_config={
                    **job.input_config,
                    "achieved_similarity": achieved_similarity,
                    "deidentification_metadata": deident_metadata,
                },
            )

            await self._publisher.publish(
                Topics.AUDIO_JOB_LIFECYCLE,
                {
                    "event_type": "deidentification_completed",
                    "job_id": str(job.id),
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "achieved_similarity": achieved_similarity,
                },
            )

            logger.info(
                "De-identification job completed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                achieved_similarity=achieved_similarity,
            )
            return job

        except Exception as exc:
            logger.error(
                "De-identification job failed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                error=str(exc),
            )
            await self._job_repo.update(job.id, status=JobStatus.FAILED, error_message=str(exc))
            raise


class TranscriptionService:
    """Orchestrates audio transcription jobs via WhisperX / faster-whisper."""

    def __init__(
        self,
        transcriber: TranscriberProtocol,
        mnpi_detector: MNPIDetectorProtocol,
        publisher: EventPublisher,
        job_repository: "AudioJobRepository",
        auto_mnpi_scan: bool = True,
    ) -> None:
        """Initialize transcription service.

        Args:
            transcriber: WhisperX / faster-whisper transcription adapter.
            mnpi_detector: MNPI detection service for auto-scan after transcription.
            publisher: Kafka event publisher.
            job_repository: Repository for job persistence.
            auto_mnpi_scan: Whether to automatically run MNPI detection after transcription.
        """
        self._transcriber = transcriber
        self._mnpi = mnpi_detector
        self._publisher = publisher
        self._job_repo = job_repository
        self._auto_mnpi_scan = auto_mnpi_scan

    async def transcribe(
        self,
        request: AudioTranscribeRequest,
        input_audio: bytes,
        tenant_id: str,
        correlation_id: str,
    ) -> AudioSynthesisJob:
        """Transcribe audio to text with optional automatic MNPI scanning.

        Args:
            request: Transcription request parameters.
            input_audio: Raw audio bytes to transcribe.
            tenant_id: Tenant context.
            correlation_id: Distributed tracing ID.

        Returns:
            Completed job with transcript populated.
        """
        job = await self._job_repo.create(
            tenant_id=tenant_id,
            job_type=JobType.TRANSCRIBE,
            status=JobStatus.PENDING,
            input_config={
                "audio_format": request.audio_format,
                "language": request.language,
                "auto_mnpi_scan": request.auto_mnpi_scan if hasattr(request, "auto_mnpi_scan") else self._auto_mnpi_scan,
            },
        )

        logger.info(
            "Transcription job started",
            job_id=str(job.id),
            tenant_id=tenant_id,
            audio_format=request.audio_format,
            correlation_id=correlation_id,
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.PROCESSING)

            # Run transcription
            transcription_result = await self._transcriber.transcribe(
                audio_bytes=input_audio,
                audio_format=request.audio_format,
                language=request.language,
            )

            transcript_text = transcription_result["text"]
            mnpi_detected: bool | None = None
            mnpi_result: dict | None = None

            # Auto MNPI scan if enabled
            run_mnpi = getattr(request, "auto_mnpi_scan", self._auto_mnpi_scan)
            if run_mnpi and transcript_text.strip():
                mnpi_result = await self._mnpi.scan(
                    text=transcript_text,
                    tenant_id=tenant_id,
                    context_metadata={"job_id": str(job.id), "job_type": "transcribe"},
                )
                mnpi_detected = mnpi_result.get("mnpi_detected", False)

                if mnpi_detected:
                    logger.warning(
                        "MNPI detected in transcription",
                        job_id=str(job.id),
                        tenant_id=tenant_id,
                        risk_level=mnpi_result.get("risk_level"),
                        matched_keywords=mnpi_result.get("matched_keywords", []),
                    )

                    # Publish MNPI alert
                    await self._publisher.publish(
                        Topics.COMPLIANCE_ALERTS,
                        {
                            "event_type": "mnpi_detected",
                            "job_id": str(job.id),
                            "tenant_id": tenant_id,
                            "risk_level": mnpi_result.get("risk_level"),
                            "correlation_id": correlation_id,
                        },
                    )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                transcript=transcript_text,
                mnpi_detected=mnpi_detected,
                mnpi_scan_result=mnpi_result,
                input_config={
                    **job.input_config,
                    "transcription_metadata": {
                        "language": transcription_result.get("language"),
                        "confidence": transcription_result.get("confidence"),
                        "segment_count": len(transcription_result.get("segments", [])),
                    },
                },
            )

            await self._publisher.publish(
                Topics.AUDIO_JOB_LIFECYCLE,
                {
                    "event_type": "transcription_completed",
                    "job_id": str(job.id),
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "mnpi_detected": mnpi_detected,
                },
            )

            logger.info(
                "Transcription job completed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                transcript_length=len(transcript_text),
                mnpi_detected=mnpi_detected,
            )
            return job

        except Exception as exc:
            logger.error(
                "Transcription job failed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                error=str(exc),
            )
            await self._job_repo.update(job.id, status=JobStatus.FAILED, error_message=str(exc))
            raise


class MNPIService:
    """Orchestrates Material Non-Public Information detection on audio transcripts.

    Designed specifically for financial sector compliance — earnings call recordings,
    analyst briefings, board discussions, and other regulated communications.
    """

    def __init__(
        self,
        mnpi_detector: MNPIDetectorProtocol,
        transcriber: TranscriberProtocol,
        publisher: EventPublisher,
        job_repository: "AudioJobRepository",
    ) -> None:
        """Initialize MNPI detection service.

        Args:
            mnpi_detector: MNPI content detector.
            transcriber: Transcriber for audio inputs.
            publisher: Kafka event publisher for compliance alerts.
            job_repository: Repository for job persistence.
        """
        self._detector = mnpi_detector
        self._transcriber = transcriber
        self._publisher = publisher
        self._job_repo = job_repository

    async def scan_audio(
        self,
        request: MNPIScanRequest,
        input_audio: bytes | None,
        tenant_id: str,
        correlation_id: str,
    ) -> AudioSynthesisJob:
        """Scan audio (or existing transcript) for MNPI content.

        If audio_bytes provided, transcribes first. If transcript provided
        directly in request, skips transcription.

        Args:
            request: MNPI scan request with audio or transcript.
            input_audio: Raw audio bytes (None if scanning transcript directly).
            tenant_id: Tenant context.
            correlation_id: Distributed tracing ID.

        Returns:
            Completed job with mnpi_detected and mnpi_scan_result populated.
        """
        job = await self._job_repo.create(
            tenant_id=tenant_id,
            job_type=JobType.MNPI_SCAN,
            status=JobStatus.PENDING,
            input_config={
                "scan_mode": "audio" if input_audio else "transcript",
                "audio_format": getattr(request, "audio_format", None),
                "context_metadata": getattr(request, "context_metadata", {}),
            },
        )

        logger.info(
            "MNPI scan job started",
            job_id=str(job.id),
            tenant_id=tenant_id,
            scan_mode="audio" if input_audio else "transcript",
            correlation_id=correlation_id,
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.PROCESSING)

            transcript_text = ""

            # Transcribe if audio input provided
            if input_audio is not None:
                audio_format = getattr(request, "audio_format", "wav")
                language = getattr(request, "language", None)

                transcription_result = await self._transcriber.transcribe(
                    audio_bytes=input_audio,
                    audio_format=audio_format,
                    language=language,
                )
                transcript_text = transcription_result["text"]
            elif hasattr(request, "transcript") and request.transcript:
                transcript_text = request.transcript

            if not transcript_text.strip():
                raise ValidationError("No text content available for MNPI scanning")

            # Run MNPI detection
            scan_result = await self._detector.scan(
                text=transcript_text,
                tenant_id=tenant_id,
                context_metadata=getattr(request, "context_metadata", {}),
            )

            mnpi_detected = scan_result.get("mnpi_detected", False)
            risk_level = scan_result.get("risk_level", "none")

            if mnpi_detected:
                logger.warning(
                    "MNPI detected in audio scan",
                    job_id=str(job.id),
                    tenant_id=tenant_id,
                    risk_level=risk_level,
                    keywords_matched=len(scan_result.get("matched_keywords", [])),
                )

                await self._publisher.publish(
                    Topics.COMPLIANCE_ALERTS,
                    {
                        "event_type": "mnpi_detected",
                        "job_id": str(job.id),
                        "tenant_id": tenant_id,
                        "risk_level": risk_level,
                        "correlation_id": correlation_id,
                        "flagged_segments_count": len(scan_result.get("flagged_segments", [])),
                    },
                )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                transcript=transcript_text,
                mnpi_detected=mnpi_detected,
                mnpi_scan_result=scan_result,
            )

            await self._publisher.publish(
                Topics.AUDIO_JOB_LIFECYCLE,
                {
                    "event_type": "mnpi_scan_completed",
                    "job_id": str(job.id),
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "mnpi_detected": mnpi_detected,
                },
            )

            logger.info(
                "MNPI scan completed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                mnpi_detected=mnpi_detected,
                risk_level=risk_level,
            )
            return job

        except Exception as exc:
            logger.error(
                "MNPI scan job failed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                error=str(exc),
            )
            await self._job_repo.update(job.id, status=JobStatus.FAILED, error_message=str(exc))
            raise


class StyleTransferService:
    """Orchestrates voice style transfer without biometric identity retention.

    Extracts stylistic features (prosody, energy, speaking rate) from source
    audio, then synthesizes output with those features applied to a target voice.
    The source speaker's biometric identity is discarded after feature extraction.
    """

    def __init__(
        self,
        style_transfer: StyleTransferProtocol,
        storage: AudioStorageProtocol,
        privacy_client: PrivacyClientProtocol,
        publisher: EventPublisher,
        job_repository: "AudioJobRepository",
    ) -> None:
        """Initialize style transfer service.

        Args:
            style_transfer: Style transfer adapter.
            storage: Audio file storage.
            privacy_client: Privacy engine validator.
            publisher: Kafka event publisher.
            job_repository: Repository for job persistence.
        """
        self._style_transfer = style_transfer
        self._storage = storage
        self._privacy = privacy_client
        self._publisher = publisher
        self._job_repo = job_repository

    async def transfer_style(
        self,
        request: StyleTransferRequest,
        source_audio: bytes,
        tenant_id: str,
        correlation_id: str,
    ) -> AudioSynthesisJob:
        """Apply style transfer from source audio to output audio.

        Extracts non-biometric style features from source and applies them
        to synthesis. Source biometrics are never stored.

        Args:
            request: Style transfer request parameters.
            source_audio: Raw source audio bytes.
            tenant_id: Tenant context.
            correlation_id: Distributed tracing ID.

        Returns:
            Completed job with style-transferred audio at output_uri.
        """
        output_format = getattr(request, "output_format", "wav")

        job = await self._job_repo.create(
            tenant_id=tenant_id,
            job_type=JobType.STYLE_TRANSFER,
            status=JobStatus.PENDING,
            input_config={
                "source_format": getattr(request, "source_format", "wav"),
                "target_style_config": getattr(request, "target_style_config", {}),
                "output_format": output_format,
                "preserve_semantics": getattr(request, "preserve_semantics", True),
            },
            output_format=output_format,
        )

        logger.info(
            "Style transfer job started",
            job_id=str(job.id),
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.PROCESSING)

            output_audio, style_metadata = await self._style_transfer.transfer_style(
                source_audio_bytes=source_audio,
                source_format=getattr(request, "source_format", "wav"),
                target_style_config=getattr(request, "target_style_config", {}),
                output_format=output_format,
                preserve_semantics=getattr(request, "preserve_semantics", True),
            )

            # Privacy validation
            privacy_result = await self._privacy.validate_audio_job(
                job_id=str(job.id),
                tenant_id=tenant_id,
                job_type="style_transfer",
                deidentification_applied=True,
                similarity_score=style_metadata.get("source_similarity_score"),
            )

            if not privacy_result["approved"]:
                raise ValidationError(
                    f"Privacy engine rejected style transfer output: {privacy_result['reason']}"
                )

            output_uri = await self._storage.upload(
                tenant_id=tenant_id,
                job_id=str(job.id),
                audio_bytes=output_audio,
                audio_format=output_format,
            )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=output_uri,
                privacy_engine_validated=True,
                input_config={
                    **job.input_config,
                    "style_metadata": style_metadata,
                },
            )

            await self._publisher.publish(
                Topics.AUDIO_JOB_LIFECYCLE,
                {
                    "event_type": "style_transfer_completed",
                    "job_id": str(job.id),
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                },
            )

            logger.info(
                "Style transfer job completed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                output_uri=output_uri,
            )
            return job

        except Exception as exc:
            logger.error(
                "Style transfer job failed",
                job_id=str(job.id),
                tenant_id=tenant_id,
                error=str(exc),
            )
            await self._job_repo.update(job.id, status=JobStatus.FAILED, error_message=str(exc))
            raise


# Type aliases for repository references (imported lazily to avoid circular imports)
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from aumos_audio_engine.adapters.repositories import AudioJobRepository, VoiceProfileRepository
