"""FastAPI router for aumos-audio-engine.

All business logic is delegated to core services. Routes handle only:
- HTTP request parsing
- Auth/tenant extraction
- Service dispatch
- HTTP response formatting
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.observability import get_logger
from aumos_common.pagination import PageRequest, PageResponse, paginate

from aumos_audio_engine.api.schemas import (
    AudioDeidentifyRequest,
    AudioJobResponse,
    AudioSynthesizeRequest,
    AudioTranscribeRequest,
    BatchAudioRequest,
    BatchAudioResponse,
    MNPIScanRequest,
    MNPIScanResponse,
    StyleTransferRequest,
    VoiceProfileCreateRequest,
    VoiceProfileResponse,
)
from aumos_audio_engine.core.services import (
    DeidentificationService,
    MNPIService,
    StyleTransferService,
    SynthesisService,
    TranscriptionService,
)
from aumos_audio_engine.dependencies import (
    get_deidentification_service,
    get_mnpi_service,
    get_style_transfer_service,
    get_synthesis_service,
    get_transcription_service,
    get_voice_profile_repository,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])


# ─── Synthesis ────────────────────────────────────────────────────────────────


@router.post(
    "/synthesize",
    response_model=AudioJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Text-to-speech synthesis",
    description="Synthesize speech from text using Coqui TTS. Returns a job that completes asynchronously.",
)
async def synthesize_audio(
    request: AudioSynthesizeRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    service: SynthesisService = Depends(get_synthesis_service),
) -> AudioJobResponse:
    """Submit a text-to-speech synthesis job.

    Args:
        request: Synthesis parameters including text and optional voice profile.
        tenant: Authenticated tenant context (injected by auth middleware).
        session: Database session.
        service: Synthesis service instance.

    Returns:
        Job response with initial status PENDING or COMPLETED if synchronous.
    """
    try:
        job = await service.synthesize(
            request=request,
            tenant_id=str(tenant.tenant_id),
            correlation_id=str(uuid.uuid4()),
        )
        return AudioJobResponse.model_validate(job)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


# ─── De-identification ────────────────────────────────────────────────────────


@router.post(
    "/deidentify",
    response_model=AudioJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Speaker de-identification",
    description="Remove biometric voice identity from audio while preserving semantic content.",
)
async def deidentify_audio(
    audio_file: UploadFile = File(..., description="Audio file to de-identify"),
    input_format: str = Form(default="wav"),
    output_format: str = Form(default="wav"),
    similarity_threshold: float | None = Form(default=None),
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    service: DeidentificationService = Depends(get_deidentification_service),
) -> AudioJobResponse:
    """Submit a speaker de-identification job.

    Args:
        audio_file: Uploaded audio file containing speaker voice.
        input_format: Audio format of the upload.
        output_format: Desired output format.
        similarity_threshold: Maximum allowed speaker similarity after de-identification.
        tenant: Authenticated tenant context.
        session: Database session.
        service: De-identification service.

    Returns:
        Job response with de-identified audio URI when complete.
    """
    audio_bytes = await audio_file.read()

    deident_request = AudioDeidentifyRequest(
        input_format=input_format,
        output_format=output_format,
        similarity_threshold=similarity_threshold,
    )

    try:
        job = await service.deidentify(
            request=deident_request,
            input_audio=audio_bytes,
            tenant_id=str(tenant.tenant_id),
            correlation_id=str(uuid.uuid4()),
        )
        return AudioJobResponse.model_validate(job)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


# ─── Transcription ────────────────────────────────────────────────────────────


@router.post(
    "/transcribe",
    response_model=AudioJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Audio transcription",
    description="Transcribe audio to text using WhisperX. Optionally auto-runs MNPI detection.",
)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    audio_format: str = Form(default="wav"),
    language: str | None = Form(default=None),
    auto_mnpi_scan: bool = Form(default=True),
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    service: TranscriptionService = Depends(get_transcription_service),
) -> AudioJobResponse:
    """Submit an audio transcription job.

    Args:
        audio_file: Uploaded audio to transcribe.
        audio_format: Format of the audio file.
        language: Language hint for transcription. Auto-detected if None.
        auto_mnpi_scan: Run MNPI detection automatically on the transcript.
        tenant: Authenticated tenant context.
        session: Database session.
        service: Transcription service.

    Returns:
        Job response with transcript and optional MNPI scan results.
    """
    audio_bytes = await audio_file.read()

    transcribe_request = AudioTranscribeRequest(
        audio_format=audio_format,
        language=language,
        auto_mnpi_scan=auto_mnpi_scan,
    )

    try:
        job = await service.transcribe(
            request=transcribe_request,
            input_audio=audio_bytes,
            tenant_id=str(tenant.tenant_id),
            correlation_id=str(uuid.uuid4()),
        )
        return AudioJobResponse.model_validate(job)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


# ─── MNPI Detection ───────────────────────────────────────────────────────────


@router.post(
    "/mnpi-scan",
    response_model=MNPIScanResponse,
    status_code=status.HTTP_200_OK,
    summary="MNPI content detection",
    description=(
        "Scan audio or transcript for Material Non-Public Information. "
        "Designed for financial sector compliance (earnings calls, analyst briefings)."
    ),
)
async def scan_mnpi(
    transcript: str | None = Form(default=None, description="Pre-transcribed text to scan"),
    audio_file: UploadFile | None = File(default=None, description="Audio file to transcribe then scan"),
    audio_format: str = Form(default="wav"),
    language: str | None = Form(default=None),
    context_metadata: str = Form(default="{}", description="JSON string with context metadata"),
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    service: MNPIService = Depends(get_mnpi_service),
) -> MNPIScanResponse:
    """Scan content for MNPI.

    Args:
        transcript: Pre-existing transcript text (alternative to audio upload).
        audio_file: Audio file to transcribe then scan.
        audio_format: Format of audio file.
        language: Language hint for transcription.
        context_metadata: JSON context (meeting type, jurisdiction, etc.).
        tenant: Authenticated tenant context.
        session: Database session.
        service: MNPI detection service.

    Returns:
        MNPI scan results with risk level and flagged segments.
    """
    import json

    if not transcript and not audio_file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either transcript or audio_file must be provided",
        )

    try:
        context = json.loads(context_metadata)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="context_metadata must be valid JSON",
        )

    audio_bytes = await audio_file.read() if audio_file else None

    scan_request = MNPIScanRequest(
        transcript=transcript,
        audio_format=audio_format if audio_bytes else None,
        language=language,
        context_metadata=context,
    )

    try:
        job = await service.scan_audio(
            request=scan_request,
            input_audio=audio_bytes,
            tenant_id=str(tenant.tenant_id),
            correlation_id=str(uuid.uuid4()),
        )

        scan_result = job.mnpi_scan_result or {}
        transcript_text = job.transcript or ""

        return MNPIScanResponse(
            job_id=job.id,
            status=job.status.value,
            mnpi_detected=job.mnpi_detected,
            risk_level=scan_result.get("risk_level"),
            confidence=scan_result.get("confidence"),
            matched_keywords=scan_result.get("matched_keywords", []),
            flagged_segments=scan_result.get("flagged_segments", []),
            transcript_excerpt=transcript_text[:500] if transcript_text else None,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


# ─── Job Status ───────────────────────────────────────────────────────────────


@router.get(
    "/jobs/{job_id}",
    response_model=AudioJobResponse,
    summary="Get audio job status",
    description="Retrieve the current status and result of an audio processing job.",
)
async def get_job(
    job_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> AudioJobResponse:
    """Get the status of an audio processing job.

    Args:
        job_id: UUID of the job to retrieve.
        tenant: Authenticated tenant context.
        session: Database session.

    Returns:
        Current job status and output metadata.

    Raises:
        404: If job does not exist or belongs to a different tenant.
    """
    from aumos_audio_engine.adapters.repositories import AudioJobRepository

    repo = AudioJobRepository(session)
    job = await repo.get_by_id(job_id, tenant_id=str(tenant.tenant_id))

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio job {job_id} not found",
        )

    return AudioJobResponse.model_validate(job)


# ─── Batch Processing ─────────────────────────────────────────────────────────


@router.post(
    "/batch",
    response_model=BatchAudioResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch audio synthesis",
    description="Submit multiple text-to-speech synthesis requests in a single batch.",
)
async def batch_synthesize(
    request: BatchAudioRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    service: SynthesisService = Depends(get_synthesis_service),
) -> BatchAudioResponse:
    """Submit a batch of synthesis jobs.

    Args:
        request: Batch request with list of synthesis jobs.
        tenant: Authenticated tenant context.
        session: Database session.
        service: Synthesis service.

    Returns:
        Batch response with submitted job IDs and any per-job errors.
    """
    job_ids: list[uuid.UUID] = []
    errors: list[dict] = []
    correlation_id = str(uuid.uuid4())

    for index, synthesis_request in enumerate(request.jobs):
        try:
            job = await service.synthesize(
                request=synthesis_request,
                tenant_id=str(tenant.tenant_id),
                correlation_id=f"{correlation_id}-{index}",
            )
            job_ids.append(job.id)
        except Exception as exc:
            error_detail = {"index": index, "error": str(exc)}
            errors.append(error_detail)
            logger.warning(
                "Batch synthesis item failed",
                index=index,
                error=str(exc),
                tenant_id=str(tenant.tenant_id),
                fail_fast=request.fail_fast,
            )
            if request.fail_fast:
                break

    return BatchAudioResponse(
        total=len(request.jobs),
        submitted=len(job_ids),
        failed=len(errors),
        job_ids=job_ids,
        errors=errors,
    )


# ─── Voice Profiles ───────────────────────────────────────────────────────────


@router.post(
    "/voice-profiles",
    response_model=VoiceProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create voice profile",
    description="Create a synthetic voice profile for reuse in synthesis jobs.",
)
async def create_voice_profile(
    request: VoiceProfileCreateRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> VoiceProfileResponse:
    """Create a new voice profile.

    Args:
        request: Voice profile creation parameters.
        tenant: Authenticated tenant context.
        session: Database session.

    Returns:
        Created voice profile.
    """
    from aumos_audio_engine.adapters.repositories import VoiceProfileRepository
    from aumos_audio_engine.core.models import VoiceProfile

    repo = VoiceProfileRepository(session)
    profile = await repo.create(
        tenant_id=str(tenant.tenant_id),
        name=request.name,
        style_config=request.style_config,
        description=request.description,
        is_synthetic=True,
    )

    return VoiceProfileResponse.model_validate(profile)


@router.get(
    "/voice-profiles",
    response_model=list[VoiceProfileResponse],
    summary="List voice profiles",
    description="List all voice profiles for the current tenant.",
)
async def list_voice_profiles(
    tenant: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> list[VoiceProfileResponse]:
    """List voice profiles for the current tenant.

    Args:
        tenant: Authenticated tenant context.
        session: Database session.

    Returns:
        List of voice profiles.
    """
    from aumos_audio_engine.adapters.repositories import VoiceProfileRepository

    repo = VoiceProfileRepository(session)
    profiles = await repo.list_all(tenant_id=str(tenant.tenant_id))

    return [VoiceProfileResponse.model_validate(p) for p in profiles]
