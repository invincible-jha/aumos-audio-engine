"""Pydantic request and response schemas for aumos-audio-engine API."""

import uuid
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator


# ─── Request Schemas ──────────────────────────────────────────────────────────


class AudioSynthesizeRequest(BaseModel):
    """Request body for POST /audio/synthesize."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to synthesize into speech",
    )
    voice_profile_id: uuid.UUID | None = Field(
        default=None,
        description="ID of a saved VoiceProfile to use for synthesis",
    )
    voice_style_config: dict | None = Field(
        default=None,
        description="Inline voice style parameters (used if voice_profile_id not provided)",
    )
    output_format: str = Field(
        default="wav",
        description="Output audio format: wav, mp3, flac, ogg",
        pattern=r"^(wav|mp3|flac|ogg|opus)$",
    )
    sample_rate: int | None = Field(
        default=None,
        ge=8000,
        le=48000,
        description="Output sample rate in Hz. Defaults to service setting (22050).",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        """Reject whitespace-only text."""
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class AudioDeidentifyRequest(BaseModel):
    """Request body for POST /audio/deidentify (multipart form data wrapper)."""

    input_format: str = Field(
        default="wav",
        description="Format of the uploaded audio",
        pattern=r"^(wav|mp3|flac|ogg|opus)$",
    )
    output_format: str = Field(
        default="wav",
        description="Desired output format",
        pattern=r"^(wav|mp3|flac|ogg|opus)$",
    )
    similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum allowed speaker similarity score. Defaults to service setting (0.85).",
    )


class AudioTranscribeRequest(BaseModel):
    """Request body for POST /audio/transcribe (multipart form data wrapper)."""

    audio_format: str = Field(
        default="wav",
        description="Format of the uploaded audio",
        pattern=r"^(wav|mp3|flac|ogg|opus|m4a)$",
    )
    language: str | None = Field(
        default=None,
        description="BCP-47 language code (e.g. 'en', 'fr'). Auto-detected if not provided.",
        min_length=2,
        max_length=10,
    )
    auto_mnpi_scan: bool = Field(
        default=True,
        description="Automatically run MNPI detection on transcription result",
    )


class MNPIScanRequest(BaseModel):
    """Request body for POST /audio/mnpi-scan."""

    transcript: str | None = Field(
        default=None,
        description="Pre-existing transcript text to scan (provide either this or audio upload)",
    )
    audio_format: str | None = Field(
        default=None,
        description="Audio format if uploading audio file instead of transcript",
        pattern=r"^(wav|mp3|flac|ogg|opus|m4a)$",
    )
    language: str | None = Field(
        default=None,
        description="Language hint for transcription step",
    )
    context_metadata: dict = Field(
        default_factory=dict,
        description="Optional context (meeting type, participants, jurisdiction, etc.)",
    )


class StyleTransferRequest(BaseModel):
    """Request body for POST /audio/style-transfer."""

    source_format: str = Field(
        default="wav",
        description="Format of the source audio",
        pattern=r"^(wav|mp3|flac|ogg|opus)$",
    )
    target_style_config: dict = Field(
        default_factory=dict,
        description="Target style overrides. Empty dict = extract style from source only.",
    )
    output_format: str = Field(
        default="wav",
        description="Output audio format",
        pattern=r"^(wav|mp3|flac|ogg|opus)$",
    )
    preserve_semantics: bool = Field(
        default=True,
        description="Verify transcript matches after style transfer",
    )


class VoiceProfileCreateRequest(BaseModel):
    """Request body for POST /audio/voice-profiles."""

    name: str = Field(..., min_length=1, max_length=255, description="Display name for this voice profile")
    style_config: dict = Field(
        default_factory=dict,
        description="Voice style parameters. Must NOT contain biometric embeddings.",
    )
    description: str | None = Field(default=None, max_length=1000)


class BatchAudioRequest(BaseModel):
    """Request body for POST /audio/batch."""

    jobs: list[AudioSynthesizeRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of synthesis requests to process in batch",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop batch processing on first failure if True",
    )


# ─── Response Schemas ─────────────────────────────────────────────────────────


class AudioJobResponse(BaseModel):
    """Response schema for a single audio processing job."""

    id: uuid.UUID
    job_type: str
    status: str
    output_format: str | None
    sample_rate: int | None
    duration_seconds: Decimal | None
    mnpi_detected: bool | None
    output_uri: str | None
    transcript: str | None
    deidentification_applied: bool
    privacy_engine_validated: bool
    error_message: str | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class MNPIScanResponse(BaseModel):
    """Response schema including MNPI scan details."""

    job_id: uuid.UUID
    status: str
    mnpi_detected: bool | None
    risk_level: str | None
    confidence: float | None
    matched_keywords: list[str]
    flagged_segments: list[dict]
    transcript_excerpt: str | None

    model_config = {"from_attributes": True}


class VoiceProfileResponse(BaseModel):
    """Response schema for a voice profile."""

    id: uuid.UUID
    name: str
    style_config: dict
    is_synthetic: bool
    description: str | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class BatchAudioResponse(BaseModel):
    """Response schema for a batch audio processing request."""

    total: int
    submitted: int
    failed: int
    job_ids: list[uuid.UUID]
    errors: list[dict]


# ─── MNPI Library Schemas ──────────────────────────────────────────────────────


class MNPIPatternCreate(BaseModel):
    """Request schema for creating a single MNPI detection pattern."""

    pattern: str = Field(..., min_length=1, max_length=2000, description="Regex or keyword pattern string")
    pattern_type: str = Field(
        default="keyword",
        description="Pattern type: 'keyword' for exact match, 'regex' for regex match",
        pattern=r"^(keyword|regex)$",
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level: 'high', 'medium', or 'low'",
        pattern=r"^(high|medium|low)$",
    )
    description: str = Field(default="", max_length=500, description="Human-readable description of what this pattern detects")
    context_window: int = Field(default=50, ge=10, le=500, description="Token window around match for LLM context")
    enabled: bool = Field(default=True, description="Whether this pattern is active")


class MNPILibraryCreateRequest(BaseModel):
    """Request body for POST /audio/mnpi/libraries."""

    name: str = Field(..., min_length=1, max_length=255, description="Library name (e.g., 'custom_finserv')")
    sector: str = Field(..., min_length=1, max_length=100, description="Industry sector this library covers")
    description: str | None = Field(default=None, max_length=1000)
    patterns: list[MNPIPatternCreate] = Field(
        default_factory=list,
        max_length=500,
        description="Initial patterns to include in the library",
    )


class MNPIPatternResponse(BaseModel):
    """Response schema for a single MNPI pattern."""

    id: uuid.UUID
    library_id: uuid.UUID
    pattern: str
    pattern_type: str
    risk_level: str
    description: str
    context_window: int
    enabled: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class MNPILibraryResponse(BaseModel):
    """Response schema for an MNPI detection library."""

    id: uuid.UUID
    name: str
    sector: str
    version: str
    pattern_count: int
    is_system_library: bool
    description: str | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class StreamingSessionResponse(BaseModel):
    """Response schema for a real-time de-identification streaming session."""

    id: uuid.UUID
    session_id: uuid.UUID
    frames_processed: int
    duration_seconds: float | None
    avg_processing_ms: float | None
    deidentification_applied: bool
    created_at: str

    model_config = {"from_attributes": True}
