"""SQLAlchemy ORM models for aumos-audio-engine.

Table prefix: aud_
All models extend AumOSModel for tenant isolation, id, and timestamps.
"""

import enum
import uuid
from decimal import Decimal

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel, TenantMixin


class JobType(str, enum.Enum):
    """Type of audio processing job."""

    SYNTHESIZE = "synthesize"
    DEIDENTIFY = "deidentify"
    TRANSCRIBE = "transcribe"
    MNPI_SCAN = "mnpi_scan"
    STYLE_TRANSFER = "style_transfer"


class JobStatus(str, enum.Enum):
    """Processing status of an audio job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioFormat(str, enum.Enum):
    """Supported audio output formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    OPUS = "opus"


class AudioSynthesisJob(AumOSModel, TenantMixin):
    """Represents a single audio processing job.

    Covers synthesis, de-identification, transcription, MNPI scanning,
    and style transfer operations. All operations are tenant-scoped.

    Table: aud_synthesis_jobs
    """

    __tablename__ = "aud_synthesis_jobs"

    job_type: Mapped[JobType] = mapped_column(
        Enum(JobType, name="aud_job_type"),
        nullable=False,
        index=True,
        doc="Type of audio processing operation",
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="aud_job_status"),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
        doc="Current processing status",
    )

    # Input configuration — captures all parameters for reproducibility
    input_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Input parameters for this job (text, voice config, audio settings)",
    )

    # Audio metadata
    duration_seconds: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 3),
        nullable=True,
        doc="Duration of input or output audio in seconds",
    )
    sample_rate: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Audio sample rate in Hz",
    )
    output_format: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        doc="Output audio format (wav, mp3, flac, ogg)",
    )

    # Compliance and privacy
    mnpi_detected: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        doc="Whether MNPI content was detected in this audio job",
    )
    mnpi_scan_result: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="Detailed MNPI scan results including matched keywords and context",
    )
    deidentification_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether speaker de-identification was applied",
    )
    privacy_engine_validated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether privacy engine validated this job before output storage",
    )

    # Output
    output_uri: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="MinIO/S3 URI for the processed audio output",
    )
    transcript: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Transcription result for transcribe or mnpi_scan jobs",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Error details if job failed",
    )

    # Voice profile reference (optional — for synthesis jobs using a profile)
    voice_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("aud_voice_profiles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Reference to the VoiceProfile used for synthesis",
    )
    voice_profile: Mapped["VoiceProfile | None"] = relationship(
        "VoiceProfile",
        back_populates="synthesis_jobs",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<AudioSynthesisJob id={self.id} type={self.job_type} status={self.status}>"


class VoiceProfile(AumOSModel, TenantMixin):
    """Represents a synthetic voice profile for TTS synthesis.

    Voice profiles define style parameters for synthesis without storing
    any biometric voice data. Profiles may be built from de-identified
    style embeddings only.

    Table: aud_voice_profiles
    """

    __tablename__ = "aud_voice_profiles"

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Human-readable name for this voice profile",
    )

    # Style configuration — all synthesis parameters, NO biometric embeddings
    style_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Voice style parameters (pitch, rate, energy, prosody). Never biometric embeddings.",
    )

    is_synthetic: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="True if this profile was generated synthetically (no source speaker biometrics)",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Optional description of this voice style",
    )

    # Relationship back to jobs
    synthesis_jobs: Mapped[list["AudioSynthesisJob"]] = relationship(
        "AudioSynthesisJob",
        back_populates="voice_profile",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<VoiceProfile id={self.id} name={self.name} synthetic={self.is_synthetic}>"


class AudioStreamSession(AumOSModel, TenantMixin):
    """Tracks a real-time de-identification streaming session.

    Records metrics for WebSocket streaming sessions (frame count,
    latency, duration) for audit and performance analysis.

    Table: aud_stream_sessions
    """

    __tablename__ = "aud_stream_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique session identifier used to correlate WebSocket connections",
    )
    frames_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of 20ms audio frames processed in this session",
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Total session duration in seconds",
    )
    avg_processing_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Average per-frame processing time in milliseconds",
    )
    deidentification_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether de-identification was applied to this session",
    )

    def __repr__(self) -> str:
        return (
            f"<AudioStreamSession id={self.id} "
            f"session_id={self.session_id} frames={self.frames_processed}>"
        )


class MNPILibrary(AumOSModel, TenantMixin):
    """A library of MNPI detection patterns for a specific industry sector.

    System libraries (is_system_library=True) are pre-seeded at startup.
    Tenant-defined libraries can extend or override system patterns.

    Table: aud_mnpi_libraries
    """

    __tablename__ = "aud_mnpi_libraries"

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Library name (e.g., 'pharma', 'energy', 'ma_general')",
    )
    sector: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Industry sector this library covers",
    )
    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0.0",
        doc="Semantic version of this library",
    )
    pattern_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of patterns in this library (denormalized for quick display)",
    )
    is_system_library: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="True if this is a pre-seeded system library (not tenant-created)",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Human-readable description of library scope and coverage",
    )

    def __repr__(self) -> str:
        return f"<MNPILibrary id={self.id} name={self.name} sector={self.sector}>"


class MNPIPattern(AumOSModel, TenantMixin):
    """A single MNPI detection pattern within a library.

    Patterns can be regex strings or exact keyword phrases.
    The risk_level determines alert severity when matched.

    Table: aud_mnpi_patterns
    """

    __tablename__ = "aud_mnpi_patterns"

    library_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("aud_mnpi_libraries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Parent MNPI library this pattern belongs to",
    )
    pattern: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Regex pattern or exact keyword for MNPI detection",
    )
    pattern_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="keyword",
        doc="Pattern type: 'keyword' for exact match, 'regex' for regex match",
    )
    context_window: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=50,
        doc="Token window around match to include in context for LLM review",
    )
    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        index=True,
        doc="Risk level: 'high' (immediate alert), 'medium' (review), 'low' (log only)",
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Human-readable description of what MNPI scenario this pattern detects",
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        doc="Whether this pattern is active in detection",
    )

    def __repr__(self) -> str:
        return (
            f"<MNPIPattern id={self.id} "
            f"library_id={self.library_id} "
            f"risk={self.risk_level} "
            f"pattern={self.pattern[:30]!r}>"
        )
