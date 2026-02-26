"""SQLAlchemy ORM models for aumos-audio-engine.

Table prefix: aud_
All models extend AumOSModel for tenant isolation, id, and timestamps.
"""

import enum
import uuid
from decimal import Decimal

from sqlalchemy import Boolean, Enum, ForeignKey, Integer, Numeric, String, Text
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
