"""SQLAlchemy repositories for aumos-audio-engine.

Extends BaseRepository from aumos-common. RLS tenant isolation
is handled automatically by the session context.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import BaseRepository
from aumos_common.observability import get_logger

from aumos_audio_engine.core.models import AudioSynthesisJob, JobStatus, VoiceProfile

logger = get_logger(__name__)


class AudioJobRepository(BaseRepository[AudioSynthesisJob]):
    """Repository for AudioSynthesisJob persistence.

    All queries are tenant-scoped via RLS set by the session context.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: Async SQLAlchemy session with RLS context set.
        """
        super().__init__(session, AudioSynthesisJob)

    async def create(
        self,
        tenant_id: str,
        job_type: str,
        status: JobStatus,
        input_config: dict,
        voice_profile_id: uuid.UUID | None = None,
        sample_rate: int | None = None,
        output_format: str | None = None,
    ) -> AudioSynthesisJob:
        """Create a new audio synthesis job.

        Args:
            tenant_id: Owning tenant.
            job_type: Job type enum value.
            status: Initial status.
            input_config: Input configuration dict.
            voice_profile_id: Optional linked voice profile.
            sample_rate: Audio sample rate.
            output_format: Output format string.

        Returns:
            Created AudioSynthesisJob with generated id.
        """
        job = AudioSynthesisJob(
            tenant_id=uuid.UUID(tenant_id),
            job_type=job_type,
            status=status,
            input_config=input_config,
            voice_profile_id=voice_profile_id,
            sample_rate=sample_rate,
            output_format=output_format,
        )
        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(job)

        logger.info(
            "AudioSynthesisJob created",
            job_id=str(job.id),
            tenant_id=tenant_id,
            job_type=str(job_type),
        )
        return job

    async def get_by_id(
        self,
        job_id: uuid.UUID,
        tenant_id: str | None = None,
    ) -> AudioSynthesisJob | None:
        """Retrieve a job by ID.

        Args:
            job_id: UUID of the job.
            tenant_id: Optional explicit tenant filter (RLS handles isolation automatically).

        Returns:
            AudioSynthesisJob if found, None otherwise.
        """
        stmt = select(AudioSynthesisJob).where(AudioSynthesisJob.id == job_id)
        if tenant_id:
            stmt = stmt.where(AudioSynthesisJob.tenant_id == uuid.UUID(tenant_id))

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
    ) -> AudioSynthesisJob:
        """Update the status of an existing job.

        Args:
            job_id: UUID of the job.
            status: New status value.

        Returns:
            Updated AudioSynthesisJob.

        Raises:
            ValueError: If job not found.
        """
        job = await self.get_by_id(job_id)
        if job is None:
            raise ValueError(f"AudioSynthesisJob {job_id} not found")

        job.status = status
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def update(
        self,
        job_id: uuid.UUID,
        **fields: Any,
    ) -> AudioSynthesisJob:
        """Update arbitrary fields on a job.

        Args:
            job_id: UUID of the job to update.
            **fields: Field names and values to update.

        Returns:
            Updated AudioSynthesisJob.

        Raises:
            ValueError: If job not found.
        """
        job = await self.get_by_id(job_id)
        if job is None:
            raise ValueError(f"AudioSynthesisJob {job_id} not found")

        for field_name, value in fields.items():
            if hasattr(job, field_name):
                setattr(job, field_name, value)

        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def list_by_tenant(
        self,
        tenant_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AudioSynthesisJob]:
        """List jobs for a tenant, newest first.

        Args:
            tenant_id: Owning tenant ID.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of AudioSynthesisJob ordered by created_at desc.
        """
        stmt = (
            select(AudioSynthesisJob)
            .where(AudioSynthesisJob.tenant_id == uuid.UUID(tenant_id))
            .order_by(AudioSynthesisJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class VoiceProfileRepository(BaseRepository[VoiceProfile]):
    """Repository for VoiceProfile persistence.

    Ensures all profiles are tenant-scoped. Never stores biometric embeddings.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: Async SQLAlchemy session with RLS context set.
        """
        super().__init__(session, VoiceProfile)

    async def create(
        self,
        tenant_id: str,
        name: str,
        style_config: dict,
        description: str | None = None,
        is_synthetic: bool = True,
    ) -> VoiceProfile:
        """Create a new voice profile.

        Args:
            tenant_id: Owning tenant.
            name: Profile display name.
            style_config: Voice style parameters (NO biometric data).
            description: Optional description.
            is_synthetic: Whether this profile was generated synthetically.

        Returns:
            Created VoiceProfile.
        """
        profile = VoiceProfile(
            tenant_id=uuid.UUID(tenant_id),
            name=name,
            style_config=style_config,
            description=description,
            is_synthetic=is_synthetic,
        )
        self.session.add(profile)
        await self.session.flush()
        await self.session.refresh(profile)

        logger.info(
            "VoiceProfile created",
            profile_id=str(profile.id),
            tenant_id=tenant_id,
            name=name,
            is_synthetic=is_synthetic,
        )
        return profile

    async def get_by_id(
        self,
        profile_id: uuid.UUID,
        tenant_id: str | None = None,
    ) -> VoiceProfile | None:
        """Retrieve a voice profile by ID.

        Args:
            profile_id: UUID of the profile.
            tenant_id: Optional tenant filter.

        Returns:
            VoiceProfile if found, None otherwise.
        """
        stmt = select(VoiceProfile).where(VoiceProfile.id == profile_id)
        if tenant_id:
            stmt = stmt.where(VoiceProfile.tenant_id == uuid.UUID(tenant_id))

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        tenant_id: str,
    ) -> list[VoiceProfile]:
        """List all voice profiles for a tenant.

        Args:
            tenant_id: Owning tenant ID.

        Returns:
            List of VoiceProfile ordered by name.
        """
        stmt = (
            select(VoiceProfile)
            .where(VoiceProfile.tenant_id == uuid.UUID(tenant_id))
            .order_by(VoiceProfile.name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
