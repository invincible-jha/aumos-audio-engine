"""FastAPI dependency injection for aumos-audio-engine services.

Provides `Depends()` factories for all core services.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import get_db_session
from aumos_common.events import EventPublisher

from aumos_audio_engine.settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


def get_event_publisher() -> EventPublisher:
    """Return Kafka event publisher instance."""
    settings = get_settings()
    return EventPublisher(bootstrap_servers=settings.kafka_bootstrap_servers)


def get_tts_engine() -> "CoquiTTSEngine":
    """Return Coqui TTS engine instance."""
    from aumos_audio_engine.adapters.tts_engine import CoquiTTSEngine

    settings = get_settings()
    return CoquiTTSEngine(settings=settings)


def get_deidentifier() -> "SpeakerDeidentifierAdapter":
    """Return speaker de-identifier adapter."""
    from aumos_audio_engine.adapters.speaker_deidentifier import SpeakerDeidentifierAdapter

    settings = get_settings()
    return SpeakerDeidentifierAdapter(settings=settings)


def get_transcriber() -> "WhisperXTranscriber":
    """Return WhisperX transcription adapter."""
    from aumos_audio_engine.adapters.transcriber import WhisperXTranscriber

    settings = get_settings()
    return WhisperXTranscriber(settings=settings)


def get_mnpi_detector() -> "MNPIDetectorAdapter":
    """Return MNPI detector adapter."""
    from aumos_audio_engine.adapters.mnpi_detector import MNPIDetectorAdapter

    settings = get_settings()
    return MNPIDetectorAdapter(settings=settings)


def get_style_transfer_engine() -> "StyleTransferAdapter":
    """Return style transfer adapter."""
    from aumos_audio_engine.adapters.style_transfer import StyleTransferAdapter

    settings = get_settings()
    return StyleTransferAdapter(settings=settings)


def get_storage_adapter() -> "StorageAdapter":
    """Return MinIO storage adapter."""
    from aumos_audio_engine.adapters.storage import StorageAdapter

    settings = get_settings()
    return StorageAdapter(settings=settings)


def get_privacy_client() -> "PrivacyEngineClient":
    """Return privacy engine HTTP client."""
    from aumos_audio_engine.adapters.privacy_client import PrivacyEngineClient

    settings = get_settings()
    return PrivacyEngineClient(settings=settings)


def get_voice_profile_repository(
    session: AsyncSession = Depends(get_db_session),
) -> "VoiceProfileRepository":
    """Return VoiceProfile repository for the current session."""
    from aumos_audio_engine.adapters.repositories import VoiceProfileRepository

    return VoiceProfileRepository(session)


def get_synthesis_service(
    session: AsyncSession = Depends(get_db_session),
) -> "SynthesisService":
    """Build and return a configured SynthesisService."""
    from aumos_audio_engine.adapters.repositories import AudioJobRepository, VoiceProfileRepository
    from aumos_audio_engine.core.services import SynthesisService

    settings = get_settings()
    return SynthesisService(
        tts_engine=get_tts_engine(),
        storage=get_storage_adapter(),
        privacy_client=get_privacy_client(),
        publisher=get_event_publisher(),
        job_repository=AudioJobRepository(session),
        voice_profile_repository=VoiceProfileRepository(session),
        default_sample_rate=settings.sample_rate,
    )


def get_deidentification_service(
    session: AsyncSession = Depends(get_db_session),
) -> "DeidentificationService":
    """Build and return a configured DeidentificationService."""
    from aumos_audio_engine.adapters.repositories import AudioJobRepository
    from aumos_audio_engine.core.services import DeidentificationService

    settings = get_settings()
    return DeidentificationService(
        deidentifier=get_deidentifier(),
        storage=get_storage_adapter(),
        privacy_client=get_privacy_client(),
        publisher=get_event_publisher(),
        job_repository=AudioJobRepository(session),
        similarity_threshold=settings.deidentification_threshold,
    )


def get_transcription_service(
    session: AsyncSession = Depends(get_db_session),
) -> "TranscriptionService":
    """Build and return a configured TranscriptionService."""
    from aumos_audio_engine.adapters.repositories import AudioJobRepository
    from aumos_audio_engine.core.services import TranscriptionService

    return TranscriptionService(
        transcriber=get_transcriber(),
        mnpi_detector=get_mnpi_detector(),
        publisher=get_event_publisher(),
        job_repository=AudioJobRepository(session),
        auto_mnpi_scan=True,
    )


def get_mnpi_service(
    session: AsyncSession = Depends(get_db_session),
) -> "MNPIService":
    """Build and return a configured MNPIService."""
    from aumos_audio_engine.adapters.repositories import AudioJobRepository
    from aumos_audio_engine.core.services import MNPIService

    return MNPIService(
        mnpi_detector=get_mnpi_detector(),
        transcriber=get_transcriber(),
        publisher=get_event_publisher(),
        job_repository=AudioJobRepository(session),
    )


def get_style_transfer_service(
    session: AsyncSession = Depends(get_db_session),
) -> "StyleTransferService":
    """Build and return a configured StyleTransferService."""
    from aumos_audio_engine.adapters.repositories import AudioJobRepository
    from aumos_audio_engine.core.services import StyleTransferService

    return StyleTransferService(
        style_transfer=get_style_transfer_engine(),
        storage=get_storage_adapter(),
        privacy_client=get_privacy_client(),
        publisher=get_event_publisher(),
        job_repository=AudioJobRepository(session),
    )


# TYPE_CHECKING imports for annotations above
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from aumos_audio_engine.adapters.mnpi_detector import MNPIDetectorAdapter
    from aumos_audio_engine.adapters.privacy_client import PrivacyEngineClient
    from aumos_audio_engine.adapters.repositories import AudioJobRepository, VoiceProfileRepository
    from aumos_audio_engine.adapters.speaker_deidentifier import SpeakerDeidentifierAdapter
    from aumos_audio_engine.adapters.storage import StorageAdapter
    from aumos_audio_engine.adapters.style_transfer import StyleTransferAdapter
    from aumos_audio_engine.adapters.transcriber import WhisperXTranscriber
    from aumos_audio_engine.adapters.tts_engine import CoquiTTSEngine
    from aumos_audio_engine.core.services import (
        DeidentificationService,
        MNPIService,
        StyleTransferService,
        SynthesisService,
        TranscriptionService,
    )
