"""AumOS Audio Engine service entry point."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)
settings = Settings()


async def _check_privacy_engine() -> bool:
    """Check that the privacy engine is reachable."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.privacy_engine_url}/live")
            return response.status_code == 200
    except Exception:
        return False


async def _check_storage() -> bool:
    """Check that MinIO/S3 storage is reachable."""
    try:
        from aumos_audio_engine.adapters.storage import StorageAdapter

        storage = StorageAdapter(settings=settings)
        return await storage.health_check()
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: object) -> AsyncGenerator[None, None]:
    """Manage service startup and shutdown lifecycle."""
    logger.info(
        "Starting aumos-audio-engine",
        service=settings.service_name,
        gpu_enabled=settings.gpu_enabled,
        tts_model=settings.tts_model,
        whisper_model=settings.whisper_model,
    )

    # Initialize database connection pool
    init_database(settings.database)

    # Ensure temp directory exists
    os.makedirs(settings.temp_dir, exist_ok=True)

    # Pre-warm TTS model if GPU is enabled (avoids cold-start latency)
    if settings.gpu_enabled:
        logger.info("GPU enabled — pre-warming TTS model", model=settings.tts_model)
        try:
            from aumos_audio_engine.adapters.tts_engine import CoquiTTSEngine

            tts_engine = CoquiTTSEngine(settings=settings)
            await tts_engine.initialize()
            app.state.tts_engine = tts_engine  # type: ignore[attr-defined]
            logger.info("TTS model pre-warmed successfully")
        except Exception as exc:
            logger.warning("TTS model pre-warm failed — will initialize on first request", error=str(exc))

    logger.info("aumos-audio-engine startup complete")
    yield

    logger.info("Shutting down aumos-audio-engine")


app = create_app(
    service_name="aumos-audio-engine",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="privacy-engine", check_fn=_check_privacy_engine),
        HealthCheck(name="storage", check_fn=_check_storage),
    ],
)

# Include API router
from aumos_audio_engine.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
