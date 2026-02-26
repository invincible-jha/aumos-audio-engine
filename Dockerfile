# ============================================================================
# Dockerfile — aumos-audio-engine (GPU-capable multi-stage build)
# ============================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --prefix=/install --no-warn-script-location .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime audio/ML dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security: non-root user
RUN groupadd -r aumos && useradd -r -g aumos -d /app -s /sbin/nologin aumos

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
WORKDIR /app

# Create temp dir for audio processing
RUN mkdir -p /tmp/aumos-audio && chown aumos:aumos /tmp/aumos-audio

# Set ownership
RUN chown -R aumos:aumos /app

USER aumos

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/live'); r.raise_for_status()" || exit 1

# Start service
CMD ["uvicorn", "aumos_audio_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
