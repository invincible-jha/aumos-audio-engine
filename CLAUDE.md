# CLAUDE.md — AumOS Audio Synthesis Engine

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-audio-engine`) is part of **Tier B: Open Core**:
Data Factory synthesis engines for enterprise audio generation and processing.

**Release Tier:** B: Open Core
**Product Mapping:** Product 1 — Data Factory (Audio Engine)
**Phase:** 1A (Months 3-8)

## Repo Purpose

`aumos-audio-engine` provides synthetic audio generation without biometric retention,
speaker de-identification preserving semantic content, WhisperX transcription, MNPI
(Material Non-Public Information) detection for financial sector compliance, and voice
style transfer. It is a core component of the AumOS Data Factory pipeline.

## Architecture Position

```
aumos-platform-core → aumos-auth-gateway → aumos-audio-engine
                                          ↘ aumos-privacy-engine (biometric validation)
                                          ↘ aumos-event-bus (job lifecycle events)
                                          ↘ aumos-data-layer (persistence)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-privacy-engine` — biometric compliance validation before audio storage

**Downstream dependents (other repos IMPORT from this):**
- `aumos-data-pipeline` — orchestrates audio jobs as pipeline stages
- `aumos-fidelity-validator` — validates audio output quality

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |
| TTS | 0.22.0+ | Coqui TTS text-to-speech synthesis |
| audiocraft | 1.2.0+ | Meta AudioCraft environmental sound synthesis |
| faster-whisper | 1.0.0+ | WhisperX / faster-whisper transcription |
| torchaudio | 2.2.0+ | Audio tensor processing |
| soundfile | 0.12.0+ | Audio file I/O |
| librosa | 0.10.0+ | Audio feature extraction |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

## Database Conventions

- Table prefix: `aud_` (e.g., `aud_synthesis_jobs`, `aud_voice_profiles`)
- ALL tenant-scoped tables: extend `AumOSModel` (gets id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table (created in migration)

## Repo-Specific Context

### Audio Processing Pipeline

1. **Input validation** — Validate text/audio input at API boundary (Pydantic schemas)
2. **Privacy check** — Call aumos-privacy-engine before storing any biometric data
3. **Job creation** — Create `AudioSynthesisJob` record with `status=pending`
4. **Processing** — Delegate to appropriate adapter (TTS, WhisperX, etc.)
5. **De-identification** — Apply speaker de-identification if processing real voice audio
6. **MNPI scan** — Scan transcripts for material non-public information
7. **Storage** — Upload processed audio to MinIO, store URI on job record
8. **Event publish** — Publish `AudioJobCompleted` event to Kafka
9. **Job update** — Update job status to `completed` with output metadata

### MNPI Detection

- Combines keyword matching with contextual LLM analysis
- Keywords loaded from configurable JSON file (path in settings)
- Context window: 50 tokens around keyword matches for LLM review
- Detection result stored on `AudioSynthesisJob.mnpi_detected`
- Detected MNPI triggers alert event on Kafka compliance topic

### Speaker De-identification

- Target: voice similarity score < `AUMOS_AUDIO_DEIDENTIFICATION_THRESHOLD` (default 0.85)
- Algorithm: pitch shifting + formant modification + temporal perturbation
- Semantic preservation: verified via round-trip transcription comparison
- No voice embeddings are stored persistently

### GPU / CPU Handling

- Check `AUMOS_AUDIO_GPU_ENABLED` setting before loading CUDA models
- All ML adapters must implement CPU fallback
- TTS and WhisperX are the primary GPU consumers

### Financial Sector Requirements

- All transcription output is treated as sensitive financial data
- MNPI detection is mandatory for all financial institution tenants
- Audit log entry for every synthesis/transcription job
- Voice biometric data must never leave the processing pipeline

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.**
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT store raw voice audio** without privacy engine validation.
8. **Do NOT skip MNPI scanning** for financial sector tenants.
9. **Do NOT put business logic in API routes.**
10. **Do NOT bypass RLS.**
