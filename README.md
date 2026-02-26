# aumos-audio-engine

**AumOS Audio Synthesis Engine** — Synthetic audio generation without biometric retention, speaker de-identification, and MNPI detection for the financial sector.

Part of **AumOS Enterprise Data Factory** (Tier B: Open Core).

## Overview

`aumos-audio-engine` provides:

- **Text-to-Speech Synthesis** — High-quality voice synthesis via Coqui TTS with configurable voice styles
- **Environmental Sound Generation** — AudioCraft-powered ambient and contextual audio synthesis
- **Speaker De-identification** — Remove biometric voice identifiers while preserving semantic content and prosody
- **Audio Transcription** — WhisperX / faster-whisper integration for accurate audio-to-text conversion
- **MNPI Detection** — Material Non-Public Information scanning for financial sector compliance
- **Voice Style Transfer** — Transfer voice characteristics without retaining biometric identity

## Architecture

```
aumos-platform-core
    └── aumos-auth-gateway
        └── aumos-audio-engine (this repo)
            ├── → aumos-privacy-engine (biometric compliance validation)
            ├── → aumos-event-bus (job lifecycle events)
            └── → aumos-data-layer (job persistence)
```

### Hexagonal Architecture

```
src/aumos_audio_engine/
├── main.py              # FastAPI entry point
├── settings.py          # Service configuration
├── api/                 # HTTP layer (routes + schemas)
├── core/                # Business logic (models + services + interfaces)
└── adapters/            # External integrations (TTS, WhisperX, AudioCraft, etc.)
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env

# Start dependencies
docker compose -f docker-compose.dev.yml up -d

# Run migrations
make migrate

# Start service
uvicorn aumos_audio_engine.main:app --reload --port 8001
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/audio/synthesize` | Text-to-speech synthesis |
| POST | `/api/v1/audio/deidentify` | Speaker de-identification |
| POST | `/api/v1/audio/transcribe` | Audio transcription |
| POST | `/api/v1/audio/mnpi-scan` | MNPI content detection |
| GET  | `/api/v1/audio/jobs/{id}` | Get job status and result |
| POST | `/api/v1/audio/batch` | Batch audio processing |

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `AUMOS_AUDIO_GPU_ENABLED` | Enable GPU acceleration | `false` |
| `AUMOS_AUDIO_TTS_MODEL` | Coqui TTS model identifier | `tts_models/en/ljspeech/tacotron2-DDC` |
| `AUMOS_AUDIO_WHISPER_MODEL` | WhisperX model size | `base` |
| `AUMOS_AUDIO_PRIVACY_ENGINE_URL` | Privacy engine service URL | `http://localhost:8010` |
| `AUMOS_AUDIO_DEIDENTIFICATION_THRESHOLD` | Voice similarity rejection threshold | `0.85` |
| `AUMOS_AUDIO_MNPI_KEYWORDS_PATH` | Path to MNPI keyword list | `/app/config/mnpi_keywords.json` |

## Development

```bash
make install    # Install with dev dependencies
make test       # Run tests with coverage
make lint       # Run ruff linter
make typecheck  # Run mypy strict type checking
make format     # Auto-format code
```

## Financial Sector Compliance

This service is designed for financial sector deployments:

- **MNPI Detection**: Scans audio transcripts for material non-public information before storage/transmission
- **Biometric Retention Prohibition**: Speaker de-identification is enforced — no raw voice prints are stored
- **Audit Trail**: All synthesis and transcription jobs are logged with tenant context
- **Data Retention**: Configurable audio retention windows per regulatory requirements

## License

Apache 2.0 — see [LICENSE](LICENSE).
