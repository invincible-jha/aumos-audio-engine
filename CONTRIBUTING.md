# Contributing to aumos-audio-engine

Thank you for your interest in contributing to the AumOS Audio Engine.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/muveraai/aumos-audio-engine
cd aumos-audio-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Copy env file
cp .env.example .env
# Edit .env with your local settings

# Start local services
docker compose -f docker-compose.dev.yml up -d
```

## Code Standards

- **Type hints** required on all function signatures
- **Docstrings** (Google style) on all public classes and functions
- **Structured logging** via `aumos_common.observability.get_logger`
- **Async by default** — all I/O must be async
- **No raw SQL** — use SQLAlchemy ORM via BaseRepository

Run checks before submitting:
```bash
make lint       # ruff checks
make typecheck  # mypy strict
make test       # pytest with coverage
```

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feature/your-feature`
2. Write tests alongside your implementation
3. Ensure `make all` passes (lint + typecheck + test)
4. Submit PR with a clear description of changes and why

## Audio / ML Specifics

- TTS models are downloaded at runtime — do not commit model files
- Speaker de-identification must preserve semantic content (verify with test fixtures)
- MNPI detection changes require review from compliance team
- GPU code must have CPU fallback paths

## Commit Message Format

Follow conventional commits:
```
feat: add voice cloning endpoint
fix: correct sample rate normalization for WhisperX
refactor: extract audio preprocessing into utility functions
test: add MNPI detection edge case tests
docs: update API schema documentation
```

## Reporting Issues

Open a GitHub issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Audio sample if relevant (ensure no biometric data)
