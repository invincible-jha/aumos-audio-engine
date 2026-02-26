# Changelog

All notable changes to `aumos-audio-engine` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial scaffolding for aumos-audio-engine
- Hexagonal architecture: api/, core/, adapters/ layers
- Coqui TTS integration for text-to-speech synthesis
- AudioCraft integration for environmental sound generation
- Speaker de-identification preserving semantic content
- WhisperX / faster-whisper transcription integration
- MNPI (Material Non-Public Information) detection for financial audio
- Voice style transfer without biometric identity retention
- SQLAlchemy models: AudioSynthesisJob, VoiceProfile
- FastAPI endpoints: synthesize, deidentify, transcribe, mnpi-scan, batch
- Kafka event publishing for job lifecycle events
- MinIO/S3 storage adapter for audio output files
- Privacy engine client for biometric compliance checks
- Full async implementation throughout
