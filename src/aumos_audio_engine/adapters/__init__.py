"""Adapters layer — external integrations for aumos-audio-engine."""

from aumos_audio_engine.adapters.audiocraft_engine import AudioCraftEngine
from aumos_audio_engine.adapters.batch_processor import AudioBatchProcessor
from aumos_audio_engine.adapters.export_handler import AudioExportHandler
from aumos_audio_engine.adapters.mnpi_detector import MNPIDetector
from aumos_audio_engine.adapters.quality_evaluator import AudioQualityEvaluator
from aumos_audio_engine.adapters.speaker_deidentifier import SpeakerDeidentifierAdapter
from aumos_audio_engine.adapters.tts_engine import CoquiTTSEngine
from aumos_audio_engine.adapters.voice_style_transfer import VoiceStyleTransfer
from aumos_audio_engine.adapters.whisperx_transcriber import WhisperXTranscriber

__all__ = [
    "AudioBatchProcessor",
    "AudioCraftEngine",
    "AudioExportHandler",
    "AudioQualityEvaluator",
    "CoquiTTSEngine",
    "MNPIDetector",
    "SpeakerDeidentifierAdapter",
    "VoiceStyleTransfer",
    "WhisperXTranscriber",
]
