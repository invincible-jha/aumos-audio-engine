"""Service-specific settings for aumos-audio-engine."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Audio engine configuration extending AumOS base settings.

    All standard settings (database, kafka, redis, jwt) are inherited from AumOSSettings.
    Audio-specific settings use the AUMOS_AUDIO_ prefix.
    """

    service_name: str = "aumos-audio-engine"

    # GPU acceleration
    gpu_enabled: bool = False

    # TTS (Coqui) settings
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    tts_vocoder_model: str = ""
    tts_use_cuda: bool = False

    # WhisperX / faster-whisper settings
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "en"

    # Audio processing settings
    sample_rate: int = 22050
    max_audio_duration_seconds: int = 300
    supported_output_formats: list[str] = ["wav", "mp3", "flac", "ogg"]

    # Speaker de-identification settings
    deidentification_threshold: float = 0.85
    pitch_shift_semitones_range: float = 4.0
    formant_shift_ratio_range: float = 0.15

    # MNPI detection settings
    mnpi_keywords_path: str = "/app/config/mnpi_keywords.json"
    mnpi_context_window_tokens: int = 50
    mnpi_confidence_threshold: float = 0.75

    # Storage
    output_bucket: str = "aumos-audio-outputs"
    temp_dir: str = "/tmp/aumos-audio"

    # Privacy engine integration
    privacy_engine_url: str = "http://localhost:8010"
    privacy_engine_timeout_seconds: int = 30

    model_config = SettingsConfigDict(env_prefix="AUMOS_AUDIO_")
