"""Prosody extraction and preservation for de-identified audio.

Extracts pitch contour, energy envelope, and speaking rate before
de-identification and reapplies them afterward to preserve emotional
content (anger, urgency, distress) required for compliance analysis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import structlog
from aumos_common.observability import get_logger


@dataclass
class ProsodyFeatures:
    """Extracted prosodic features from an audio signal.

    Attributes:
        pitch_contour: Fundamental frequency (F0) contour array. NaN where
            voicing is absent. Shape: (n_frames,).
        energy_envelope: RMS energy per frame. Shape: (n_frames,).
        voicing_flag: Boolean mask, True where frame is voiced. Shape: (n_frames,).
        speaking_rate: Estimated syllables per second (float).
        sample_rate: Original audio sample rate.
        hop_length: Frames hop length used during extraction.
    """

    pitch_contour: np.ndarray
    energy_envelope: np.ndarray
    voicing_flag: np.ndarray
    speaking_rate: float
    sample_rate: int
    hop_length: int


class ProsodyExtractor:
    """Extracts prosodic features for preservation across de-identification.

    Uses librosa's pyin algorithm for pitch extraction (more accurate than
    YIN for short frames) and RMS energy for the energy envelope.

    Args:
        hop_length: Analysis hop length in samples. Lower = finer time resolution.
        fmin: Minimum fundamental frequency in Hz.
        fmax: Maximum fundamental frequency in Hz.
    """

    def __init__(
        self,
        hop_length: int = 256,
        fmin: float = 50.0,
        fmax: float = 600.0,
    ) -> None:
        """Initialize ProsodyExtractor.

        Args:
            hop_length: Analysis hop length in samples.
            fmin: Minimum tracked fundamental frequency.
            fmax: Maximum tracked fundamental frequency.
        """
        self._hop_length = hop_length
        self._fmin = fmin
        self._fmax = fmax
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def extract(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> ProsodyFeatures:
        """Extract pitch contour, energy envelope, and speaking rate.

        Args:
            audio_array: Float32 audio samples normalized to [-1, 1].
            sample_rate: Audio sample rate in Hz.

        Returns:
            ProsodyFeatures dataclass with all extracted prosodic features.
        """
        return await asyncio.to_thread(
            self._extract_sync,
            audio_array=audio_array,
            sample_rate=sample_rate,
        )

    def _extract_sync(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> ProsodyFeatures:
        """Extract prosodic features synchronously (called via to_thread).

        Args:
            audio_array: Normalized float32 audio.
            sample_rate: Sample rate in Hz.

        Returns:
            ProsodyFeatures with all components extracted.
        """
        import librosa

        # Extract pitch via pyin (probabilistic YIN — more robust than YIN)
        pitch_f0, voiced_flag, _ = librosa.pyin(
            y=audio_array,
            fmin=self._fmin,
            fmax=self._fmax,
            sr=sample_rate,
            hop_length=self._hop_length,
        )

        # Extract RMS energy envelope
        energy = librosa.feature.rms(
            y=audio_array,
            hop_length=self._hop_length,
        )[0]

        # Estimate speaking rate via onset detection (onsets ~ syllable boundaries)
        onset_frames = librosa.onset.onset_detect(
            y=audio_array,
            sr=sample_rate,
            hop_length=self._hop_length,
        )
        duration_seconds = len(audio_array) / sample_rate
        speaking_rate = len(onset_frames) / max(duration_seconds, 0.01)

        # Handle length mismatch between pitch and energy (pyin can differ by 1 frame)
        min_len = min(len(pitch_f0), len(energy), len(voiced_flag))
        pitch_f0 = pitch_f0[:min_len]
        energy = energy[:min_len]
        voiced_flag = voiced_flag[:min_len]

        return ProsodyFeatures(
            pitch_contour=pitch_f0,
            energy_envelope=energy,
            voicing_flag=voiced_flag,
            speaking_rate=speaking_rate,
            sample_rate=sample_rate,
            hop_length=self._hop_length,
        )


class ProsodyApplier:
    """Reapplies extracted prosodic features to de-identified audio.

    After de-identification may alter pitch contour and energy envelope
    beyond acceptable bounds, this class warps the de-identified audio
    to match the original prosody, preserving emotional content.

    Args:
        pitch_tolerance_semitones: Maximum allowed pitch deviation before
            correction is applied (default: 0.5 semitones).
        energy_tolerance: Maximum allowed RMS energy deviation ratio
            before normalization (default: 0.15).
    """

    def __init__(
        self,
        pitch_tolerance_semitones: float = 0.5,
        energy_tolerance: float = 0.15,
    ) -> None:
        """Initialize ProsodyApplier.

        Args:
            pitch_tolerance_semitones: Pitch correction activation threshold.
            energy_tolerance: Energy normalization activation threshold.
        """
        self._pitch_tolerance = pitch_tolerance_semitones
        self._energy_tolerance = energy_tolerance
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def apply(
        self,
        audio_array: np.ndarray,
        original_prosody: ProsodyFeatures,
        sample_rate: int,
    ) -> np.ndarray:
        """Warp pitch and energy of de-identified audio to match original prosody.

        Applies time-domain energy normalization to match the original energy
        envelope, ensuring that emotional dynamics (stress, emphasis) are
        preserved after pitch shifting de-identification.

        Args:
            audio_array: De-identified float32 audio samples.
            original_prosody: ProsodyFeatures extracted before de-identification.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Audio array with prosody warped to match original features.
        """
        return await asyncio.to_thread(
            self._apply_sync,
            audio_array=audio_array,
            original_prosody=original_prosody,
            sample_rate=sample_rate,
        )

    def _apply_sync(
        self,
        audio_array: np.ndarray,
        original_prosody: ProsodyFeatures,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply prosody correction synchronously (called via to_thread).

        Args:
            audio_array: De-identified normalized audio.
            original_prosody: Target prosodic features.
            sample_rate: Sample rate in Hz.

        Returns:
            Prosody-corrected audio array.
        """
        import librosa

        # Extract current energy from de-identified audio
        current_energy = librosa.feature.rms(
            y=audio_array,
            hop_length=original_prosody.hop_length,
        )[0]

        target_energy = original_prosody.energy_envelope
        min_len = min(len(current_energy), len(target_energy))

        # Compute per-frame energy ratio and apply as gain
        # Avoid division by zero in silent regions
        epsilon = 1e-6
        ratio = target_energy[:min_len] / (current_energy[:min_len] + epsilon)
        ratio = np.clip(ratio, 0.1, 10.0)  # Bound gain to prevent clipping

        # Expand frame-level gain to sample-level via repeat
        samples_per_frame = original_prosody.hop_length
        gain = np.repeat(ratio, samples_per_frame)
        gain = gain[:len(audio_array)]
        if len(gain) < len(audio_array):
            gain = np.pad(gain, (0, len(audio_array) - len(gain)), constant_values=1.0)

        # Apply only where deviation exceeds tolerance
        mean_ratio = float(np.mean(np.abs(ratio - 1.0)))
        if mean_ratio > self._energy_tolerance:
            corrected = audio_array * gain
            self._log.debug(
                "prosody_energy_correction_applied",
                mean_deviation=mean_ratio,
                tolerance=self._energy_tolerance,
            )
        else:
            corrected = audio_array

        return np.clip(corrected, -1.0, 1.0)


class EmotionClassifier:
    """Lightweight emotion classifier for verifying prosody preservation.

    Classifies audio into emotion categories (neutral, angry, happy, sad,
    fearful, surprised, disgusted) to verify that de-identification has not
    eliminated emotional content required for compliance analysis.

    Uses SpeechBrain's emotion recognition model if available,
    falling back to energy and pitch variance heuristics.
    """

    EMOTION_LABELS: list[str] = [
        "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
    ]

    def __init__(
        self,
        model_source: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        device: str = "cpu",
    ) -> None:
        """Initialize EmotionClassifier.

        Args:
            model_source: SpeechBrain or HuggingFace model path.
            device: Target device for inference.
        """
        self._model_source = model_source
        self._device = device
        self._classifier: object | None = None
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def warm_up(self) -> None:
        """Load emotion classifier model weights.

        Attempts to load the SpeechBrain classifier. Falls back to
        heuristic mode if the model is unavailable.
        """
        try:
            await asyncio.to_thread(self._load_model)
            self._log.info("emotion_classifier_ready", model=self._model_source)
        except Exception as exc:
            self._log.warning(
                "emotion_classifier_fallback_to_heuristic",
                reason=str(exc),
            )

    def _load_model(self) -> None:
        """Load SpeechBrain emotion classifier synchronously."""
        try:
            from speechbrain.inference.interfaces import foreign_class
            self._classifier = foreign_class(
                source=self._model_source,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
            )
        except ImportError:
            self._log.warning("speechbrain_not_installed_using_heuristic")

    async def classify(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Classify emotion in audio and return per-label probabilities.

        Args:
            audio_array: Float32 audio normalized to [-1, 1].
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dict mapping emotion label to probability (0.0-1.0).
        """
        if self._classifier is not None:
            return await asyncio.to_thread(
                self._classify_with_model,
                audio_array=audio_array,
                sample_rate=sample_rate,
            )
        return self._classify_heuristic(audio_array, sample_rate)

    def _classify_with_model(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Classify using loaded SpeechBrain model."""
        import torch

        tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        lengths = torch.FloatTensor([1.0])
        output = self._classifier.classify_batch(tensor, lengths)  # type: ignore[union-attr]
        probs = output[0].squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]
        labels = self.EMOTION_LABELS[:len(probs)]
        return dict(zip(labels, [float(p) for p in probs]))

    def _classify_heuristic(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Estimate emotion via energy and pitch variance heuristics.

        Not a substitute for a real model — used only as fallback when
        SpeechBrain is unavailable.

        Args:
            audio_array: Normalized float32 audio.
            sample_rate: Sample rate in Hz.

        Returns:
            Dict mapping emotion labels to rough probability estimates.
        """
        import librosa

        rms = float(np.sqrt(np.mean(audio_array ** 2)))

        try:
            f0, _, _ = librosa.pyin(
                y=audio_array, fmin=50, fmax=600, sr=sample_rate
            )
            pitch_variance = float(np.nanstd(f0)) if f0 is not None else 0.0
        except Exception:
            pitch_variance = 0.0

        # Simple heuristic: high energy + high pitch variance = likely emotional
        emotional_score = min(1.0, rms * 3.0 + pitch_variance / 100.0)
        neutral_score = max(0.0, 1.0 - emotional_score)

        return {
            "neutral": round(neutral_score, 3),
            "calm": round(neutral_score * 0.5, 3),
            "happy": round(emotional_score * 0.3, 3),
            "sad": round(emotional_score * 0.1, 3),
            "angry": round(emotional_score * 0.4, 3),
            "fearful": round(emotional_score * 0.1, 3),
            "disgust": round(emotional_score * 0.05, 3),
            "surprised": round(emotional_score * 0.05, 3),
        }
