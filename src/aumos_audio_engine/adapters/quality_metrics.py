"""Audio quality metrics adapter for de-identification fidelity measurement.

Computes PESQ (Perceptual Evaluation of Speech Quality) and POLQA-estimate
scores comparing de-identified audio against original. Scores are stored
on the job record and included in fidelity validator payloads.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import structlog
from aumos_common.observability import get_logger


@dataclass
class AudioQualityResult:
    """Result of audio quality evaluation between original and de-identified audio.

    Attributes:
        pesq_score: PESQ MOS-LQO score (range: -0.5 to 4.5, higher = better quality).
            Acceptable range: >= 1.0 (de-identified audio is intelligible).
        polqa_estimate: POLQA-like estimate computed via P.563 approximation.
            Range: 1.0 to 5.0, higher = better quality.
        snr_db: Signal-to-noise ratio in dB between original and de-identified.
        original_rms: RMS energy of original audio.
        deidentified_rms: RMS energy of de-identified audio.
        sample_rate: Sample rate used for evaluation.
        mode: PESQ mode used ("wb" = wideband 16kHz, "nb" = narrowband 8kHz).
    """

    pesq_score: float
    polqa_estimate: float
    snr_db: float
    original_rms: float
    deidentified_rms: float
    sample_rate: int
    mode: str


class AudioQualityEvaluator:
    """Computes perceptual audio quality metrics between original and processed audio.

    Primary metric is PESQ (ITU-T P.862) which correlates strongly with
    human subjective speech quality ratings. Higher scores indicate better
    preservation of speech intelligibility after de-identification.

    PESQ score interpretation:
    - 4.0-4.5: Excellent (imperceptible degradation)
    - 3.0-4.0: Good (some degradation, fully intelligible)
    - 2.0-3.0: Fair (noticeable degradation but intelligible)
    - 1.0-2.0: Poor (significant degradation, may affect intelligibility)
    - < 1.0: Bad (unintelligible)

    Args:
        default_mode: PESQ mode ("wb" for 16kHz wideband, "nb" for 8kHz narrowband).
    """

    def __init__(self, default_mode: str = "wb") -> None:
        """Initialize AudioQualityEvaluator.

        Args:
            default_mode: Default PESQ mode ("wb" or "nb").
        """
        if default_mode not in {"wb", "nb"}:
            raise ValueError(f"PESQ mode must be 'wb' or 'nb', got '{default_mode}'")
        self._default_mode = default_mode
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def compute(
        self,
        original_audio: np.ndarray,
        deidentified_audio: np.ndarray,
        sample_rate: int,
        mode: str | None = None,
    ) -> AudioQualityResult:
        """Compute PESQ and supplementary quality metrics.

        Args:
            original_audio: Reference float32 audio samples normalized to [-1, 1].
            deidentified_audio: De-identified float32 audio samples.
            sample_rate: Audio sample rate in Hz.
            mode: PESQ mode override ("wb" or "nb").

        Returns:
            AudioQualityResult with all computed metrics.
        """
        effective_mode = mode or self._default_mode
        return await asyncio.to_thread(
            self._compute_sync,
            original_audio=original_audio,
            deidentified_audio=deidentified_audio,
            sample_rate=sample_rate,
            mode=effective_mode,
        )

    def _compute_sync(
        self,
        original_audio: np.ndarray,
        deidentified_audio: np.ndarray,
        sample_rate: int,
        mode: str,
    ) -> AudioQualityResult:
        """Compute quality metrics synchronously (called via to_thread).

        Args:
            original_audio: Reference audio float32 array.
            deidentified_audio: Processed audio float32 array.
            sample_rate: Sample rate in Hz.
            mode: PESQ evaluation mode.

        Returns:
            AudioQualityResult with PESQ score and supplementary metrics.
        """
        # Ensure equal length — trim to shorter
        min_len = min(len(original_audio), len(deidentified_audio))
        ref = original_audio[:min_len]
        deg = deidentified_audio[:min_len]

        # Resample for PESQ mode requirements
        target_rate = 16000 if mode == "wb" else 8000
        if sample_rate != target_rate:
            ref = self._resample(ref, sample_rate, target_rate)
            deg = self._resample(deg, sample_rate, target_rate)

        pesq_score = self._compute_pesq(ref, deg, target_rate, mode)
        polqa_estimate = self._estimate_polqa(pesq_score)
        snr_db = self._compute_snr(ref, deg)

        original_rms = float(np.sqrt(np.mean(original_audio ** 2)))
        deidentified_rms = float(np.sqrt(np.mean(deidentified_audio ** 2)))

        self._log.info(
            "audio_quality_computed",
            pesq=pesq_score,
            polqa_estimate=polqa_estimate,
            snr_db=snr_db,
            mode=mode,
        )

        return AudioQualityResult(
            pesq_score=pesq_score,
            polqa_estimate=polqa_estimate,
            snr_db=snr_db,
            original_rms=original_rms,
            deidentified_rms=deidentified_rms,
            sample_rate=sample_rate,
            mode=mode,
        )

    def _compute_pesq(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
        sample_rate: int,
        mode: str,
    ) -> float:
        """Compute PESQ score using the pesq Python package.

        Args:
            reference: Reference float32 audio.
            degraded: Degraded float32 audio.
            sample_rate: Sample rate (must be 8000 or 16000).
            mode: PESQ mode ("wb" or "nb").

        Returns:
            PESQ MOS-LQO score.
        """
        try:
            from pesq import pesq

            # Convert to 16-bit PCM as required by pesq
            ref_16 = (reference * 32768.0).clip(-32768, 32767).astype(np.int16)
            deg_16 = (degraded * 32768.0).clip(-32768, 32767).astype(np.int16)
            return float(pesq(sample_rate, ref_16.astype(np.float32) / 32768.0,
                              deg_16.astype(np.float32) / 32768.0, mode))
        except ImportError:
            self._log.warning("pesq_package_not_installed_using_snr_fallback")
            return self._snr_to_pesq_estimate(self._compute_snr(reference, degraded))

    def _estimate_polqa(self, pesq_score: float) -> float:
        """Estimate POLQA score from PESQ using empirical mapping.

        ITU-T P.863 (POLQA) typically rates 0.3-0.5 higher than PESQ for
        clean speech. This is an approximation for reporting purposes.

        Args:
            pesq_score: PESQ MOS-LQO score.

        Returns:
            Estimated POLQA MOS-LQO score.
        """
        # Linear approximation: POLQA ≈ PESQ * 1.08 + 0.2 (empirical)
        return min(5.0, max(1.0, pesq_score * 1.08 + 0.2))

    def _compute_snr(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """Compute signal-to-noise ratio in dB.

        Args:
            reference: Reference signal.
            degraded: Degraded signal.

        Returns:
            SNR in dB (higher = less degradation).
        """
        signal_power = np.mean(reference ** 2)
        noise = reference - degraded
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-12:
            return 100.0
        return float(10.0 * np.log10(signal_power / noise_power))

    def _snr_to_pesq_estimate(self, snr_db: float) -> float:
        """Convert SNR to approximate PESQ score when pesq package unavailable.

        Args:
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Rough PESQ estimate based on SNR.
        """
        # Very rough linear mapping: SNR 0dB ≈ 1.0, SNR 30dB ≈ 4.5
        return min(4.5, max(-0.5, snr_db / 10.0 + 1.0))

    def _resample(
        self,
        audio: np.ndarray,
        original_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Resample audio to target rate using librosa.

        Args:
            audio: Float32 audio array.
            original_rate: Source sample rate.
            target_rate: Target sample rate.

        Returns:
            Resampled float32 audio array.
        """
        import librosa
        return librosa.resample(y=audio, orig_sr=original_rate, target_sr=target_rate)
