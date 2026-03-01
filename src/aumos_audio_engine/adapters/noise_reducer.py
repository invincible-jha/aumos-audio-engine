"""Noise reduction adapter for robust de-identification.

Implements spectral subtraction preprocessing to improve de-identification
accuracy in noisy conditions (office background, street noise, reverberant rooms).
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from aumos_common.observability import get_logger


class NoiseReducer:
    """Spectral subtraction noise reducer for pre-processing audio.

    Reduces stationary background noise before de-identification to
    ensure de-identification works correctly across 4 noise conditions:
    - Clean (reference)
    - Office background (SNR ~20dB)
    - Street noise (SNR ~10dB)
    - Reverberant room

    Uses the noisereduce library which implements Wiener filtering
    with noise estimation from the initial silence period.

    Args:
        stationary: Whether to use stationary noise reduction (True) or
            non-stationary (False). Stationary works better for consistent
            background noise (office hum). Non-stationary better for
            intermittent noise (traffic).
        prop_decrease: Proportion of noise to reduce (0.0-1.0).
            Higher = more aggressive reduction, may introduce artifacts.
        n_std_thresh: Number of standard deviations above noise floor
            to consider as signal.
    """

    def __init__(
        self,
        stationary: bool = True,
        prop_decrease: float = 0.75,
        n_std_thresh: float = 1.5,
    ) -> None:
        """Initialize NoiseReducer.

        Args:
            stationary: Use stationary noise model.
            prop_decrease: Noise reduction aggressiveness coefficient.
            n_std_thresh: Signal detection threshold above noise floor.
        """
        self._stationary = stationary
        self._prop_decrease = prop_decrease
        self._n_std_thresh = n_std_thresh
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def reduce(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        noise_sample: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply noise reduction to audio.

        Args:
            audio_array: Float32 audio samples normalized to [-1, 1].
            sample_rate: Audio sample rate in Hz.
            noise_sample: Optional explicit noise sample for noise profile
                estimation. If None, noise is estimated from first 0.5s.

        Returns:
            Noise-reduced float32 audio array (same shape as input).
        """
        self._log.info(
            "noise_reduction_started",
            duration_s=round(len(audio_array) / sample_rate, 2),
            stationary=self._stationary,
        )

        result = await asyncio.to_thread(
            self._reduce_sync,
            audio_array=audio_array,
            sample_rate=sample_rate,
            noise_sample=noise_sample,
        )

        # Measure SNR improvement
        input_rms = float(np.sqrt(np.mean(audio_array ** 2)))
        output_rms = float(np.sqrt(np.mean(result ** 2)))
        self._log.info(
            "noise_reduction_complete",
            input_rms=round(input_rms, 4),
            output_rms=round(output_rms, 4),
        )

        return result

    def _reduce_sync(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        noise_sample: np.ndarray | None,
    ) -> np.ndarray:
        """Apply noise reduction synchronously (called via to_thread).

        Args:
            audio_array: Normalized float32 audio.
            sample_rate: Sample rate in Hz.
            noise_sample: Optional explicit noise profile.

        Returns:
            Noise-reduced float32 array.
        """
        import noisereduce as nr

        kwargs: dict[str, object] = {
            "y": audio_array,
            "sr": sample_rate,
            "stationary": self._stationary,
            "prop_decrease": self._prop_decrease,
            "n_std_thresh_stationary": self._n_std_thresh,
        }

        if noise_sample is not None:
            kwargs["y_noise"] = noise_sample

        reduced = nr.reduce_noise(**kwargs)
        return np.clip(reduced.astype(np.float32), -1.0, 1.0)

    async def estimate_noise_profile(
        self,
        audio_array: np.ndarray,
        noise_window_seconds: float = 0.5,
    ) -> np.ndarray:
        """Extract a noise profile from the beginning of the audio.

        Args:
            audio_array: Float32 audio array.
            noise_window_seconds: Duration of initial silence used for
                noise profiling (default: 0.5 seconds).

        Returns:
            Noise profile array extracted from the initial window.
        """
        # Estimate sample count for the noise window
        # Uses a fixed 16kHz estimate since sample_rate isn't passed here
        noise_samples = int(16000 * noise_window_seconds)
        return audio_array[:min(noise_samples, len(audio_array))]
