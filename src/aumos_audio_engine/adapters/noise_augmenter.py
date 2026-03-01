"""Noise augmentation adapter — GAP-85/87 competitive gap implementation.

Augments synthetic audio with realistic background noise profiles for
training data diversity. Also provides noise-robust de-identification
preprocessing using spectral subtraction (GAP-87).

Noise profiles: clean, office (SNR 20dB), street (SNR 10dB), reverberant room.
"""

from __future__ import annotations

import asyncio
import io
from enum import Enum
from typing import Any

import numpy as np
import structlog
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class NoiseProfile(str, Enum):
    """Standard noise profiles for audio augmentation certification."""

    CLEAN = "clean"
    OFFICE = "office"  # SNR ~20dB — keyboard, HVAC, ambient voices
    STREET = "street"  # SNR ~10dB — traffic, footsteps, wind
    REVERBERANT = "reverberant"  # Room reverberation, echoes
    CALL_CENTER = "call_center"  # Phone compression + background chatter
    FACTORY = "factory"  # Industrial machinery noise


# Target SNR dB for each profile
PROFILE_SNR_DB: dict[NoiseProfile, float] = {
    NoiseProfile.CLEAN: 40.0,
    NoiseProfile.OFFICE: 20.0,
    NoiseProfile.STREET: 10.0,
    NoiseProfile.REVERBERANT: 15.0,
    NoiseProfile.CALL_CENTER: 18.0,
    NoiseProfile.FACTORY: 5.0,
}


class NoiseAugmenter:
    """Adds realistic noise profiles to synthetic audio for training diversity.

    Implements additive white Gaussian noise (AWGN), recorded noise mixing,
    and room impulse response (RIR) convolution for reverb simulation.
    Also provides noise-robust preprocessing (spectral subtraction) for
    de-identification in noisy conditions.

    Args:
        sample_rate: Target sample rate in Hz (default: 16000).
        seed: Random seed for reproducibility (None for non-deterministic).
        noise_cache_dir: Directory for noise sample cache files.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        seed: int | None = None,
        noise_cache_dir: str = "/tmp/noise-cache",
    ) -> None:
        self._sample_rate = sample_rate
        self._rng = np.random.default_rng(seed)
        self._noise_cache_dir = noise_cache_dir
        self._log = logger.bind(adapter="noise_augmenter")

    async def augment(
        self,
        audio_bytes: bytes,
        noise_profile: NoiseProfile | str,
        snr_db: float | None = None,
        input_format: str = "wav",
        output_format: str = "wav",
    ) -> bytes:
        """Apply noise augmentation to audio.

        Adds noise matching the specified profile at the target SNR.
        If snr_db is provided it overrides the profile default.

        Args:
            audio_bytes: Input audio bytes.
            noise_profile: Noise profile to apply.
            snr_db: Target signal-to-noise ratio in dB. Uses profile default if None.
            input_format: Input audio format ("wav", "mp3", etc.).
            output_format: Output audio format.

        Returns:
            Augmented audio bytes with noise applied.
        """
        profile = NoiseProfile(noise_profile) if isinstance(noise_profile, str) else noise_profile
        target_snr = snr_db if snr_db is not None else PROFILE_SNR_DB[profile]

        self._log.info(
            "noise_augmenter.augment",
            profile=profile.value,
            target_snr_db=target_snr,
        )

        return await asyncio.to_thread(
            self._augment_sync,
            audio_bytes,
            profile,
            target_snr,
            input_format,
            output_format,
        )

    def _augment_sync(
        self,
        audio_bytes: bytes,
        profile: NoiseProfile,
        snr_db: float,
        input_format: str,
        output_format: str,
    ) -> bytes:
        """Synchronous noise augmentation — called from thread pool."""
        import librosa  # type: ignore[import]
        import soundfile as sf  # type: ignore[import]

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self._sample_rate, mono=True)

        if profile == NoiseProfile.CLEAN:
            # Minimal noise — just add negligible AWGN
            noise = self._generate_awgn(len(audio), snr_db=40.0)
            augmented = audio + noise
        elif profile in (NoiseProfile.OFFICE, NoiseProfile.STREET, NoiseProfile.CALL_CENTER, NoiseProfile.FACTORY):
            noise = self._generate_profile_noise(len(audio), profile)
            augmented = self._mix_at_snr(audio, noise, snr_db)
        elif profile == NoiseProfile.REVERBERANT:
            augmented = self._apply_reverb(audio)
            # Also add some noise at the target SNR
            noise = self._generate_awgn(len(augmented), snr_db=snr_db + 5.0)
            augmented = augmented + noise
        else:
            noise = self._generate_awgn(len(audio), snr_db)
            augmented = audio + noise

        # Normalize to prevent clipping
        max_val = np.abs(augmented).max()
        if max_val > 0:
            augmented = augmented / max_val * 0.95

        buf = io.BytesIO()
        sf.write(buf, augmented, samplerate=self._sample_rate, format=output_format.upper())
        return buf.getvalue()

    def _generate_awgn(self, num_samples: int, snr_db: float) -> np.ndarray:
        """Generate additive white Gaussian noise at the specified SNR."""
        # For AWGN at a given SNR relative to unit-amplitude signal
        noise_std = 10.0 ** (-snr_db / 20.0)
        return self._rng.normal(0, noise_std, num_samples).astype(np.float32)

    def _generate_profile_noise(self, num_samples: int, profile: NoiseProfile) -> np.ndarray:
        """Generate colored noise matching a specific acoustic environment.

        Office noise: pink-ish spectrum (f^-1 roll-off).
        Street noise: broadband with low-frequency bias.
        Call center: band-limited (telephone bandwidth 300-3400 Hz).
        Factory: low-frequency dominant with periodic components.
        """
        frequencies = np.fft.rfftfreq(num_samples, d=1.0 / self._sample_rate)
        frequencies[0] = 1.0  # Avoid division by zero at DC

        if profile == NoiseProfile.OFFICE:
            # Pink noise (1/f spectrum)
            spectrum = 1.0 / np.sqrt(frequencies)
        elif profile == NoiseProfile.STREET:
            # Brown noise (1/f^2 spectrum) for traffic rumble
            spectrum = 1.0 / frequencies
        elif profile == NoiseProfile.CALL_CENTER:
            # Band-limited noise 300–3400 Hz (telephone bandwidth)
            spectrum = np.ones(len(frequencies))
            spectrum[frequencies < 300] = 0.01
            spectrum[frequencies > 3400] = 0.01
        elif profile == NoiseProfile.FACTORY:
            # Low-frequency dominant with 50/60 Hz harmonics
            spectrum = 1.0 / frequencies
            for harmonic in [50, 100, 150, 60, 120, 180]:
                idx = np.argmin(np.abs(frequencies - harmonic))
                if idx < len(spectrum):
                    spectrum[max(0, idx - 1) : idx + 2] *= 5.0
        else:
            spectrum = np.ones(len(frequencies))

        # Generate random phase
        phases = self._rng.uniform(0, 2 * np.pi, len(frequencies))
        complex_spectrum = spectrum * np.exp(1j * phases)
        noise = np.fft.irfft(complex_spectrum, n=num_samples).astype(np.float32)
        return noise

    def _mix_at_snr(
        self,
        signal: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        """Mix signal and noise at the target SNR in dB."""
        signal_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
        noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9

        desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise * (desired_noise_rms / noise_rms)

        if len(scaled_noise) < len(signal):
            repeats = (len(signal) // len(scaled_noise)) + 1
            scaled_noise = np.tile(scaled_noise, repeats)
        return signal + scaled_noise[: len(signal)]

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply synthetic room reverberation via exponential decay convolution.

        Generates a simple exponentially-decaying impulse response to simulate
        a small-to-medium reverberant room (RT60 ~0.3s).
        """
        rt60 = 0.3  # seconds
        ir_length = int(rt60 * self._sample_rate)
        decay = np.exp(-3.0 * np.linspace(0, rt60, ir_length))
        ir = self._rng.normal(0, 1, ir_length).astype(np.float32) * decay.astype(np.float32)
        ir = ir / (np.abs(ir).max() + 1e-8)

        reverbed = np.convolve(audio, ir, mode="full")[: len(audio)]
        return reverbed.astype(np.float32)

    async def denoise_spectral_subtraction(
        self,
        audio_bytes: bytes,
        noise_estimate_frames: int = 10,
        oversubtraction_factor: float = 1.2,
        input_format: str = "wav",
        output_format: str = "wav",
    ) -> bytes:
        """Apply spectral subtraction denoising as preprocessing for de-identification.

        Estimates noise spectrum from the first N frames (assumed noise-only),
        then subtracts the estimate from all frames to reduce background noise.
        Improves de-identification accuracy in noisy conditions (GAP-87).

        Args:
            audio_bytes: Noisy input audio bytes.
            noise_estimate_frames: Number of initial frames used for noise estimation.
            oversubtraction_factor: Alpha parameter controlling aggressiveness (1.0–2.0).
            input_format: Input audio format.
            output_format: Output audio format.

        Returns:
            Denoised audio bytes suitable for de-identification processing.
        """
        self._log.info("noise_augmenter.spectral_subtraction")
        return await asyncio.to_thread(
            self._denoise_spectral_sync,
            audio_bytes,
            noise_estimate_frames,
            oversubtraction_factor,
            input_format,
            output_format,
        )

    def _denoise_spectral_sync(
        self,
        audio_bytes: bytes,
        noise_frames: int,
        alpha: float,
        input_format: str,
        output_format: str,
    ) -> bytes:
        """Synchronous spectral subtraction — called from thread pool."""
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self._sample_rate, mono=True)

        frame_length = 512
        hop_length = 128

        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise from initial frames
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Spectral subtraction with over-subtraction factor
        enhanced_magnitude = magnitude - alpha * noise_estimate
        # Half-wave rectification (ensure non-negative)
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)

        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        denoised = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))

        buf = io.BytesIO()
        sf.write(buf, denoised.astype(np.float32), samplerate=self._sample_rate, format=output_format.upper())
        return buf.getvalue()

    def get_available_profiles(self) -> list[dict[str, Any]]:
        """Return metadata for all available noise profiles.

        Returns:
            List of dicts with profile name, description, and default SNR.
        """
        descriptions = {
            NoiseProfile.CLEAN: "Minimal noise (SNR ~40dB) — ideal baseline",
            NoiseProfile.OFFICE: "Office environment noise — HVAC, keyboard, ambient voices (SNR ~20dB)",
            NoiseProfile.STREET: "Street/outdoor noise — traffic, footsteps, wind (SNR ~10dB)",
            NoiseProfile.REVERBERANT: "Reverberant room — echoes, reflection (SNR ~15dB)",
            NoiseProfile.CALL_CENTER: "Call center noise — phone compression, background chatter (SNR ~18dB)",
            NoiseProfile.FACTORY: "Industrial factory — machinery, low-frequency dominant (SNR ~5dB)",
        }
        return [
            {
                "profile": profile.value,
                "description": descriptions[profile],
                "default_snr_db": PROFILE_SNR_DB[profile],
            }
            for profile in NoiseProfile
        ]
