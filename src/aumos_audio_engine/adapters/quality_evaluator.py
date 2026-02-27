"""Audio synthesis quality evaluation adapter.

Implements AudioQualityEvaluatorProtocol. Computes objective quality metrics
for synthesised audio including MOS estimation (PESQ proxy), speaker similarity,
pitch contour comparison (F0 + DTW), prosody matching, SNR, and an aggregated
fidelity score.
"""

import asyncio
import io
from typing import Any

import librosa  # type: ignore[import-untyped]
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine  # type: ignore[import-untyped]

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)

# ── Quality metric weight coefficients ──────────────────────────────────────
# Weights must sum to 1.0
_METRIC_WEIGHTS: dict[str, float] = {
    "mos_estimate":         0.30,
    "speaker_similarity":   0.25,
    "pitch_contour_dtw":    0.20,
    "prosody_match":        0.15,
    "snr_comparison":       0.10,
}


class AudioQualityEvaluator:
    """Objective audio quality evaluator for synthesised speech.

    Computes a multi-dimensional quality report comparing a synthesised audio
    sample against a reference (ground truth or target). All signal processing
    is offloaded to the thread-pool executor for async compatibility.

    Metrics:
        - MOS estimate: PESQ-proxy using wideband MFCC distance-to-silence model.
        - Speaker similarity: Cosine distance between MFCC mean embeddings.
        - Pitch contour DTW: Dynamic-time-warped F0 curve distance.
        - Prosody match: Energy envelope and syllable rate comparison.
        - SNR comparison: Signal-to-noise ratio difference between reference and synthesis.
        - Aggregated fidelity score: Weighted combination of all metrics.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize quality evaluator with service settings.

        Args:
            settings: Audio engine settings (used for sample_rate baseline).
        """
        self._settings = settings

    async def evaluate(
        self,
        synthesised_audio: bytes,
        reference_audio: bytes,
        synthesised_format: str = "wav",
        reference_format: str = "wav",
    ) -> dict:
        """Compute quality metrics comparing synthesised audio against a reference.

        Args:
            synthesised_audio: Synthesised audio bytes to evaluate.
            reference_audio: Reference (ground-truth) audio bytes for comparison.
            synthesised_format: Container format of synthesised audio.
            reference_format: Container format of reference audio.

        Returns:
            Dict with keys:
                - 'mos_estimate': float [1.0–5.0] — MOS proxy score.
                - 'speaker_similarity': float [0.0–1.0] — Cosine similarity of MFCC embeddings.
                - 'pitch_contour_dtw': float [0.0–1.0] — Normalised DTW distance (higher = better).
                - 'prosody_match': float [0.0–1.0] — Energy envelope and rate similarity.
                - 'snr_comparison': float [0.0–1.0] — SNR ratio similarity.
                - 'fidelity_score': float [0.0–1.0] — Weighted aggregate of all metrics.
                - 'metric_details': dict — Raw numeric values before normalisation.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._evaluate_sync,
            synthesised_audio,
            reference_audio,
            synthesised_format,
            reference_format,
        )

        logger.info(
            "Audio quality evaluation complete",
            fidelity_score=round(result.get("fidelity_score", 0.0), 4),
            mos_estimate=round(result.get("mos_estimate", 0.0), 3),
            speaker_similarity=round(result.get("speaker_similarity", 0.0), 4),
        )

        return result

    def _evaluate_sync(
        self,
        synthesised_audio: bytes,
        reference_audio: bytes,
        synthesised_format: str,
        reference_format: str,
    ) -> dict:
        """Synchronous quality evaluation pipeline (runs in thread pool)."""
        synth_array, synth_sr = self._load_audio(synthesised_audio, synthesised_format)
        ref_array, ref_sr = self._load_audio(reference_audio, reference_format)

        # Resample synthesised to reference sample rate for fair comparison
        if synth_sr != ref_sr:
            synth_array = librosa.resample(synth_array, orig_sr=synth_sr, target_sr=ref_sr)
            synth_sr = ref_sr

        sr = ref_sr

        # ── Compute individual metrics ───────────────────────────────────────
        mos_estimate = self._estimate_mos(synth_array, ref_array, sr)
        speaker_similarity = self._compute_speaker_similarity(synth_array, ref_array, sr)
        pitch_dtw_score = self._compute_pitch_contour_dtw(synth_array, ref_array, sr)
        prosody_score = self._compute_prosody_match(synth_array, ref_array, sr)
        snr_score = self._compute_snr_comparison(synth_array, ref_array, sr)

        # ── Aggregate fidelity score ─────────────────────────────────────────
        component_scores = {
            "mos_estimate":       (mos_estimate - 1.0) / 4.0,  # Normalise 1–5 → 0–1
            "speaker_similarity":  speaker_similarity,
            "pitch_contour_dtw":   pitch_dtw_score,
            "prosody_match":       prosody_score,
            "snr_comparison":      snr_score,
        }

        fidelity_score = sum(
            component_scores[metric] * weight
            for metric, weight in _METRIC_WEIGHTS.items()
        )

        metric_details = {
            "mos_raw":                     round(mos_estimate, 3),
            "speaker_similarity_raw":      round(speaker_similarity, 4),
            "pitch_dtw_normalised":        round(pitch_dtw_score, 4),
            "prosody_match_raw":           round(prosody_score, 4),
            "snr_score_raw":               round(snr_score, 4),
            "synth_duration_seconds":      round(len(synth_array) / sr, 3),
            "ref_duration_seconds":        round(len(ref_array) / sr, 3),
            "sample_rate_hz":              sr,
        }

        return {
            "mos_estimate":       round(mos_estimate, 3),
            "speaker_similarity": round(speaker_similarity, 4),
            "pitch_contour_dtw":  round(pitch_dtw_score, 4),
            "prosody_match":      round(prosody_score, 4),
            "snr_comparison":     round(snr_score, 4),
            "fidelity_score":     round(float(np.clip(fidelity_score, 0.0, 1.0)), 4),
            "metric_details":     metric_details,
        }

    async def evaluate_standalone(self, audio_bytes: bytes, audio_format: str = "wav") -> dict:
        """Compute standalone quality metrics without a reference audio.

        Useful for estimating absolute quality when no reference is available.
        Returns signal-level metrics only (SNR, spectral centroid, energy variance).

        Args:
            audio_bytes: Audio bytes to evaluate.
            audio_format: Container format.

        Returns:
            Dict with standalone quality metrics.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._evaluate_standalone_sync,
            audio_bytes,
            audio_format,
        )

    def _evaluate_standalone_sync(self, audio_bytes: bytes, audio_format: str) -> dict:
        """Synchronous standalone evaluation."""
        audio_array, sample_rate = self._load_audio(audio_bytes, audio_format)

        snr_db = self._estimate_snr(audio_array, sample_rate)
        spectral_centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate))
        )
        rms_energy = float(np.sqrt(np.mean(audio_array ** 2)))
        duration_seconds = len(audio_array) / sample_rate

        # Simple articulation proxy: ratio of voiced to total frames
        zero_crossings = librosa.feature.zero_crossing_rate(audio_array)
        voiced_ratio = float(np.mean(zero_crossings < 0.15))

        return {
            "snr_db":              round(snr_db, 2),
            "spectral_centroid_hz": round(spectral_centroid, 1),
            "rms_energy":           round(rms_energy, 6),
            "voiced_ratio":         round(voiced_ratio, 4),
            "duration_seconds":     round(duration_seconds, 3),
            "sample_rate_hz":       sample_rate,
        }

    # ── Private signal processing methods ────────────────────────────────────

    def _estimate_mos(
        self,
        synth: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate MOS (Mean Opinion Score) via MFCC distance proxy.

        PESQ/POLQA are patented and require native libraries. This proxy uses
        the MFCC spectral distance between the two signals, mapped to the 1–5 MOS scale.
        A distance of 0 → MOS 5.0; distance ≥ 30 → MOS 1.0.

        Args:
            synth: Synthesised audio float32 array.
            reference: Reference audio float32 array.
            sample_rate: Common sample rate.

        Returns:
            Estimated MOS score in range [1.0, 5.0].
        """
        try:
            n_mfcc = 13
            synth_mfcc = librosa.feature.mfcc(y=synth, sr=sample_rate, n_mfcc=n_mfcc)
            ref_mfcc = librosa.feature.mfcc(y=reference, sr=sample_rate, n_mfcc=n_mfcc)

            # Align lengths by truncating to shorter one
            min_frames = min(synth_mfcc.shape[1], ref_mfcc.shape[1])
            synth_mfcc = synth_mfcc[:, :min_frames]
            ref_mfcc = ref_mfcc[:, :min_frames]

            # Frame-wise RMSE across MFCC coefficients
            rmse = float(np.sqrt(np.mean((synth_mfcc - ref_mfcc) ** 2)))

            # Linear mapping: distance 0 → 5.0, distance 30 → 1.0
            mos = max(1.0, min(5.0, 5.0 - (rmse / 30.0) * 4.0))
            return mos

        except Exception as exc:
            logger.warning("MOS estimation failed, returning mid-range", error=str(exc))
            return 3.0

    def _compute_speaker_similarity(
        self,
        synth: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute speaker similarity using MFCC mean embedding cosine similarity.

        Args:
            synth: Synthesised audio array.
            reference: Reference audio array.
            sample_rate: Sample rate.

        Returns:
            Cosine similarity score [0.0, 1.0].
        """
        try:
            synth_mfcc = librosa.feature.mfcc(y=synth, sr=sample_rate, n_mfcc=20).mean(axis=1)
            ref_mfcc = librosa.feature.mfcc(y=reference, sr=sample_rate, n_mfcc=20).mean(axis=1)

            similarity = 1.0 - cosine(synth_mfcc, ref_mfcc)
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception as exc:
            logger.warning("Speaker similarity computation failed", error=str(exc))
            return 0.5

    def _compute_pitch_contour_dtw(
        self,
        synth: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compare F0 pitch contours using Dynamic Time Warping.

        Extracts fundamental frequency (F0) using librosa PYIN, then computes
        DTW alignment cost normalised by sequence length.

        Args:
            synth: Synthesised audio array.
            reference: Reference audio array.
            sample_rate: Sample rate.

        Returns:
            Similarity score [0.0, 1.0] where 1.0 = identical pitch contour.
        """
        try:
            f0_synth, voiced_flag_synth, _ = librosa.pyin(
                synth, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            f0_ref, voiced_flag_ref, _ = librosa.pyin(
                reference, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )

            # Extract only voiced frames
            synth_voiced = f0_synth[voiced_flag_synth].astype(np.float32)
            ref_voiced = f0_ref[voiced_flag_ref].astype(np.float32)

            if len(synth_voiced) == 0 or len(ref_voiced) == 0:
                return 0.5  # Insufficient pitch data

            # DTW alignment — use librosa's DTW
            dtw_matrix, _ = librosa.sequence.dtw(
                synth_voiced.reshape(1, -1),
                ref_voiced.reshape(1, -1),
                metric="euclidean",
            )
            dtw_cost = float(dtw_matrix[-1, -1])

            # Normalise by the geometric mean of sequence lengths
            normalised_cost = dtw_cost / max(
                1.0, np.sqrt(len(synth_voiced) * len(ref_voiced))
            )

            # Map to similarity: cost 0 → 1.0, cost ≥ 200 → 0.0
            similarity = max(0.0, 1.0 - normalised_cost / 200.0)
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as exc:
            logger.warning("Pitch contour DTW computation failed", error=str(exc))
            return 0.5

    def _compute_prosody_match(
        self,
        synth: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute prosody similarity via energy envelope and syllable rate comparison.

        Args:
            synth: Synthesised audio array.
            reference: Reference audio array.
            sample_rate: Sample rate.

        Returns:
            Prosody match score [0.0, 1.0].
        """
        try:
            frame_length = 2048
            hop_length = 512

            # Energy envelopes via RMS
            synth_rms = librosa.feature.rms(y=synth, frame_length=frame_length, hop_length=hop_length)[0]
            ref_rms = librosa.feature.rms(y=reference, frame_length=frame_length, hop_length=hop_length)[0]

            # Normalise envelopes
            synth_rms_norm = synth_rms / (np.max(synth_rms) + 1e-8)
            ref_rms_norm = ref_rms / (np.max(ref_rms) + 1e-8)

            # Align by truncating to shorter
            min_len = min(len(synth_rms_norm), len(ref_rms_norm))
            synth_rms_norm = synth_rms_norm[:min_len]
            ref_rms_norm = ref_rms_norm[:min_len]

            energy_correlation = float(np.corrcoef(synth_rms_norm, ref_rms_norm)[0, 1])
            energy_score = (energy_correlation + 1.0) / 2.0  # Map [-1, 1] → [0, 1]

            # Syllable rate proxy: zero-crossing rate
            synth_zcr = float(np.mean(librosa.feature.zero_crossing_rate(synth)))
            ref_zcr = float(np.mean(librosa.feature.zero_crossing_rate(reference)))

            zcr_ratio = min(synth_zcr, ref_zcr) / (max(synth_zcr, ref_zcr) + 1e-8)
            rate_score = float(zcr_ratio)

            return float(np.clip((energy_score * 0.7 + rate_score * 0.3), 0.0, 1.0))

        except Exception as exc:
            logger.warning("Prosody match computation failed", error=str(exc))
            return 0.5

    def _compute_snr_comparison(
        self,
        synth: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compare SNR between synthesised and reference audio.

        Computes SNR for each signal and returns a similarity score based on
        how close the two SNR values are.

        Args:
            synth: Synthesised audio array.
            reference: Reference audio array.
            sample_rate: Sample rate.

        Returns:
            SNR similarity score [0.0, 1.0].
        """
        try:
            synth_snr = self._estimate_snr(synth, sample_rate)
            ref_snr = self._estimate_snr(reference, sample_rate)

            # Similarity: 1.0 when equal, decays with absolute difference
            snr_diff = abs(synth_snr - ref_snr)
            similarity = max(0.0, 1.0 - snr_diff / 40.0)  # 40 dB range
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as exc:
            logger.warning("SNR comparison failed", error=str(exc))
            return 0.5

    @staticmethod
    def _estimate_snr(audio: np.ndarray, sample_rate: int) -> float:
        """Estimate signal-to-noise ratio using a simple spectral subtraction approach.

        Uses the minimum spectral magnitude across frames as the noise floor estimate.

        Args:
            audio: Audio float32 array.
            sample_rate: Sample rate.

        Returns:
            Estimated SNR in decibels.
        """
        stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        noise_floor = np.min(stft, axis=1, keepdims=True)
        signal_power = np.mean(stft ** 2)
        noise_power = np.mean(noise_floor ** 2)

        if noise_power < 1e-10:
            return 60.0  # Effectively infinite SNR

        snr_db = 10.0 * np.log10(signal_power / noise_power)
        return float(np.clip(snr_db, -20.0, 80.0))

    @staticmethod
    def _load_audio(audio_bytes: bytes, audio_format: str) -> tuple[np.ndarray, int]:
        """Load audio bytes into a float32 mono numpy array.

        Args:
            audio_bytes: Raw audio bytes.
            audio_format: Container format (for error context only; soundfile auto-detects).

        Returns:
            Tuple of (float32 mono audio array, sample_rate).
        """
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer, dtype="float32")

        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        return audio_data.astype(np.float32), int(sample_rate)

    async def health_check(self) -> bool:
        """Return True if the quality evaluator is operational."""
        try:
            import librosa as _  # noqa: F401
            import soundfile as _sf  # noqa: F401
            return True
        except ImportError:
            return False
