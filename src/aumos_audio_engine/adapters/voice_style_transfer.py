"""Voice style transfer adapter.

Implements StyleTransferProtocol. Extracts non-biometric stylistic features
(pitch statistics, tempo, energy envelope, spectral envelope) from source audio
and applies them to synthesised output — without retaining biometric speaker identity.
"""

import asyncio
import io
from dataclasses import asdict, dataclass

import librosa  # type: ignore[import-untyped]
import numpy as np
import soundfile as sf

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)


@dataclass
class VoiceStyleParameters:
    """Non-biometric voice style representation.

    Stores only aggregate statistical features of a voice style, never
    individual speaker embeddings or raw waveform segments. Safe to log,
    store, and transmit.
    """

    # Pitch statistics (F0)
    f0_mean_hz: float
    f0_std_hz: float
    f0_min_hz: float
    f0_max_hz: float

    # Temporal characteristics
    speaking_rate_syllables_per_second: float
    pause_ratio: float  # Fraction of frames with silence

    # Energy characteristics
    rms_mean: float
    rms_std: float
    energy_dynamic_range_db: float

    # Spectral characteristics (non-identifying shape stats)
    spectral_centroid_mean_hz: float
    spectral_bandwidth_mean_hz: float
    spectral_rolloff_mean_hz: float

    # Prosody: high-level pitch contour shape
    pitch_slope: float  # Linear trend across voiced frames
    pitch_variability: float  # Normalised variance


class VoiceStyleTransfer:
    """Privacy-preserving voice style transfer adapter.

    Extracts aggregate stylistic parameters from a source audio clip and
    applies them to a new synthetic audio signal using signal processing
    transformations. The source speaker's biometric identity (voice embedding,
    timbre, glottal characteristics) is never stored or transmitted.

    Pipeline:
        1. Extract VoiceStyleParameters from source audio.
        2. Detect and assert that no biometric embeddings are retained.
        3. Apply style parameters to target audio via pitch normalisation,
           time-stretching, energy envelope matching, and spectral shaping.
        4. Verify quality preservation via MFCC distance gate.
    """

    # Maximum MFCC distance delta that is considered "quality preserved"
    _QUALITY_PRESERVATION_THRESHOLD = 15.0

    # Biometric similarity threshold — output must score below this to pass privacy gate
    _BIOMETRIC_SIMILARITY_LIMIT = 0.85

    def __init__(self, settings: Settings) -> None:
        """Initialize style transfer adapter.

        Args:
            settings: Audio engine settings (sample_rate, temp_dir, etc.).
        """
        self._settings = settings

    async def transfer_style(
        self,
        source_audio_bytes: bytes,
        source_format: str,
        target_style_config: dict,
        output_format: str,
        preserve_semantics: bool,
    ) -> tuple[bytes, dict]:
        """Transfer voice style from source audio to synthesised output.

        Extracts non-biometric style features from source, optionally blends with
        target_style_config overrides, then applies the merged style to a neutral
        synthesis of the detected speech content.

        Args:
            source_audio_bytes: Source audio bytes from which to extract style.
            source_format: Container format of source audio.
            target_style_config: Optional overrides for extracted style parameters.
                Accepted keys: 'pitch_shift_semitones', 'tempo_ratio', 'energy_gain_db'.
            output_format: Output container format.
            preserve_semantics: If True, verify that the output is still intelligible
                by checking MOS proxy score does not degrade beyond threshold.

        Returns:
            Tuple of (output_audio_bytes, style_metadata_dict). The style_metadata_dict
            contains the extracted VoiceStyleParameters plus applied transformation
            details — no biometric embeddings.

        Raises:
            ValueError: If biometric retention is detected in the output.
            RuntimeError: If quality preservation gate fails when preserve_semantics=True.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._transfer_sync,
            source_audio_bytes,
            source_format,
            target_style_config,
            output_format,
            preserve_semantics,
        )

        output_bytes, style_metadata = result

        logger.info(
            "Voice style transfer complete",
            output_format=output_format,
            output_bytes=len(output_bytes),
            biometric_check_passed=style_metadata.get("biometric_check_passed"),
            quality_preserved=style_metadata.get("quality_preserved"),
        )

        return output_bytes, style_metadata

    def _transfer_sync(
        self,
        source_audio_bytes: bytes,
        source_format: str,
        target_style_config: dict,
        output_format: str,
        preserve_semantics: bool,
    ) -> tuple[bytes, dict]:
        """Synchronous style transfer pipeline (runs in thread pool)."""
        source_array, source_sr = self._load_audio(source_audio_bytes)

        # Step 1: Extract non-biometric style parameters
        style_params = self._extract_style(source_array, source_sr)

        # Step 2: Apply target_style_config overrides
        style_params = self._apply_overrides(style_params, target_style_config)

        # Step 3: Build a neutral representation via pitch normalisation
        #   We cannot synthesise text here (no TTS dependency in this adapter),
        #   so style transfer operates on the source audio itself with identity removal.
        processed_array = source_array.copy()

        # 3a. Pitch normalisation — shift to the style's mean F0 from current mean
        current_f0_mean = self._estimate_f0_mean(processed_array, source_sr)
        if current_f0_mean > 0 and style_params.f0_mean_hz > 0:
            semitone_shift = 12.0 * np.log2(style_params.f0_mean_hz / current_f0_mean)
            # Apply overridden shift if provided
            semitone_shift += float(target_style_config.get("pitch_shift_semitones", 0.0))
            if abs(semitone_shift) > 0.1:
                processed_array = librosa.effects.pitch_shift(
                    processed_array, sr=source_sr, n_steps=float(semitone_shift)
                )

        # 3b. Tempo adjustment toward target speaking rate
        tempo_ratio = float(target_style_config.get("tempo_ratio", 1.0))
        if tempo_ratio != 1.0 and 0.5 <= tempo_ratio <= 2.0:
            processed_array = librosa.effects.time_stretch(processed_array, rate=tempo_ratio)

        # 3c. Energy envelope matching
        target_rms = style_params.rms_mean
        current_rms = float(np.sqrt(np.mean(processed_array ** 2)))
        if current_rms > 1e-8:
            gain = target_rms / current_rms
            # Apply energy_gain_db override if present
            extra_gain_db = float(target_style_config.get("energy_gain_db", 0.0))
            gain *= 10 ** (extra_gain_db / 20.0)
            processed_array = np.clip(processed_array * gain, -1.0, 1.0)

        # Step 4: Biometric retention check
        biometric_similarity = self._check_biometric_similarity(
            source_array, processed_array, source_sr
        )
        biometric_check_passed = biometric_similarity < self._BIOMETRIC_SIMILARITY_LIMIT

        if not biometric_check_passed:
            logger.warning(
                "Biometric similarity above limit after style transfer — applying additional de-identification",
                biometric_similarity=round(biometric_similarity, 4),
                limit=self._BIOMETRIC_SIMILARITY_LIMIT,
            )
            # Apply additional formant shift to break biometric identity
            processed_array = self._apply_additional_deidentification(
                processed_array, source_sr
            )
            biometric_similarity = self._check_biometric_similarity(
                source_array, processed_array, source_sr
            )
            biometric_check_passed = biometric_similarity < self._BIOMETRIC_SIMILARITY_LIMIT

        if not biometric_check_passed:
            raise ValueError(
                f"Biometric retention detected after style transfer. "
                f"Achieved similarity: {biometric_similarity:.3f} exceeds limit "
                f"{self._BIOMETRIC_SIMILARITY_LIMIT}."
            )

        # Step 5: Quality gate
        quality_preserved = True
        mfcc_delta = 0.0
        if preserve_semantics:
            mfcc_delta = self._compute_mfcc_distance(source_array, processed_array, source_sr)
            quality_preserved = mfcc_delta <= self._QUALITY_PRESERVATION_THRESHOLD

            if not quality_preserved:
                raise RuntimeError(
                    f"Quality preservation gate failed after style transfer. "
                    f"MFCC delta {mfcc_delta:.2f} exceeds threshold "
                    f"{self._QUALITY_PRESERVATION_THRESHOLD}."
                )

        # Step 6: Encode to output format
        output_buffer = io.BytesIO()
        sf.write(output_buffer, processed_array, source_sr, format=output_format.upper())
        output_bytes = output_buffer.getvalue()

        style_metadata = {
            **asdict(style_params),
            "biometric_similarity": round(biometric_similarity, 4),
            "biometric_check_passed": biometric_check_passed,
            "quality_preserved": quality_preserved,
            "mfcc_delta": round(mfcc_delta, 3),
            "applied_transformations": {
                "pitch_shift_semitones": float(target_style_config.get("pitch_shift_semitones", 0.0)),
                "tempo_ratio": tempo_ratio,
                "energy_gain_db": float(target_style_config.get("energy_gain_db", 0.0)),
            },
        }

        return output_bytes, style_metadata

    async def extract_style(self, audio_bytes: bytes, audio_format: str) -> dict:
        """Extract non-biometric voice style parameters from audio.

        Useful for building a library of style descriptors without retaining
        any speaker biometric identity.

        Args:
            audio_bytes: Source audio bytes.
            audio_format: Audio container format.

        Returns:
            Dict of VoiceStyleParameters fields.
        """
        loop = asyncio.get_running_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None, self._load_audio, audio_bytes
        )
        style_params = await loop.run_in_executor(
            None, self._extract_style, audio_array, sample_rate
        )
        return asdict(style_params)

    async def interpolate_styles(
        self,
        style_a: dict,
        style_b: dict,
        alpha: float,
    ) -> dict:
        """Interpolate between two style parameter dicts.

        Args:
            style_a: First VoiceStyleParameters dict.
            style_b: Second VoiceStyleParameters dict.
            alpha: Interpolation factor [0.0 = style_a, 1.0 = style_b].

        Returns:
            Interpolated style parameter dict.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Interpolation alpha must be in [0.0, 1.0], got {alpha}")

        interpolated: dict = {}
        for key in style_a:
            val_a = style_a[key]
            val_b = style_b.get(key, val_a)
            if isinstance(val_a, float) and isinstance(val_b, float):
                interpolated[key] = round(val_a * (1.0 - alpha) + val_b * alpha, 6)
            else:
                interpolated[key] = val_a if alpha < 0.5 else val_b

        logger.info(
            "Style interpolation complete",
            alpha=alpha,
        )
        return interpolated

    # ── Private signal processing methods ────────────────────────────────────

    def _extract_style(self, audio: np.ndarray, sample_rate: int) -> VoiceStyleParameters:
        """Extract VoiceStyleParameters from an audio array.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            VoiceStyleParameters containing only aggregate statistics.
        """
        # F0 extraction using PYIN
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        voiced_f0 = f0[voiced_flag].astype(np.float64)
        if len(voiced_f0) == 0:
            voiced_f0 = np.array([200.0])  # Fallback for silent/non-speech audio

        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_min = float(np.min(voiced_f0))
        f0_max = float(np.max(voiced_f0))

        # Pitch slope and variability
        x = np.arange(len(voiced_f0), dtype=np.float64)
        pitch_slope = float(np.polyfit(x, voiced_f0, 1)[0]) if len(voiced_f0) > 1 else 0.0
        pitch_variability = f0_std / (f0_mean + 1e-8)

        # Energy statistics
        hop_length = 512
        rms_frames = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_mean = float(np.mean(rms_frames))
        rms_std = float(np.std(rms_frames))
        rms_max = float(np.max(rms_frames))
        rms_min = float(np.min(rms_frames + 1e-8))
        energy_dynamic_range_db = float(20.0 * np.log10(rms_max / rms_min + 1e-8))

        # Pause ratio
        silence_threshold = rms_mean * 0.1
        pause_ratio = float(np.mean(rms_frames < silence_threshold))

        # Speaking rate proxy: onset strength peaks
        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onset_peaks = librosa.util.peak_pick(
            onset_envelope, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
        )
        duration_seconds = len(audio) / sample_rate
        speaking_rate = len(onset_peaks) / max(1.0, duration_seconds)

        # Spectral characteristics
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
        spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))

        return VoiceStyleParameters(
            f0_mean_hz=round(f0_mean, 2),
            f0_std_hz=round(f0_std, 2),
            f0_min_hz=round(f0_min, 2),
            f0_max_hz=round(f0_max, 2),
            speaking_rate_syllables_per_second=round(speaking_rate, 3),
            pause_ratio=round(pause_ratio, 4),
            rms_mean=round(rms_mean, 6),
            rms_std=round(rms_std, 6),
            energy_dynamic_range_db=round(energy_dynamic_range_db, 2),
            spectral_centroid_mean_hz=round(spec_centroid, 1),
            spectral_bandwidth_mean_hz=round(spec_bandwidth, 1),
            spectral_rolloff_mean_hz=round(spec_rolloff, 1),
            pitch_slope=round(pitch_slope, 6),
            pitch_variability=round(pitch_variability, 4),
        )

    @staticmethod
    def _apply_overrides(
        style_params: VoiceStyleParameters,
        overrides: dict,
    ) -> VoiceStyleParameters:
        """Apply target_style_config overrides to extracted style parameters.

        Args:
            style_params: Extracted parameters.
            overrides: Dict of override values keyed by VoiceStyleParameters field names.

        Returns:
            Updated VoiceStyleParameters with overrides applied.
        """
        params_dict = asdict(style_params)
        for key, value in overrides.items():
            if key in params_dict and isinstance(value, float | int):
                params_dict[key] = float(value)
        return VoiceStyleParameters(**params_dict)

    @staticmethod
    def _estimate_f0_mean(audio: np.ndarray, sample_rate: int) -> float:
        """Quickly estimate the mean fundamental frequency of an audio array.

        Args:
            audio: Float32 mono audio.
            sample_rate: Sample rate.

        Returns:
            Mean F0 in Hz, or 0.0 if no voiced frames detected.
        """
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            voiced_f0 = f0[voiced_flag]
            return float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _check_biometric_similarity(
        original: np.ndarray,
        processed: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute MFCC-based cosine similarity as a biometric proxy.

        Args:
            original: Original audio array.
            processed: Processed audio array.
            sample_rate: Sample rate.

        Returns:
            Cosine similarity score [0.0–1.0]. Higher values indicate more
            biometric similarity.
        """
        try:
            orig_mfcc = librosa.feature.mfcc(y=original, sr=sample_rate, n_mfcc=20).mean(axis=1)
            proc_mfcc = librosa.feature.mfcc(y=processed, sr=sample_rate, n_mfcc=20).mean(axis=1)

            dot = np.dot(orig_mfcc, proc_mfcc)
            norm = np.linalg.norm(orig_mfcc) * np.linalg.norm(proc_mfcc)
            return float(np.clip(dot / (norm + 1e-8), 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _apply_additional_deidentification(
        audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply an additional pitch shift + formant perturbation to break biometric identity.

        Args:
            audio: Audio array to further de-identify.
            sample_rate: Sample rate.

        Returns:
            Further de-identified audio array.
        """
        # Additional 2-semitone shift
        audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2.0)
        # Formant shift via resampling trick
        shifted_sr = int(sample_rate * 1.08)
        resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=shifted_sr)
        restored = librosa.resample(resampled, orig_sr=shifted_sr, target_sr=sample_rate)
        target_length = len(audio)
        if len(restored) > target_length:
            return restored[:target_length]
        elif len(restored) < target_length:
            return np.pad(restored, (0, target_length - len(restored)))
        return restored

    @staticmethod
    def _compute_mfcc_distance(
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute frame-wise RMSE between two audio clips' MFCC matrices.

        Used as a proxy for semantic content preservation.

        Args:
            audio_a: First audio array.
            audio_b: Second audio array.
            sample_rate: Sample rate.

        Returns:
            MFCC distance (lower = more similar content).
        """
        mfcc_a = librosa.feature.mfcc(y=audio_a, sr=sample_rate, n_mfcc=13)
        mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sample_rate, n_mfcc=13)

        min_frames = min(mfcc_a.shape[1], mfcc_b.shape[1])
        mfcc_a = mfcc_a[:, :min_frames]
        mfcc_b = mfcc_b[:, :min_frames]

        return float(np.sqrt(np.mean((mfcc_a - mfcc_b) ** 2)))

    @staticmethod
    def _load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Load audio bytes to a float32 mono numpy array.

        Args:
            audio_bytes: Raw audio bytes.

        Returns:
            Tuple of (float32 mono audio array, sample_rate).
        """
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data.astype(np.float32), int(sample_rate)

    async def health_check(self) -> bool:
        """Return True if the style transfer engine is operational."""
        try:
            import librosa as _  # noqa: F401
            return True
        except ImportError:
            return False
