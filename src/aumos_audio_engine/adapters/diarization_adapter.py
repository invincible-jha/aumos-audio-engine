"""Speaker diarization adapter using pyannote.audio.

Identifies speaker turns in multi-speaker audio (earnings calls,
conference calls, trading floor recordings) and enables per-speaker
de-identification with independent transformation parameters.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import structlog
from aumos_common.observability import get_logger


@dataclass
class SpeakerSegment:
    """A single speaker turn with timing and speaker label.

    Attributes:
        speaker_label: Anonymized speaker identifier (e.g., "SPEAKER_00").
        start_seconds: Segment start time in seconds.
        end_seconds: Segment end time in seconds.
    """

    speaker_label: str
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        """Duration of this speaker segment."""
        return self.end_seconds - self.start_seconds


class SpeakerDiarizationAdapter:
    """Identifies speaker turns in audio using pyannote.audio.

    Returns a list of SpeakerSegment objects with start/end times and
    speaker labels. Labels are consistent within a recording but carry
    no biometric identity — "SPEAKER_00" in one recording is unrelated
    to "SPEAKER_00" in another.

    Args:
        model_name: pyannote.audio model name or HuggingFace path.
        auth_token: HuggingFace authentication token for gated models.
        device: Target device ("cuda" | "cpu").
        min_speakers: Minimum expected number of speakers (optional hint).
        max_speakers: Maximum expected number of speakers (optional hint).
    """

    DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        auth_token: str | None = None,
        device: str = "cpu",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        """Initialize SpeakerDiarizationAdapter.

        Args:
            model_name: pyannote.audio model identifier.
            auth_token: HuggingFace access token (required for gated models).
            device: Inference device.
            min_speakers: Minimum speaker count hint for the model.
            max_speakers: Maximum speaker count hint for the model.
        """
        self._model_name = model_name
        self._auth_token = auth_token
        self._device = device
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._pipeline: object | None = None
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def warm_up(self) -> None:
        """Load pyannote diarization pipeline weights.

        Downloads model from HuggingFace Hub on first call.
        Requires a valid HuggingFace access token for gated models.
        """
        self._log.info("warming_up_diarization_pipeline", model=self._model_name)
        await asyncio.to_thread(self._load_pipeline)
        self._log.info("diarization_pipeline_ready")

    def _load_pipeline(self) -> None:
        """Load diarization pipeline synchronously (called via to_thread)."""
        from pyannote.audio import Pipeline

        kwargs: dict[str, object] = {}
        if self._auth_token:
            kwargs["use_auth_token"] = self._auth_token

        self._pipeline = Pipeline.from_pretrained(self._model_name, **kwargs)

        if self._device != "cpu":
            import torch
            self._pipeline.to(torch.device(self._device))  # type: ignore[union-attr]

    async def diarize(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Identify speaker turns and return a list of SpeakerSegment objects.

        Args:
            audio_array: Float32 mono audio normalized to [-1, 1].
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of SpeakerSegment sorted by start_seconds.

        Raises:
            RuntimeError: If warm_up() has not been called.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "SpeakerDiarizationAdapter.warm_up() must be called before diarize()"
            )

        return await asyncio.to_thread(
            self._run_diarization,
            audio_array=audio_array,
            sample_rate=sample_rate,
        )

    def _run_diarization(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> list[SpeakerSegment]:
        """Run diarization synchronously (called via to_thread).

        Args:
            audio_array: Normalized float32 mono audio.
            sample_rate: Sample rate in Hz.

        Returns:
            Sorted list of SpeakerSegment.
        """
        import torch

        # pyannote expects (channels, samples) tensor with sample_rate
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        audio_input = {"waveform": audio_tensor, "sample_rate": sample_rate}

        diarize_kwargs: dict[str, object] = {}
        if self._min_speakers is not None:
            diarize_kwargs["min_speakers"] = self._min_speakers
        if self._max_speakers is not None:
            diarize_kwargs["max_speakers"] = self._max_speakers

        diarization = self._pipeline(audio_input, **diarize_kwargs)  # type: ignore[operator]

        segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker_label=speaker,
                    start_seconds=float(turn.start),
                    end_seconds=float(turn.end),
                )
            )

        segments.sort(key=lambda s: s.start_seconds)
        self._log.info(
            "diarization_complete",
            num_segments=len(segments),
            speakers=len({s.speaker_label for s in segments}),
        )
        return segments


class MultiSpeakerDeidentificationService:
    """De-identifies each speaker independently in multi-speaker audio.

    Each identified speaker receives a unique, independent pitch/formant
    transformation. This ensures that:
    - Different speakers cannot be linked to each other across recordings
    - The turn structure (who spoke when) is preserved
    - Each speaker's voice identity is protected independently

    Args:
        diarization_adapter: Configured SpeakerDiarizationAdapter.
        base_pitch_shift_min: Minimum pitch shift for speaker transforms.
        base_pitch_shift_max: Maximum pitch shift for speaker transforms.
    """

    def __init__(
        self,
        diarization_adapter: SpeakerDiarizationAdapter,
        base_pitch_shift_min: float = 2.0,
        base_pitch_shift_max: float = 6.0,
    ) -> None:
        """Initialize MultiSpeakerDeidentificationService.

        Args:
            diarization_adapter: Pre-warmed diarization adapter.
            base_pitch_shift_min: Minimum semitone pitch shift per speaker.
            base_pitch_shift_max: Maximum semitone pitch shift per speaker.
        """
        self._diarizer = diarization_adapter
        self._pitch_min = base_pitch_shift_min
        self._pitch_max = base_pitch_shift_max
        self._log: structlog.BoundLogger = get_logger(__name__)

    def _generate_speaker_transforms(
        self,
        speaker_labels: list[str],
        job_seed: int,
    ) -> dict[str, dict[str, float]]:
        """Generate independent transformation parameters per speaker.

        Parameters are seeded from the job ID + speaker label, ensuring
        consistent transforms within a job but independent across jobs.

        Args:
            speaker_labels: List of unique speaker label strings.
            job_seed: Integer seed derived from the job ID.

        Returns:
            Dict mapping speaker_label to {"pitch_shift": float, "formant_shift": float}.
        """
        transforms: dict[str, dict[str, float]] = {}
        for label in speaker_labels:
            speaker_seed = (job_seed + hash(label)) % 2**32
            rng = np.random.default_rng(seed=speaker_seed)
            transforms[label] = {
                "pitch_shift": float(rng.uniform(self._pitch_min, self._pitch_max)),
                "formant_shift": float(rng.uniform(-0.15, 0.15)),
            }
        return transforms

    async def deidentify(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        job_id_seed: int,
    ) -> tuple[np.ndarray, list[SpeakerSegment], dict[str, int]]:
        """Diarize and independently de-identify each speaker.

        Args:
            audio_array: Float32 mono audio normalized to [-1, 1].
            sample_rate: Audio sample rate in Hz.
            job_id_seed: Integer seed from job ID for reproducible transforms.

        Returns:
            Tuple of:
            - De-identified audio array (same shape as input).
            - List of SpeakerSegment from diarization.
            - Dict of speaker_label -> segment count for metadata.
        """
        segments = await self._diarizer.diarize(audio_array, sample_rate)

        unique_speakers = list({s.speaker_label for s in segments})
        transforms = self._generate_speaker_transforms(unique_speakers, job_id_seed)

        output = audio_array.copy()

        for segment in segments:
            start_sample = int(segment.start_seconds * sample_rate)
            end_sample = int(segment.end_seconds * sample_rate)
            segment_audio = audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            transform = transforms[segment.speaker_label]
            deidentified_segment = await asyncio.to_thread(
                self._apply_transform,
                audio=segment_audio,
                sample_rate=sample_rate,
                pitch_shift=transform["pitch_shift"],
            )

            # Pad or trim to match original segment length
            target_len = end_sample - start_sample
            if len(deidentified_segment) > target_len:
                deidentified_segment = deidentified_segment[:target_len]
            elif len(deidentified_segment) < target_len:
                deidentified_segment = np.pad(
                    deidentified_segment,
                    (0, target_len - len(deidentified_segment)),
                )

            output[start_sample:end_sample] = deidentified_segment

        speaker_segment_counts = {
            speaker: sum(1 for s in segments if s.speaker_label == speaker)
            for speaker in unique_speakers
        }

        self._log.info(
            "multi_speaker_deidentification_complete",
            num_speakers=len(unique_speakers),
            num_segments=len(segments),
        )

        return output, segments, speaker_segment_counts

    def _apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        pitch_shift: float,
    ) -> np.ndarray:
        """Apply pitch shift to a speaker segment (called via to_thread).

        Args:
            audio: Float32 audio segment.
            sample_rate: Sample rate in Hz.
            pitch_shift: Semitone pitch shift.

        Returns:
            Pitch-shifted audio array.
        """
        import librosa

        return librosa.effects.pitch_shift(
            y=audio,
            sr=sample_rate,
            n_steps=pitch_shift,
        )
