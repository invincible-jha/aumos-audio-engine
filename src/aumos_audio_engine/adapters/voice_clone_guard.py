"""Voice cloning protection adapter — GAP-82 competitive gap implementation.

Detects and blocks attempts to use real speaker audio for unauthorized
voice cloning. Applies speaker verification to identify if reference
audio belongs to a registered protected identity, and enforces consent
policies before allowing cloning operations.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Any

import numpy as np
import structlog
from aumos_common.errors import ValidationError
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class VoiceCloneGuard:
    """Protects against unauthorized voice cloning by verifying speaker consent.

    Computes speaker embeddings from reference audio and compares them
    against a registry of protected identities. Blocks cloning operations
    when the cosine similarity exceeds the protection threshold.

    Uses ECAPA-TDNN speaker embeddings for high-accuracy speaker verification
    with a default threshold of 0.75 (cosine similarity).

    Args:
        protection_threshold: Cosine similarity above which reference audio
            is considered to match a protected identity (default: 0.75).
        device: Torch device for embedding computation ("cuda", "cpu").
        embedding_dim: Speaker embedding dimensionality (default: 192 for ECAPA-TDNN).
        cache_dir: Model weight cache directory.
    """

    def __init__(
        self,
        protection_threshold: float = 0.75,
        device: str = "cpu",
        embedding_dim: int = 192,
        cache_dir: str = "/tmp/model-cache",
    ) -> None:
        self._threshold = protection_threshold
        self._device = device
        self._embedding_dim = embedding_dim
        self._cache_dir = cache_dir
        # In-memory registry: speaker_id -> embedding vector
        # Production: this should be backed by a database with tenant isolation
        self._protected_registry: dict[str, np.ndarray] = {}
        self._consent_registry: dict[str, set[str]] = {}  # speaker_id -> set of granted tenant_ids
        self._embedding_model: Any = None
        self._log = logger.bind(adapter="voice_clone_guard")

    async def initialize(self) -> None:
        """Load speaker embedding model.

        Uses SpeechBrain ECAPA-TDNN for speaker verification.
        """
        self._log.info("voice_clone_guard.initialize.start")
        await asyncio.to_thread(self._load_model_sync)
        self._log.info("voice_clone_guard.initialize.complete")

    def _load_model_sync(self) -> None:
        """Synchronous model loading — called from thread pool."""
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore[import]

            self._embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self._cache_dir,
                run_opts={"device": self._device},
            )
            self._log.info("voice_clone_guard.model_loaded", model="ecapa-tdnn")
        except Exception as exc:
            self._log.warning("voice_clone_guard.model_load_failed", error=str(exc))
            # Fall back to a hash-based fingerprint approach when model unavailable
            self._embedding_model = None

    async def check_cloning_permission(
        self,
        reference_audio_bytes: bytes,
        requesting_tenant_id: uuid.UUID,
        reference_description: str = "",
    ) -> dict[str, Any]:
        """Check whether reference audio can be used for voice cloning.

        Computes speaker embedding for the reference audio and verifies it
        against the protected identity registry. Returns permission status
        and reasoning.

        Args:
            reference_audio_bytes: Raw audio bytes of the reference speaker.
            requesting_tenant_id: Tenant requesting the cloning operation.
            reference_description: Human-readable description of the reference
                (for audit logging).

        Returns:
            Dict with keys:
            - allowed: bool — whether cloning is permitted
            - reason: str — explanation of the decision
            - matched_speaker_id: str | None — ID of matched protected identity
            - similarity_score: float — maximum similarity found
            - requires_consent: bool — whether explicit consent was checked
        """
        self._log.info(
            "voice_clone_guard.check",
            tenant_id=str(requesting_tenant_id),
            audio_bytes=len(reference_audio_bytes),
        )

        embedding = await self._extract_embedding(reference_audio_bytes)
        max_similarity = 0.0
        matched_speaker_id: str | None = None

        for speaker_id, protected_embedding in self._protected_registry.items():
            similarity = float(np.dot(embedding, protected_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(protected_embedding) + 1e-8
            ))
            if similarity > max_similarity:
                max_similarity = similarity
                matched_speaker_id = speaker_id

        if max_similarity >= self._threshold and matched_speaker_id is not None:
            # Check consent registry
            tenant_id_str = str(requesting_tenant_id)
            consented_tenants = self._consent_registry.get(matched_speaker_id, set())
            has_consent = tenant_id_str in consented_tenants

            self._log.warning(
                "voice_clone_guard.protected_match",
                speaker_id=matched_speaker_id,
                similarity=round(max_similarity, 4),
                has_consent=has_consent,
            )

            return {
                "allowed": has_consent,
                "reason": (
                    f"Reference audio matches protected identity '{matched_speaker_id}' "
                    f"(similarity: {max_similarity:.3f}). "
                    + ("Consent on file." if has_consent else "No consent granted for this tenant.")
                ),
                "matched_speaker_id": matched_speaker_id,
                "similarity_score": round(max_similarity, 4),
                "requires_consent": True,
            }

        self._log.info(
            "voice_clone_guard.no_match",
            max_similarity=round(max_similarity, 4),
            threshold=self._threshold,
        )
        return {
            "allowed": True,
            "reason": f"No protected identity matched (max similarity: {max_similarity:.3f})",
            "matched_speaker_id": None,
            "similarity_score": round(max_similarity, 4),
            "requires_consent": False,
        }

    async def register_protected_speaker(
        self,
        speaker_id: str,
        reference_audio_bytes: bytes,
        granted_tenant_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a speaker as a protected identity.

        Once registered, any reference audio that matches this speaker above
        the protection threshold will require explicit consent.

        Args:
            speaker_id: Unique identifier for this protected speaker.
            reference_audio_bytes: Reference audio for embedding registration.
            granted_tenant_ids: List of tenant IDs pre-authorized to clone this voice.

        Returns:
            Dict with registration status and speaker fingerprint hash.
        """
        embedding = await self._extract_embedding(reference_audio_bytes)
        self._protected_registry[speaker_id] = embedding
        self._consent_registry[speaker_id] = set(granted_tenant_ids or [])

        fingerprint = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
        self._log.info(
            "voice_clone_guard.speaker_registered",
            speaker_id=speaker_id,
            fingerprint=fingerprint,
            granted_tenants=len(granted_tenant_ids or []),
        )
        return {
            "speaker_id": speaker_id,
            "fingerprint": fingerprint,
            "granted_tenant_count": len(granted_tenant_ids or []),
        }

    async def grant_consent(
        self,
        speaker_id: str,
        tenant_id: str,
    ) -> None:
        """Grant a tenant permission to clone a protected speaker's voice.

        Args:
            speaker_id: Protected speaker identifier.
            tenant_id: Tenant being granted consent.

        Raises:
            ValidationError: If speaker_id is not registered.
        """
        if speaker_id not in self._protected_registry:
            raise ValidationError(f"Speaker '{speaker_id}' is not a registered protected identity.")
        self._consent_registry.setdefault(speaker_id, set()).add(tenant_id)
        self._log.info("voice_clone_guard.consent_granted", speaker_id=speaker_id, tenant_id=tenant_id)

    async def revoke_consent(
        self,
        speaker_id: str,
        tenant_id: str,
    ) -> None:
        """Revoke a tenant's permission to clone a protected speaker's voice.

        Args:
            speaker_id: Protected speaker identifier.
            tenant_id: Tenant whose consent is being revoked.
        """
        if speaker_id in self._consent_registry:
            self._consent_registry[speaker_id].discard(tenant_id)
        self._log.info("voice_clone_guard.consent_revoked", speaker_id=speaker_id, tenant_id=tenant_id)

    async def _extract_embedding(self, audio_bytes: bytes) -> np.ndarray:
        """Extract speaker embedding from audio bytes.

        Uses ECAPA-TDNN if available, otherwise falls back to an MFCC-based
        fingerprint for basic protection without the full model.

        Args:
            audio_bytes: Raw audio bytes (WAV format preferred).

        Returns:
            Normalized speaker embedding vector.
        """
        if self._embedding_model is not None:
            return await asyncio.to_thread(self._extract_embedding_sync, audio_bytes)
        return self._extract_mfcc_fingerprint(audio_bytes)

    def _extract_embedding_sync(self, audio_bytes: bytes) -> np.ndarray:
        """Synchronous ECAPA-TDNN embedding extraction."""
        import io
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        waveform = waveform.to(self._device)
        with torch.no_grad():
            embedding = self._embedding_model.encode_batch(waveform)

        emb = embedding.squeeze().cpu().numpy()
        # Normalize to unit vector for cosine similarity
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _extract_mfcc_fingerprint(self, audio_bytes: bytes) -> np.ndarray:
        """MFCC-based speaker fingerprint as fallback when model is unavailable.

        Less accurate than ECAPA-TDNN but provides basic protection.
        """
        import io
        import librosa  # type: ignore[import]

        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self._embedding_dim)
        fingerprint = mfcc.mean(axis=1)
        return fingerprint / (np.linalg.norm(fingerprint) + 1e-8)
