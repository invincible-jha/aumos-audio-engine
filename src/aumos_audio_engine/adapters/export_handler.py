"""Audio file export adapter.

Implements AudioExportHandlerProtocol. Handles multi-format audio encoding
(WAV, MP3, OGG, FLAC), metadata embedding, and upload to MinIO/S3 compatible
object storage using the aioboto3 / boto3 client.
"""

import asyncio
import io
import struct
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)

# ── Supported export formats and their soundfile format identifiers ───────────
_FORMAT_MAP: dict[str, str] = {
    "wav":  "WAV",
    "flac": "FLAC",
    "ogg":  "OGG",
}

# Formats that require pydub/ffmpeg (not natively supported by soundfile)
_FFMPEG_FORMATS = frozenset({"mp3"})


class AudioExportHandler:
    """Multi-format audio export handler with MinIO/S3 storage integration.

    Supports:
        - WAV export (configurable sample rate and bit depth: 16/24/32-bit PCM).
        - MP3 export (configurable bitrate via pydub/lameenc).
        - OGG/Vorbis export (soundfile native).
        - FLAC lossless export (soundfile native).
        - ID3/Vorbis metadata embedding.
        - Upload to MinIO or S3-compatible object storage.
        - Batch export across multiple audio items.

    All I/O operations use async wrappers around blocking calls executed in
    the thread-pool executor.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize export handler.

        Args:
            settings: Audio engine settings containing storage bucket and temp_dir.
        """
        self._settings = settings
        self._s3_client: Any = None

    async def _get_s3_client(self) -> Any:
        """Lazily create and return a boto3 S3 client.

        Returns:
            Configured boto3 S3 client (compatible with MinIO).
        """
        if self._s3_client is not None:
            return self._s3_client

        loop = asyncio.get_running_loop()
        self._s3_client = await loop.run_in_executor(None, self._build_s3_client)
        return self._s3_client

    def _build_s3_client(self) -> Any:
        """Build boto3 S3 client synchronously."""
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.config import Config  # type: ignore[import-untyped]

            storage_url = getattr(self._settings, "storage_endpoint_url", None)
            access_key = getattr(self._settings, "storage_access_key", None)
            secret_key = getattr(self._settings, "storage_secret_key", None)

            client_kwargs: dict[str, Any] = {
                "service_name": "s3",
                "config": Config(signature_version="s3v4"),
            }
            if storage_url:
                client_kwargs["endpoint_url"] = storage_url
            if access_key and secret_key:
                client_kwargs["aws_access_key_id"] = access_key
                client_kwargs["aws_secret_access_key"] = secret_key

            return boto3.client(**client_kwargs)
        except ImportError:
            logger.error("boto3 package not installed. Install with: pip install boto3>=1.34.0")
            raise

    async def export_wav(
        self,
        audio_bytes: bytes,
        sample_rate: int = 22050,
        bit_depth: int = 16,
        metadata: dict | None = None,
    ) -> bytes:
        """Export audio as WAV with configurable bit depth.

        Args:
            audio_bytes: Source audio bytes (any format readable by soundfile).
            sample_rate: Target output sample rate in Hz.
            bit_depth: PCM bit depth: 16, 24, or 32.
            metadata: Optional dict with keys like 'title', 'artist', 'copyright'.

        Returns:
            WAV-encoded audio bytes.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._export_wav_sync,
            audio_bytes,
            sample_rate,
            bit_depth,
            metadata or {},
        )
        logger.info("WAV export complete", output_bytes=len(result), bit_depth=bit_depth)
        return result

    def _export_wav_sync(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        bit_depth: int,
        metadata: dict,
    ) -> bytes:
        """Synchronous WAV encoding."""
        audio_array, source_sr = self._load_audio(audio_bytes)

        if source_sr != sample_rate:
            import librosa  # type: ignore[import-untyped]
            audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=sample_rate)

        subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
        subtype = subtype_map.get(bit_depth, "PCM_16")

        output_buffer = io.BytesIO()
        sf.write(output_buffer, audio_array, sample_rate, format="WAV", subtype=subtype)
        wav_bytes = output_buffer.getvalue()

        if metadata:
            wav_bytes = self._embed_wav_metadata(wav_bytes, metadata)

        return wav_bytes

    async def export_mp3(
        self,
        audio_bytes: bytes,
        bitrate_kbps: int = 192,
        sample_rate: int = 44100,
        metadata: dict | None = None,
    ) -> bytes:
        """Export audio as MP3 with configurable bitrate.

        Uses lameenc for pure-Python MP3 encoding (no ffmpeg dependency).

        Args:
            audio_bytes: Source audio bytes.
            bitrate_kbps: MP3 bitrate in kbps (64, 96, 128, 192, 256, 320).
            sample_rate: Target sample rate in Hz.
            metadata: Optional ID3 metadata dict ('title', 'artist', 'copyright').

        Returns:
            MP3-encoded audio bytes.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._export_mp3_sync,
            audio_bytes,
            bitrate_kbps,
            sample_rate,
            metadata or {},
        )
        logger.info("MP3 export complete", output_bytes=len(result), bitrate_kbps=bitrate_kbps)
        return result

    def _export_mp3_sync(
        self,
        audio_bytes: bytes,
        bitrate_kbps: int,
        sample_rate: int,
        metadata: dict,
    ) -> bytes:
        """Synchronous MP3 encoding using lameenc."""
        try:
            import lameenc  # type: ignore[import-untyped]
        except ImportError:
            logger.error("lameenc not installed. Install with: pip install lameenc>=1.7.0")
            raise

        audio_array, source_sr = self._load_audio(audio_bytes)

        if source_sr != sample_rate:
            import librosa  # type: ignore[import-untyped]
            audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=sample_rate)

        # Convert to int16 PCM for lameenc
        pcm_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate_kbps)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(2)  # 2 = high quality, 7 = fast

        mp3_data = encoder.encode(pcm_int16.tobytes())
        mp3_data += encoder.flush()

        if metadata:
            mp3_data = self._embed_id3_tags(mp3_data, metadata)

        return bytes(mp3_data)

    async def export_ogg(
        self,
        audio_bytes: bytes,
        sample_rate: int = 44100,
        quality: float = 0.6,
        metadata: dict | None = None,
    ) -> bytes:
        """Export audio as OGG/Vorbis.

        Args:
            audio_bytes: Source audio bytes.
            sample_rate: Target sample rate in Hz.
            quality: Vorbis quality level [-0.1, 1.0]. 0.6 ≈ 192 kbps.
            metadata: Optional Vorbis comment dict.

        Returns:
            OGG/Vorbis encoded audio bytes.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._export_soundfile_sync,
            audio_bytes,
            sample_rate,
            "OGG",
            "VORBIS",
            metadata or {},
        )
        logger.info("OGG export complete", output_bytes=len(result))
        return result

    async def export_flac(
        self,
        audio_bytes: bytes,
        sample_rate: int = 44100,
        metadata: dict | None = None,
    ) -> bytes:
        """Export audio as FLAC (lossless).

        Args:
            audio_bytes: Source audio bytes.
            sample_rate: Target sample rate in Hz.
            metadata: Optional Vorbis/FLAC comment dict.

        Returns:
            FLAC encoded audio bytes.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._export_soundfile_sync,
            audio_bytes,
            sample_rate,
            "FLAC",
            "PCM_16",
            metadata or {},
        )
        logger.info("FLAC export complete", output_bytes=len(result))
        return result

    def _export_soundfile_sync(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        sf_format: str,
        sf_subtype: str,
        metadata: dict,
    ) -> bytes:
        """Generic soundfile-based export (runs in thread pool).

        Args:
            audio_bytes: Source audio bytes.
            sample_rate: Target sample rate.
            sf_format: soundfile format string (e.g., 'OGG', 'FLAC').
            sf_subtype: soundfile subtype string (e.g., 'VORBIS', 'PCM_16').
            metadata: Optional metadata dict for embedding.

        Returns:
            Encoded audio bytes.
        """
        audio_array, source_sr = self._load_audio(audio_bytes)

        if source_sr != sample_rate:
            import librosa  # type: ignore[import-untyped]
            audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=sample_rate)

        output_buffer = io.BytesIO()
        sf.write(output_buffer, audio_array, sample_rate, format=sf_format, subtype=sf_subtype)
        return output_buffer.getvalue()

    async def export(
        self,
        audio_bytes: bytes,
        output_format: str,
        export_options: dict | None = None,
        metadata: dict | None = None,
    ) -> bytes:
        """Export audio to the specified format using appropriate encoder.

        Dispatches to the correct format-specific method based on output_format.

        Args:
            audio_bytes: Source audio bytes.
            output_format: Target format: 'wav', 'mp3', 'ogg', 'flac'.
            export_options: Format-specific options dict. Keys vary by format:
                - wav: 'sample_rate' (int), 'bit_depth' (int).
                - mp3: 'sample_rate' (int), 'bitrate_kbps' (int).
                - ogg: 'sample_rate' (int), 'quality' (float).
                - flac: 'sample_rate' (int).
            metadata: Optional metadata for embedding.

        Returns:
            Encoded audio bytes.

        Raises:
            ValueError: If output_format is not supported.
        """
        options = export_options or {}
        meta = metadata or {}

        if output_format == "wav":
            return await self.export_wav(
                audio_bytes,
                sample_rate=int(options.get("sample_rate", self._settings.sample_rate)),
                bit_depth=int(options.get("bit_depth", 16)),
                metadata=meta,
            )
        elif output_format == "mp3":
            return await self.export_mp3(
                audio_bytes,
                bitrate_kbps=int(options.get("bitrate_kbps", 192)),
                sample_rate=int(options.get("sample_rate", 44100)),
                metadata=meta,
            )
        elif output_format == "ogg":
            return await self.export_ogg(
                audio_bytes,
                sample_rate=int(options.get("sample_rate", 44100)),
                quality=float(options.get("quality", 0.6)),
                metadata=meta,
            )
        elif output_format == "flac":
            return await self.export_flac(
                audio_bytes,
                sample_rate=int(options.get("sample_rate", 44100)),
                metadata=meta,
            )
        else:
            raise ValueError(
                f"Unsupported export format '{output_format}'. "
                f"Supported: wav, mp3, ogg, flac"
            )

    async def upload(
        self,
        tenant_id: str,
        job_id: str,
        audio_bytes: bytes,
        audio_format: str,
    ) -> str:
        """Upload audio bytes to MinIO/S3 object storage.

        Constructs a namespaced storage path:
        ``{bucket}/{tenant_id}/{job_id}/output.{audio_format}``

        Args:
            tenant_id: Tenant identifier for namespace isolation.
            job_id: Job identifier for unique file naming.
            audio_bytes: Audio data to upload.
            audio_format: File extension for the stored object.

        Returns:
            Storage URI in the form ``s3://{bucket}/{tenant_id}/{job_id}/output.{format}``.
        """
        client = await self._get_s3_client()
        object_key = str(
            PurePosixPath(tenant_id) / job_id / f"output.{audio_format}"
        )
        bucket = self._settings.output_bucket

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._upload_sync,
            client,
            bucket,
            object_key,
            audio_bytes,
            audio_format,
        )

        storage_uri = f"s3://{bucket}/{object_key}"
        logger.info(
            "Audio uploaded to storage",
            storage_uri=storage_uri,
            bytes_uploaded=len(audio_bytes),
            audio_format=audio_format,
        )
        return storage_uri

    def _upload_sync(
        self,
        client: Any,
        bucket: str,
        object_key: str,
        audio_bytes: bytes,
        audio_format: str,
    ) -> None:
        """Synchronous S3 upload (runs in thread pool)."""
        content_type_map = {
            "wav":  "audio/wav",
            "mp3":  "audio/mpeg",
            "ogg":  "audio/ogg",
            "flac": "audio/flac",
        }
        content_type = content_type_map.get(audio_format, "application/octet-stream")

        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=io.BytesIO(audio_bytes),
            ContentType=content_type,
            ContentLength=len(audio_bytes),
        )

    async def download(self, storage_uri: str) -> bytes:
        """Download audio bytes from object storage.

        Args:
            storage_uri: Storage URI in the form ``s3://bucket/key``.

        Returns:
            Raw audio bytes.
        """
        bucket, object_key = self._parse_storage_uri(storage_uri)
        client = await self._get_s3_client()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._download_sync,
            client,
            bucket,
            object_key,
        )

    def _download_sync(self, client: Any, bucket: str, object_key: str) -> bytes:
        """Synchronous S3 download."""
        response = client.get_object(Bucket=bucket, Key=object_key)
        return response["Body"].read()

    async def delete(self, storage_uri: str) -> None:
        """Delete an audio object from storage.

        Args:
            storage_uri: Storage URI to delete.
        """
        bucket, object_key = self._parse_storage_uri(storage_uri)
        client = await self._get_s3_client()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: client.delete_object(Bucket=bucket, Key=object_key),
        )
        logger.info("Audio object deleted", storage_uri=storage_uri)

    async def batch_export(
        self,
        items: list[tuple[bytes, str, dict]],
        output_format: str,
        export_options: dict | None = None,
    ) -> list[bytes]:
        """Export multiple audio items to the specified format in parallel.

        Args:
            items: List of (audio_bytes, source_format, metadata) tuples.
            output_format: Target format for all items.
            export_options: Format-specific options applied to all items.

        Returns:
            List of encoded audio bytes in the same order as items.
        """
        tasks = [
            self.export(
                audio_bytes=audio_bytes,
                output_format=output_format,
                export_options=export_options,
                metadata=metadata,
            )
            for audio_bytes, _, metadata in items
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: list[bytes] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch export item failed",
                    item_index=index,
                    output_format=output_format,
                    error=str(result),
                )
                raise result
            output.append(result)  # type: ignore[arg-type]

        logger.info(
            "Batch export complete",
            total_items=len(items),
            output_format=output_format,
        )
        return output

    # ── Metadata embedding helpers ────────────────────────────────────────────

    @staticmethod
    def _embed_wav_metadata(wav_bytes: bytes, metadata: dict) -> bytes:
        """Embed metadata into a WAV file using LIST INFO chunks.

        This is a lightweight pure-Python implementation that injects
        RIFF INFO tags (INAM, IART, ICOP) into the WAV container.

        Args:
            wav_bytes: Valid WAV bytes.
            metadata: Dict with optional keys: 'title', 'artist', 'copyright'.

        Returns:
            WAV bytes with INFO chunk embedded after the fmt chunk.
        """
        # Build INFO sub-chunks
        info_chunks = b""
        tag_map = {
            "title":     b"INAM",
            "artist":    b"IART",
            "copyright": b"ICOP",
            "comment":   b"ICMT",
        }
        for field_name, tag in tag_map.items():
            if field_name in metadata and metadata[field_name]:
                value = metadata[field_name].encode("utf-8") + b"\x00"
                if len(value) % 2 == 1:
                    value += b"\x00"  # Pad to even length
                info_chunks += tag + struct.pack("<I", len(value) - 1) + value

        if not info_chunks:
            return wav_bytes

        # Wrap in LIST INFO chunk
        list_chunk = b"LIST" + struct.pack("<I", 4 + len(info_chunks)) + b"INFO" + info_chunks

        # Insert LIST chunk before data chunk in the RIFF container
        riff_header = wav_bytes[:12]
        remaining = wav_bytes[12:]
        new_wav = riff_header + list_chunk + remaining

        # Update RIFF size field
        new_size = len(new_wav) - 8
        new_wav = new_wav[:4] + struct.pack("<I", new_size) + new_wav[8:]
        return new_wav

    @staticmethod
    def _embed_id3_tags(mp3_bytes: bytes, metadata: dict) -> bytes:
        """Prepend a minimal ID3v2.3 tag block to MP3 bytes.

        Supports TIT2 (title), TPE1 (artist), TCOP (copyright) frames.

        Args:
            mp3_bytes: Raw MP3 audio bytes.
            metadata: Dict with optional 'title', 'artist', 'copyright' keys.

        Returns:
            MP3 bytes with ID3v2.3 tag prepended.
        """
        frame_map = {
            "title":     b"TIT2",
            "artist":    b"TPE1",
            "copyright": b"TCOP",
        }
        frames_data = b""
        for field_name, frame_id in frame_map.items():
            if field_name in metadata and metadata[field_name]:
                # Encoding byte 0x03 = UTF-8
                text_bytes = b"\x03" + metadata[field_name].encode("utf-8")
                frames_data += frame_id + struct.pack(">I", len(text_bytes)) + b"\x00\x00" + text_bytes

        if not frames_data:
            return mp3_bytes

        # ID3v2 header: "ID3" + version 2.3.0 + flags 0x00 + syncsafe size
        tag_size = len(frames_data)
        syncsafe_size = (
            ((tag_size >> 21) & 0x7F) << 24
            | ((tag_size >> 14) & 0x7F) << 16
            | ((tag_size >> 7) & 0x7F) << 8
            | (tag_size & 0x7F)
        )
        id3_header = b"ID3\x03\x00\x00" + struct.pack(">I", syncsafe_size)
        return id3_header + frames_data + mp3_bytes

    @staticmethod
    def _parse_storage_uri(storage_uri: str) -> tuple[str, str]:
        """Parse a storage URI into bucket and object key.

        Args:
            storage_uri: URI in the form 's3://bucket/key/path'.

        Returns:
            Tuple of (bucket_name, object_key).

        Raises:
            ValueError: If the URI format is invalid.
        """
        if not storage_uri.startswith("s3://"):
            raise ValueError(f"Invalid storage URI (must start with 's3://'): {storage_uri}")
        without_scheme = storage_uri[len("s3://"):]
        parts = without_scheme.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage URI (missing object key): {storage_uri}")
        return parts[0], parts[1]

    @staticmethod
    def _load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode audio bytes to float32 mono numpy array.

        Args:
            audio_bytes: Raw audio bytes in any soundfile-supported format.

        Returns:
            Tuple of (float32 mono audio array, sample_rate).
        """
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data.astype(np.float32), int(sample_rate)

    async def health_check(self) -> bool:
        """Return True if the export handler and storage backend are reachable."""
        try:
            client = await self._get_s3_client()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: client.list_buckets(),
            )
            return True
        except Exception as exc:
            logger.warning("Export handler health check failed", error=str(exc))
            return False
