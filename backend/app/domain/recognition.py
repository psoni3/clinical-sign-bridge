"""Domain models for recognition request validation and media."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


class RecognitionConstraints:
    """Upload constraints for video segments (M1 scope)."""

    MAX_DURATION_MS = 30_000  # 30 seconds
    MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB (conservative for prototype)

    # Do not retain raw video by default (spec requirement)
    RETAIN_RAW_VIDEO = False


@dataclass
class MediaMetadata:
    """Media characteristics for audit and quality assessment."""

    duration_ms: int
    size_bytes: int
    codec: str | None = None
    frame_rate: float | None = None
    width: int | None = None
    height: int | None = None


class ValidationError(Exception):
    """Base exception for validation failures."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class SegmentTooLongError(ValidationError):
    """Raised when video segment exceeds duration limit."""

    def __init__(self) -> None:
        super().__init__(
            "SEGMENT_TOO_LONG", f"Video segment exceeds {RecognitionConstraints.MAX_DURATION_MS}ms"
        )


class SegmentTooLargeError(ValidationError):
    """Raised when video segment exceeds size limit."""

    def __init__(self) -> None:
        super().__init__(
            "SEGMENT_TOO_LARGE",
            f"Video segment exceeds {RecognitionConstraints.MAX_SIZE_BYTES} bytes",
        )


class MediaInvalidError(ValidationError):
    """Raised when media cannot be processed."""

    def __init__(self, detail: str = "") -> None:
        msg = "Media is invalid or unsupported"
        if detail:
            msg += f": {detail}"
        super().__init__("CAMERA_MEDIA_INVALID", msg)


@dataclass
class Recognition:
    """Recognition request lifecycle tracking.

    Idempotency: consultation_id + session_id + client_request_id uniquely
    identifies a request. Replaying with same key returns same result.
    """

    id: UUID
    consultation_id: UUID
    session_id: UUID
    client_request_id: str  # Idempotency key from client
    status: str  # RecognitionStatus enum
    media_metadata: MediaMetadata
    temporary_media_reference: str | None  # Temporary URI, then deleted
    created_at: datetime
    completed_at: datetime | None = None
