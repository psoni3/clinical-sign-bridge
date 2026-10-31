"""Domain models and value objects for consultation workflow."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID


class ConsultationStatus(StrEnum):
    """Consultation lifecycle states."""

    CREATED = "CREATED"
    JOINED = "JOINED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"


class SessionType(StrEnum):
    """Session participant type."""

    PATIENT_DEVICE = "PATIENT_DEVICE"
    DOCTOR_VIEW = "DOCTOR_VIEW"


class ConsultationError(Exception):
    """Base exception for consultation domain errors."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ConsultationExpiredError(ConsultationError):
    """Raised when consultation has expired."""

    def __init__(self) -> None:
        super().__init__("CONSULTATION_EXPIRED", "Consultation has expired")


class ConsultationCompletedError(ConsultationError):
    """Raised when consultation has been completed."""

    def __init__(self) -> None:
        super().__init__("CONSULTATION_COMPLETED", "Consultation is completed")


class ConsultationNotJoinedError(ConsultationError):
    """Raised when consultation not yet joined."""

    def __init__(self) -> None:
        super().__init__("CONSULTATION_NOT_JOINED", "Consultation has not been joined")


class SessionNotFoundError(ConsultationError):
    """Raised when session is not found or invalid."""

    def __init__(self) -> None:
        super().__init__("SESSION_MISMATCH", "Session not found or invalid")


class SessionExpiredError(ConsultationError):
    """Raised when session has expired."""

    def __init__(self) -> None:
        super().__init__("SESSION_EXPIRED", "Session has expired")


@dataclass(frozen=True)
class JoinCode:
    """Value object for consultation join codes.

    Join codes are high-entropy, short-lived, and stored as hashes.
    """

    plaintext: str
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if join code has expired."""
        now = now or datetime.now(UTC)
        return now >= self.expires_at


@dataclass(frozen=True)
class SessionToken:
    """Value object for session tokens.

    Session tokens bind a participant to a consultation and are
    temporarily valid. Stored as hashes; not persisted plaintext.
    """

    plaintext: str
    expires_at: datetime
    session_type: str

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if session token has expired."""
        now = now or datetime.now(UTC)
        return now >= self.expires_at


@dataclass
class Consultation:
    """Consultation aggregate root."""

    id: UUID
    display_label: str | None
    status: str
    expires_at: datetime
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if consultation has expired."""
        now = now or datetime.now(UTC)
        return now >= self.expires_at

    def is_active_or_completed(self) -> bool:
        """Check if consultation is in a terminal or active state."""
        return self.status in ("ACTIVE", "COMPLETED")

    def validate_active_for_operation(self, now: datetime | None = None) -> None:
        """Validate consultation can accept operations."""
        now = now or datetime.now(UTC)

        if self.is_expired(now):
            raise ConsultationExpiredError()

        if self.status == "COMPLETED":
            raise ConsultationCompletedError()

        if self.status == "CREATED":
            raise ConsultationNotJoinedError()


@dataclass
class ConsultationSession:
    """Session aggregate for a participant."""

    id: UUID
    consultation_id: UUID
    session_type: str
    device_label: str | None
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    last_seen_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if session has expired."""
        now = now or datetime.now(UTC)
        return now >= self.expires_at

    def is_revoked(self) -> bool:
        """Check if session has been explicitly revoked."""
        return self.revoked_at is not None

    def validate_active(self, now: datetime | None = None) -> None:
        """Validate session is still valid."""
        now = now or datetime.now(UTC)

        if self.is_revoked():
            raise SessionExpiredError()

        if self.is_expired(now):
            raise SessionExpiredError()
