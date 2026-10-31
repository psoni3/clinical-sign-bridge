"""Session and consultation validation helpers."""

from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException

from app.domain.consultation import Consultation, ConsultationSession


def validate_session_active(
    session: ConsultationSession | None,
    now: datetime | None = None,
) -> ConsultationSession:
    """Validate session is active (not expired, not revoked). Returns the session."""
    now = now or datetime.now(UTC)

    if session is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "SESSION_NOT_FOUND"},
        )

    if session.is_expired(now):
        raise HTTPException(
            status_code=401,
            detail={"code": "SESSION_EXPIRED"},
        )

    if session.is_revoked():
        raise HTTPException(
            status_code=401,
            detail={"code": "SESSION_REVOKED"},
        )

    return session


def validate_session_belongs_to_consultation(
    session: ConsultationSession,
    consultation_id: UUID,
) -> None:
    """Validate session belongs to consultation."""
    if session.consultation_id != consultation_id:
        raise HTTPException(
            status_code=401,
            detail={"code": "SESSION_MISMATCH"},
        )


def validate_session_matches_resource(
    session: ConsultationSession,
    resource_session_id: UUID,
) -> None:
    """Prevent another session in the same consultation from acting on a resource."""
    if session.id != resource_session_id:
        raise HTTPException(
            status_code=401,
            detail={"code": "SESSION_MISMATCH"},
        )


def validate_consultation_active(
    consultation: Consultation | None,
    now: datetime | None = None,
) -> Consultation:
    """Validate consultation is active (not expired, not completed). Returns the consultation."""
    now = now or datetime.now(UTC)

    if consultation is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "CONSULTATION_NOT_FOUND"},
        )

    if consultation.is_expired(now):
        raise HTTPException(
            status_code=410,
            detail={"code": "CONSULTATION_EXPIRED"},
        )

    if consultation.status == "COMPLETED":
        raise HTTPException(
            status_code=410,
            detail={"code": "CONSULTATION_COMPLETED"},
        )

    if consultation.status not in ("ACTIVE", "JOINED"):
        raise HTTPException(
            status_code=409,
            detail={"code": "CONSULTATION_NOT_ACTIVE"},
        )

    return consultation
