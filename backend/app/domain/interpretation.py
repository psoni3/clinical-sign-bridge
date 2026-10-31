"""Domain models for message interpretation and confirmation."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class InterpretationStatus(StrEnum):
    """Interpretation lifecycle states."""

    UNVERIFIED = "UNVERIFIED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class RecognitionStatus(StrEnum):
    """Recognition processing states."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class ConfirmationAction(StrEnum):
    """Patient actions on AI-generated interpretations."""

    CONFIRM = "CONFIRM"
    CORRECT = "CORRECT"
    REJECT = "REJECT"
    CLARIFICATION_REQUESTED = "CLARIFICATION_REQUESTED"


class MessageType(StrEnum):
    """Message classification."""

    TEXT = "TEXT"
    AI_TRANSLATION = "AI_TRANSLATION"
    VISUAL_RESPONSE = "VISUAL_RESPONSE"
    SYSTEM = "SYSTEM"


class VerificationStatus(StrEnum):
    """Message verification/confirmation status."""

    PATIENT_CONFIRMED = "PATIENT_CONFIRMED"
    CLINICIAN_REVIEWED = "CLINICIAN_REVIEWED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MessageProvenance(StrEnum):
    """Authoritative source of message content."""

    PATIENT_TYPED = "PATIENT_TYPED"
    PATIENT_CORRECTED = "PATIENT_CORRECTED"
    AI_CANDIDATE_CONFIRMED = "AI_CANDIDATE_CONFIRMED"
    DOCTOR_AUTHORED = "DOCTOR_AUTHORED"
    SYSTEM = "SYSTEM"


class SenderType(StrEnum):
    """Message sender classification."""

    PATIENT = "PATIENT"
    DOCTOR = "DOCTOR"
    SYSTEM = "SYSTEM"


class InterpretationError(Exception):
    """Base exception for interpretation domain errors."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class InterpretationNotConfirmableError(InterpretationError):
    """Raised when interpretation cannot be confirmed."""

    def __init__(self, reason: str = "") -> None:
        msg = "Interpretation not confirmable"
        if reason:
            msg += f": {reason}"
        super().__init__("INTERPRETATION_NOT_CONFIRMABLE", msg)


class CandidateNotFoundError(InterpretationError):
    """Raised when candidate does not exist."""

    def __init__(self) -> None:
        super().__init__("CANDIDATE_NOT_FOUND", "Candidate not found")


class DuplicateConfirmationError(InterpretationError):
    """Raised when interpretation was already confirmed/corrected."""

    def __init__(self) -> None:
        super().__init__("IDEMPOTENCY_CONFLICT", "Interpretation already confirmed or corrected")


@dataclass
class InterpretationCandidate:
    """A candidate interpretation from recognition."""

    id: UUID
    interpretation_id: UUID
    rank: int
    text: str
    confidence: float | None
    selected: bool = False
    created_at: datetime | None = None


@dataclass
class Interpretation:
    """AI-generated interpretation awaiting patient confirmation.

    Enforces critical invariant: interpretation must remain UNVERIFIED
    until patient explicitly confirms or corrects it. Only then can
    a doctor-facing message be created.
    """

    id: UUID
    recognition_id: UUID
    status: InterpretationStatus
    model_version_id: UUID | None
    quality_status: str  # "ACCEPTABLE", "LOW_QUALITY", "NEEDS_CLARIFICATION"
    candidates: list[InterpretationCandidate]
    created_at: datetime
    resolved_at: datetime | None = None

    def is_confirmable(self) -> bool:
        """Check if interpretation can be confirmed by patient."""
        return (
            self.status == InterpretationStatus.UNVERIFIED
            and self.quality_status == "ACCEPTABLE"
            and len(self.candidates) > 0
        )

    def validate_candidate_belongs_to_this_interpretation(self, candidate_id: UUID) -> None:
        """Enforce candidate must belong to this interpretation."""
        if not any(c.id == candidate_id for c in self.candidates):
            raise CandidateNotFoundError()


@dataclass
class Confirmation:
    """Patient action confirming, correcting or rejecting an interpretation.

    Must occur in same transaction as message creation to enforce invariant.
    """

    id: UUID
    interpretation_id: UUID
    candidate_id: UUID | None
    session_id: UUID
    action: ConfirmationAction
    correction_text: str | None
    created_message_id: UUID | None
    created_at: datetime


@dataclass
class Message:
    """Conversation message with authoritative provenance.

    Every AI-backed message references a confirmation record.
    Patient corrections are labeled distinctly from AI candidates.
    Doctor messages require no confirmation.
    """

    id: UUID
    conversation_id: UUID
    sender_type: SenderType
    sender_session_id: UUID | None
    message_type: MessageType
    text: str
    provenance: MessageProvenance
    verification_status: VerificationStatus
    created_at: datetime
    confirmed_at: datetime | None = None
    revision_of_message_id: UUID | None = None
    confirmation_id: UUID | None = None
