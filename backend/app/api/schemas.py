"""API request and response schemas (Pydantic)."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class ApiModel(BaseModel):
    """Keep Python names idiomatic while exposing the camelCase API contract."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


# ===== Consultation Schemas =====


class CreateConsultationRequest(ApiModel):
    """POST /consultations"""

    display_label: str | None = None
    expires_in_minutes: int = Field(default=120, gt=0, le=1440)


class CreateConsultationResponse(ApiModel):
    """Response to consultation creation."""

    model_config = ConfigDict(from_attributes=True)

    consultation_id: UUID
    join_code: str  # Plaintext only in creation response
    status: str
    expires_at: datetime


class JoinConsultationRequest(ApiModel):
    """POST /consultations/{consultationId}/join"""

    join_code: str
    device_label: str | None = None
    session_type: Literal["PATIENT_DEVICE", "DOCTOR_VIEW"] = "PATIENT_DEVICE"


class JoinConsultationResponse(ApiModel):
    """Response to join request."""

    model_config = ConfigDict(from_attributes=True)

    consultation_id: UUID
    session_id: UUID
    session_token: str  # Plaintext only in join response
    status: str


class GetConsultationResponse(ApiModel):
    """GET /consultations/{consultationId}"""

    model_config = ConfigDict(from_attributes=True)

    consultation_id: UUID
    conversation_id: UUID
    display_label: str | None
    status: str
    expires_at: datetime
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    is_expired: bool


class StartConsultationResponse(ApiModel):
    """POST /consultations/{consultationId}/start"""

    model_config = ConfigDict(from_attributes=True)

    consultation_id: UUID
    status: str
    started_at: datetime


class CompleteConsultationResponse(ApiModel):
    """POST /consultations/{consultationId}/complete"""

    model_config = ConfigDict(from_attributes=True)

    consultation_id: UUID
    status: str
    completed_at: datetime


# ===== Consent Schemas =====


class CreateConsentRequest(ApiModel):
    """POST /consultations/{consultationId}/consents"""

    session_id: UUID
    consent_type: Literal["PROCESSING_FOR_LIVE_TRANSLATION", "RESEARCH_VIDEO_COLLECTION"]
    policy_version: str = Field(min_length=1, max_length=64)
    accepted: bool


class CreateConsentResponse(ApiModel):
    """Response to consent creation."""

    model_config = ConfigDict(from_attributes=True)

    consent_id: UUID
    consultation_id: UUID
    consent_type: str
    policy_version: str
    accepted: bool
    captured_at: datetime


# ===== Recognition Schemas =====


class RecognitionResponse(ApiModel):
    """Response when recognition is queued."""

    model_config = ConfigDict(from_attributes=True)

    recognition_id: UUID
    status: str  # QUEUED, PROCESSING, COMPLETED, FAILED, UNSUPPORTED
    created_at: datetime
    interpretation_id: UUID | None = None


# ===== Interpretation Schemas =====


class CandidateDto(ApiModel):
    """Candidate interpretation."""

    model_config = ConfigDict(from_attributes=True)

    candidate_id: UUID
    rank: int
    text: str
    confidence: float | None = None


class GetInterpretationResponse(ApiModel):
    """GET /interpretations/{interpretationId}"""

    model_config = ConfigDict(from_attributes=True)

    interpretation_id: UUID
    status: str  # UNVERIFIED, CONFIRMED, REJECTED, NEEDS_CLARIFICATION
    model_version: str
    quality: dict[str, Any]  # {"status": "ACCEPTABLE|LOW_QUALITY|...", "messages": [...]}
    candidates: list[CandidateDto]
    available_actions: list[str]


# ===== Confirmation Schemas =====


class ConfirmInterpretationRequest(ApiModel):
    """POST /interpretations/{interpretationId}/confirm"""

    candidate_id: UUID
    session_id: UUID
    idempotency_key: UUID


class ConfirmInterpretationResponse(ApiModel):
    """Response to confirmation."""

    model_config = ConfigDict(from_attributes=True)

    confirmation_id: UUID
    interpretation_id: UUID
    created_message_id: UUID | None = None
    created_at: datetime


class CorrectInterpretationRequest(ApiModel):
    """POST /interpretations/{interpretationId}/correct"""

    candidate_id: UUID
    corrected_text: str = Field(min_length=1, max_length=1000)
    session_id: UUID
    idempotency_key: UUID


class CorrectInterpretationResponse(ApiModel):
    """Response to correction."""

    model_config = ConfigDict(from_attributes=True)

    confirmation_id: UUID
    interpretation_id: UUID
    created_message_id: UUID
    created_at: datetime


class RejectInterpretationRequest(ApiModel):
    """POST /interpretations/{interpretationId}/reject"""

    session_id: UUID
    reason: str = Field(default="NONE_OF_THESE")


class RejectInterpretationResponse(ApiModel):
    confirmation_id: UUID
    interpretation_id: UUID
    status: str


class ClarificationRequestRequest(ApiModel):
    """POST /interpretations/{interpretationId}/request-clarification"""

    session_id: UUID


class ClarificationRequestResponse(ApiModel):
    confirmation_id: UUID
    status: str


# ===== Message Schemas =====


class PostMessageRequest(ApiModel):
    """POST /conversations/{conversationId}/messages"""

    sender_type: Literal["DOCTOR", "PATIENT"]
    session_id: UUID
    message_type: Literal["TEXT", "VISUAL_RESPONSE"]
    text: str = Field(min_length=1, max_length=2000)


class PostMessageResponse(ApiModel):
    message_id: UUID
    conversation_id: UUID
    created_at: datetime


class MessageDto(ApiModel):
    """Message in conversation timeline."""

    model_config = ConfigDict(from_attributes=True)

    message_id: UUID
    sender_type: str
    message_type: str
    text: str
    provenance: str
    verification_status: str
    created_at: datetime
    confirmed_at: datetime | None = None
    acknowledged_at: datetime | None = None


class GetMessagesResponse(ApiModel):
    """GET /conversations/{conversationId}/messages"""

    messages: list[MessageDto]
    total: int
    offset: int
    limit: int


class AcknowledgeMessageRequest(ApiModel):
    """POST /messages/{messageId}/acknowledge (doctor only)"""

    session_id: UUID


class AcknowledgeMessageResponse(ApiModel):
    """Response to acknowledgement."""

    model_config = ConfigDict(from_attributes=True)

    message_id: UUID
    acknowledged_at: datetime


# ===== Doctor Schemas =====


class DoctorConsultationListItem(ApiModel):
    """Item in doctor consultation list."""

    model_config = ConfigDict(from_attributes=True)

    consultation_id: UUID
    display_label: str | None
    status: str
    created_at: datetime
    last_activity_at: datetime
    is_active: bool


class GetDoctorConsultationsResponse(ApiModel):
    """GET /doctors/consultations"""

    consultations: list[DoctorConsultationListItem]
    total: int


# ===== Error Response =====


class ErrorResponse(ApiModel):
    """Standardized error response."""

    code: str
    message: str
    correlation_id: str | None = None
    field_errors: dict[str, list[str]] | None = None
