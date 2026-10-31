"""SQLAlchemy persistence models for the PostgreSQL workflow schema."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON_DOCUMENT}


class Consultation(Base):
    __tablename__ = "consultations"
    __table_args__ = (Index("ix_consultations_status_expires_at", "status", "expires_at"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    display_label: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    join_code_hash: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConsultationSession(Base):
    __tablename__ = "consultation_sessions"
    __table_args__ = (
        Index("ix_sessions_consultation_expires_at", "consultation_id", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultations.id"), nullable=False)
    session_type: Mapped[str] = mapped_column(String(32), nullable=False)
    session_token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    device_label: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_consultation_id", "consultation_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultations.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created_at", "conversation_id", "created_at"),
        UniqueConstraint("confirmation_id", name="uq_messages_confirmation_id"),
        CheckConstraint(
            "message_type != 'AI_TRANSLATION' OR confirmation_id IS NOT NULL",
            name="ck_translation_message_has_confirmation",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    sender_type: Mapped[str] = mapped_column(String(32), nullable=False)
    sender_session_id: Mapped[UUID | None] = mapped_column(ForeignKey("consultation_sessions.id"))
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    revision_of_message_id: Mapped[UUID | None] = mapped_column(ForeignKey("messages.id"))
    confirmation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("confirmations.id", use_alter=True, name="fk_messages_confirmation_id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Recognition(Base):
    __tablename__ = "recognitions"
    __table_args__ = (
        UniqueConstraint(
            "consultation_id",
            "session_id",
            "client_request_id",
            name="uq_recognitions_consultation_session_request",
        ),
        Index("ix_recognitions_consultation_created_at", "consultation_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultations.id"), nullable=False)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("consultation_sessions.id"), nullable=False)
    client_request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    media_metadata: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    temporary_media_reference: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    dataset_version: Mapped[str | None] = mapped_column(String(128))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    approval_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Interpretation(Base):
    __tablename__ = "interpretations"
    __table_args__ = (Index("ix_interpretations_recognition_id", "recognition_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    recognition_id: Mapped[UUID] = mapped_column(ForeignKey("recognitions.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class InterpretationCandidate(Base):
    __tablename__ = "interpretation_candidates"
    __table_args__ = (
        UniqueConstraint("interpretation_id", "rank", name="uq_candidates_interpretation_rank"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    interpretation_id: Mapped[UUID] = mapped_column(
        ForeignKey("interpretations.id"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    selected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Confirmation(Base):
    __tablename__ = "confirmations"
    __table_args__ = (
        UniqueConstraint(
            "interpretation_id",
            "idempotency_key",
            name="uq_confirmations_interpretation_idempotency",
        ),
        Index("ix_confirmations_interpretation_created_at", "interpretation_id", "created_at"),
        Index(
            "uq_confirmations_interpretation_success",
            "interpretation_id",
            unique=True,
            postgresql_where=text("action IN ('CONFIRM', 'CORRECT')"),
            sqlite_where=text("action IN ('CONFIRM', 'CORRECT')"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    interpretation_id: Mapped[UUID] = mapped_column(
        ForeignKey("interpretations.id"), nullable=False
    )
    candidate_id: Mapped[UUID | None] = mapped_column(ForeignKey("interpretation_candidates.id"))
    session_id: Mapped[UUID] = mapped_column(ForeignKey("consultation_sessions.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    correction_text: Mapped[str | None] = mapped_column(Text)
    created_message_id: Mapped[UUID | None] = mapped_column(ForeignKey("messages.id"))
    idempotency_key: Mapped[UUID | None] = mapped_column(Uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Consent(Base):
    __tablename__ = "consents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultations.id"), nullable=False)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("consultation_sessions.id"), nullable=False)
    consent_type: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IslSign(Base):
    __tablename__ = "isl_signs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    gloss: Mapped[str] = mapped_column(Text, nullable=False)
    english_meaning: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    regional_variant: Mapped[str | None] = mapped_column(String(128))
    reference_media_uri: Mapped[str | None] = mapped_column(Text)
    approval_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_consultation_created_at", "consultation_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    consultation_id: Mapped[UUID | None] = mapped_column(ForeignKey("consultations.id"))
    session_id: Mapped[UUID | None] = mapped_column(ForeignKey("consultation_sessions.id"))
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_id: Mapped[UUID | None] = mapped_column(Uuid)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_DOCUMENT, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
