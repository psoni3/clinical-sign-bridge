"""Consultation message persistence and history endpoints."""

from collections.abc import Generator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.helpers.validation import (
    validate_consultation_active,
    validate_session_active,
    validate_session_belongs_to_consultation,
)
from app.api.schemas import (
    AcknowledgeMessageRequest,
    AcknowledgeMessageResponse,
    GetMessagesResponse,
    MessageDto,
    PostMessageRequest,
    PostMessageResponse,
)
from app.api.websocket_handler import connection_manager
from app.config import get_settings
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import AuditEvent, Conversation, Message

router = APIRouter(prefix="/api/v1", tags=["messages"])


def get_db() -> Generator[Session, None, None]:
    database_url = get_settings().database_url.get_secret_value()
    with Session(get_engine(database_url)) as session:
        yield session


def _validate_session(
    db: Session,
    session_id: UUID,
    consultation_id: UUID,
    expected_type: str | None = None,
) -> None:
    session = ConsultationSessionRepository(db).get_by_id(session_id)
    session = validate_session_active(session)
    validate_session_belongs_to_consultation(session, consultation_id)
    if expected_type is not None and session.session_type != expected_type:
        raise HTTPException(status_code=403, detail={"code": "SESSION_TYPE_MISMATCH"})


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=PostMessageResponse,
    status_code=201,
)
async def post_message(
    conversation_id: UUID,
    request: PostMessageRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> PostMessageResponse:
    now = datetime.now(UTC)
    try:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail={"code": "CONVERSATION_NOT_FOUND"})
        consultation = ConsultationRepository(db).get_by_id(conversation.consultation_id)
        validate_consultation_active(consultation, now)
        sender_session_type = "DOCTOR_VIEW" if request.sender_type == "DOCTOR" else "PATIENT_DEVICE"
        _validate_session(db, request.session_id, conversation.consultation_id, sender_session_type)

        patient_message = request.sender_type == "PATIENT"
        provenance = (
            "DOCTOR_AUTHORED"
            if request.sender_type == "DOCTOR"
            else (
                "PATIENT_VISUAL_RESPONSE"
                if request.message_type == "VISUAL_RESPONSE"
                else "PATIENT_TYPED"
            )
        )
        message = Message(
            id=uuid4(),
            conversation_id=conversation_id,
            sender_type=request.sender_type,
            sender_session_id=request.session_id,
            message_type=request.message_type,
            text=request.text,
            provenance=provenance,
            verification_status="PATIENT_CONFIRMED" if patient_message else "NOT_APPLICABLE",
            created_at=now,
            confirmed_at=now if patient_message else None,
        )
        db.add(message)
        db.add(
            AuditEvent(
                id=uuid4(),
                consultation_id=conversation.consultation_id,
                session_id=request.session_id,
                action="CREATE_MESSAGE",
                resource_type="Message",
                resource_id=message.id,
                outcome="SUCCESS",
                correlation_id=getattr(req.state, "correlation_id", ""),
                event_metadata={
                    "sender_type": request.sender_type,
                    "message_type": request.message_type,
                    "provenance": provenance,
                },
                created_at=now,
            )
        )
        db.commit()
        await connection_manager.publish(
            conversation.consultation_id,
            "message.received",
            message.id,
        )
        return PostMessageResponse(
            message_id=message.id,
            conversation_id=message.conversation_id,
            created_at=message.created_at,
        )
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR"}) from None


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=GetMessagesResponse,
)
def get_messages(
    conversation_id: UUID,
    session_id: UUID = Query(..., alias="sessionId"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> GetMessagesResponse:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail={"code": "CONVERSATION_NOT_FOUND"})
    consultation = ConsultationRepository(db).get_by_id(conversation.consultation_id)
    if consultation is None:
        raise HTTPException(status_code=404, detail={"code": "CONSULTATION_NOT_FOUND"})
    if consultation.status != "COMPLETED":
        validate_consultation_active(consultation)
    _validate_session(db, session_id, conversation.consultation_id)
    total = (
        db.scalar(select(func.count(Message.id)).where(Message.conversation_id == conversation_id))
        or 0
    )
    rows = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at), desc(Message.id))
        .offset(offset)
        .limit(limit)
    ).all()
    return GetMessagesResponse(
        messages=[
            MessageDto(
                message_id=row.id,
                sender_type=row.sender_type,
                message_type=row.message_type,
                text=row.text,
                provenance=row.provenance,
                verification_status=row.verification_status,
                created_at=row.created_at,
                confirmed_at=row.confirmed_at,
                acknowledged_at=row.acknowledged_at,
            )
            for row in rows
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post(
    "/messages/{message_id}/acknowledge",
    response_model=AcknowledgeMessageResponse,
)
async def acknowledge_message(
    message_id: UUID,
    request: AcknowledgeMessageRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> AcknowledgeMessageResponse:
    now = datetime.now(UTC)
    try:
        message = db.get(Message, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail={"code": "MESSAGE_NOT_FOUND"})
        conversation = db.get(Conversation, message.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail={"code": "CONVERSATION_NOT_FOUND"})
        validate_consultation_active(
            ConsultationRepository(db).get_by_id(conversation.consultation_id), now
        )
        _validate_session(
            db,
            request.session_id,
            conversation.consultation_id,
            expected_type="DOCTOR_VIEW",
        )
        message.acknowledged_at = now
        db.add(
            AuditEvent(
                id=uuid4(),
                consultation_id=conversation.consultation_id,
                session_id=request.session_id,
                action="ACKNOWLEDGE_MESSAGE",
                resource_type="Message",
                resource_id=message_id,
                outcome="SUCCESS",
                correlation_id=getattr(req.state, "correlation_id", ""),
                event_metadata={},
                created_at=now,
            )
        )
        db.commit()
        await connection_manager.publish(
            conversation.consultation_id,
            "message.acknowledged",
            message_id,
        )
        return AcknowledgeMessageResponse(message_id=message_id, acknowledged_at=now)
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR"}) from None
