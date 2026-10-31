"""Consultation API routes."""

import logging
from collections.abc import Generator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.schemas import (
    CompleteConsultationResponse,
    CreateConsultationRequest,
    CreateConsultationResponse,
    DoctorConsultationListItem,
    GetConsultationResponse,
    GetDoctorConsultationsResponse,
    JoinConsultationRequest,
    JoinConsultationResponse,
    StartConsultationResponse,
)
from app.api.websocket_handler import connection_manager
from app.application.consultation_service import (
    CompleteConsultationService,
    CreateConsultationService,
    JoinConsultationService,
    StartConsultationService,
)
from app.config import get_settings
from app.domain.security import JoinAttemptLimiter, verify_join_code
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import (
    AuditEvent,
    Conversation,
    Message,
)
from app.infrastructure.persistence.entities import (
    Consultation as ConsultationEntity,
)

router = APIRouter(prefix="/api/v1", tags=["consultations"])
join_attempt_limiter = JoinAttemptLimiter()
logger = logging.getLogger(__name__)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    database_url = get_settings().database_url.get_secret_value()
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session


@router.post("/consultations", response_model=CreateConsultationResponse, status_code=201)
async def create_consultation(
    request: CreateConsultationRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> CreateConsultationResponse:
    """Create consultation with temporary join code.

    POST /api/v1/consultations

    Returns plaintext join code only in creation response.
    Join code is stored as hash and cannot be retrieved later.
    """
    try:
        # Execute domain service
        consultation, join_code = CreateConsultationService.execute(
            display_label=request.display_label,
            expires_in_minutes=request.expires_in_minutes,
        )

        # Persist with hashed join code
        repo = ConsultationRepository(db)
        repo.save_with_join_code(consultation, join_code.plaintext)
        db.flush()
        conversation = Conversation(
            id=uuid4(),
            consultation_id=consultation.id,
            status="ACTIVE",
            created_at=consultation.created_at,
        )
        db.add(conversation)
        db.add(
            AuditEvent(
                id=uuid4(),
                consultation_id=consultation.id,
                action="CREATE_CONSULTATION",
                resource_type="Consultation",
                resource_id=consultation.id,
                outcome="SUCCESS",
                correlation_id=getattr(req.state, "correlation_id", ""),
                event_metadata={},
                created_at=consultation.created_at,
            )
        )
        db.commit()

        # Return with plaintext code (only in creation response)
        return CreateConsultationResponse(
            consultation_id=consultation.id,
            join_code=join_code.plaintext,
            status=consultation.status,  # Already string
            expires_at=consultation.expires_at,
        )

    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(
            "Could not create consultation; database error type=%s correlation_id=%s",
            type(exc).__name__,
            getattr(req.state, "correlation_id", ""),
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from None


@router.post(
    "/consultations/{consultation_id}/join",
    response_model=JoinConsultationResponse,
)
async def join_consultation(
    consultation_id: UUID,
    request: JoinConsultationRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> JoinConsultationResponse:
    """Join consultation and create patient session.

    POST /api/v1/consultations/{consultation_id}/join

    Validates join code and creates temporary session.
    Returns plaintext session token only in response.
    """
    try:
        repo = ConsultationRepository(db)
        consultation = repo.get_by_id(consultation_id)

        if not consultation:
            raise HTTPException(
                status_code=404,
                detail={"code": "CONSULTATION_NOT_FOUND"},
            )

        # Verify not expired
        if consultation.is_expired():
            raise HTTPException(
                status_code=410,
                detail={"code": "CONSULTATION_EXPIRED"},
            )

        client = req.client.host if req.client else "unknown"
        if not join_attempt_limiter.allow(consultation_id, client):
            raise HTTPException(
                status_code=429,
                detail={"code": "RATE_LIMITED"},
            )

        # Verify join code
        join_code_hash = repo.get_join_code_hash(consultation_id)
        if not join_code_hash or not verify_join_code(request.join_code, join_code_hash):
            raise HTTPException(
                status_code=401,
                detail={"code": "INVALID_JOIN_CODE"},
            )

        # Create session
        session, session_token = JoinConsultationService.execute(
            consultation=consultation,
            join_code_plaintext=request.join_code,
            join_code_hash=join_code_hash,
            device_label=request.device_label,
            session_type=request.session_type,
        )

        # Persist session with hashed token
        session_repo = ConsultationSessionRepository(db)
        session_repo.save(session, session_token.plaintext)

        # Update consultation status if still CREATED
        if request.session_type == "PATIENT_DEVICE" and consultation.status == "CREATED":
            consultation.status = "JOINED"
            repo.save(consultation)

        db.commit()

        return JoinConsultationResponse(
            consultation_id=consultation.id,
            session_id=session.id,
            session_token=session_token.plaintext,
            status=consultation.status,  # Already string
        )

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from None


@router.get(
    "/consultations/{consultation_id}",
    response_model=GetConsultationResponse,
)
async def get_consultation(
    consultation_id: UUID,
    req: Request,
    db: Session = Depends(get_db),
) -> GetConsultationResponse:
    """Get consultation status.

    GET /api/v1/consultations/{consultation_id}
    """
    try:
        repo = ConsultationRepository(db)
        consultation = repo.get_by_id(consultation_id)

        if not consultation:
            raise HTTPException(
                status_code=404,
                detail={"code": "CONSULTATION_NOT_FOUND"},
            )

        return GetConsultationResponse(
            consultation_id=consultation.id,
            display_label=consultation.display_label,
            status=consultation.status,  # Already string
            conversation_id=db.execute(
                select(Conversation.id).where(Conversation.consultation_id == consultation_id)
            ).scalar_one(),
            expires_at=consultation.expires_at,
            created_at=consultation.created_at,
            started_at=consultation.started_at,
            completed_at=consultation.completed_at,
            is_expired=consultation.is_expired(),
        )

    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from None


@router.post(
    "/consultations/{consultation_id}/start",
    response_model=StartConsultationResponse,
)
async def start_consultation(
    consultation_id: UUID,
    req: Request,
    db: Session = Depends(get_db),
) -> StartConsultationResponse:
    """Start consultation (doctor joins).

    POST /api/v1/consultations/{consultation_id}/start
    """
    try:
        repo = ConsultationRepository(db)
        consultation = repo.get_by_id(consultation_id)

        if not consultation:
            raise HTTPException(
                status_code=404,
                detail={"code": "CONSULTATION_NOT_FOUND"},
            )

        if consultation.is_expired():
            raise HTTPException(
                status_code=410,
                detail={"code": "CONSULTATION_EXPIRED"},
            )

        consultation = StartConsultationService.execute(consultation)
        repo.save(consultation)
        db.commit()
        await connection_manager.publish(
            consultation_id,
            "consultation.started",
            consultation_id,
        )

        return StartConsultationResponse(
            consultation_id=consultation.id,
            status=consultation.status,  # Already string
            started_at=consultation.started_at or datetime.now(UTC),
        )

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from None


@router.post(
    "/consultations/{consultation_id}/complete",
    response_model=CompleteConsultationResponse,
)
async def complete_consultation(
    consultation_id: UUID,
    req: Request,
    db: Session = Depends(get_db),
) -> CompleteConsultationResponse:
    """Complete consultation.

    POST /api/v1/consultations/{consultation_id}/complete
    """
    try:
        repo = ConsultationRepository(db)
        consultation = repo.get_by_id(consultation_id)

        if not consultation:
            raise HTTPException(
                status_code=404,
                detail={"code": "CONSULTATION_NOT_FOUND"},
            )

        if consultation.is_expired():
            raise HTTPException(
                status_code=410,
                detail={"code": "CONSULTATION_EXPIRED"},
            )

        consultation = CompleteConsultationService.execute(consultation)
        repo.save(consultation)
        db.commit()
        await connection_manager.publish(
            consultation_id,
            "consultation.completed",
            consultation_id,
        )

        return CompleteConsultationResponse(
            consultation_id=consultation.id,
            status=consultation.status,  # Already string
            completed_at=consultation.completed_at or datetime.now(UTC),
        )

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from None


@router.get(
    "/doctors/consultations",
    response_model=GetDoctorConsultationsResponse,
)
def list_doctor_consultations(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    include_completed: bool = Query(default=False, alias="includeCompleted"),
    db: Session = Depends(get_db),
) -> GetDoctorConsultationsResponse:
    now = datetime.now(UTC)
    active_filter = and_(
        ConsultationEntity.status.in_(("CREATED", "JOINED", "ACTIVE")),
        ConsultationEntity.expires_at > now,
    )
    consultation_filter = (
        or_(active_filter, ConsultationEntity.status == "COMPLETED")
        if include_completed
        else active_filter
    )
    last_activity = (
        select(func.max(Message.created_at))
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(Conversation.consultation_id == ConsultationEntity.id)
        .scalar_subquery()
    )
    rows = db.execute(
        select(
            ConsultationEntity,
            func.coalesce(last_activity, ConsultationEntity.created_at),
        )
        .where(consultation_filter)
        .order_by(ConsultationEntity.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    total = db.scalar(select(func.count(ConsultationEntity.id)).where(consultation_filter)) or 0
    items = [
        DoctorConsultationListItem(
            consultation_id=consultation.id,
            display_label=consultation.display_label,
            status=consultation.status,
            created_at=consultation.created_at,
            last_activity_at=last_activity_at,
            is_active=consultation.status in {"CREATED", "JOINED", "ACTIVE"},
        )
        for consultation, last_activity_at in rows
    ]
    return GetDoctorConsultationsResponse(consultations=items, total=total)
