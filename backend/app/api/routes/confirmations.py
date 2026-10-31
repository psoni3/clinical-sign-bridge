"""Confirmation/correction/rejection endpoints (M1-007).

Critical invariant: Confirmation, message creation, interpretation state change,
and audit event must all occur in ONE database transaction. Database failure leaves
no partial confirmed message.
"""

from collections.abc import Generator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.helpers.validation import (
    validate_consultation_active,
    validate_session_active,
    validate_session_belongs_to_consultation,
    validate_session_matches_resource,
)
from app.api.schemas import (
    ClarificationRequestRequest,
    ClarificationRequestResponse,
    ConfirmInterpretationRequest,
    ConfirmInterpretationResponse,
    CorrectInterpretationRequest,
    CorrectInterpretationResponse,
    RejectInterpretationRequest,
    RejectInterpretationResponse,
)
from app.config import get_settings
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import (
    AuditEvent,
)
from app.infrastructure.persistence.entities import (
    Confirmation as ConfirmationEntity,
)
from app.infrastructure.persistence.entities import (
    Conversation as ConversationEntity,
)
from app.infrastructure.persistence.entities import (
    Interpretation as InterpretationEntity,
)
from app.infrastructure.persistence.entities import (
    InterpretationCandidate as CandidateEntity,
)
from app.infrastructure.persistence.entities import (
    Message as MessageEntity,
)
from app.infrastructure.persistence.entities import (
    Recognition as RecognitionEntity,
)
from app.infrastructure.persistence.outbox import enqueue_notification

router = APIRouter(prefix="/api/v1", tags=["interpretations"])


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    database_url = get_settings().database_url.get_secret_value()
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session


@router.post(
    "/interpretations/{interpretation_id}/confirm",
    response_model=ConfirmInterpretationResponse,
    status_code=201,
)
async def confirm_interpretation(
    interpretation_id: UUID,
    request: ConfirmInterpretationRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> ConfirmInterpretationResponse:
    """Patient confirms AI candidate (M1-007).

    Validates:
    - Interpretation exists and is UNVERIFIED with ACCEPTABLE quality
    - Candidate belongs to interpretation
    - Session is active, not expired, not revoked
    - Session belongs to consultation
    - Consultation is active, not expired, not completed
    - Idempotency: same idempotency_key returns same confirmation
    - All changes transactional (confirmation + message + state + audit)
    """
    now = datetime.now(UTC)

    try:
        interp = db.execute(
            select(InterpretationEntity).where(InterpretationEntity.id == interpretation_id)
        ).scalar_one_or_none()

        if not interp:
            raise HTTPException(
                status_code=404,
                detail={"code": "INTERPRETATION_NOT_FOUND"},
            )

        existing_request = db.execute(
            select(ConfirmationEntity).where(
                ConfirmationEntity.interpretation_id == interpretation_id,
                ConfirmationEntity.idempotency_key == request.idempotency_key,
            )
        ).scalar_one_or_none()
        if existing_request is not None:
            if (
                existing_request.action != "CONFIRM"
                or existing_request.candidate_id != request.candidate_id
                or existing_request.session_id != request.session_id
            ):
                raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT"})
            retry_session = validate_session_active(
                ConsultationSessionRepository(db).get_by_id(request.session_id), now
            )
            if retry_session.session_type != "PATIENT_DEVICE":
                raise HTTPException(
                    status_code=403,
                    detail={"code": "PATIENT_SESSION_REQUIRED"},
                )
            recognition = db.get(RecognitionEntity, interp.recognition_id)
            if recognition is None:
                raise HTTPException(status_code=404, detail={"code": "RECOGNITION_NOT_FOUND"})
            validate_session_belongs_to_consultation(retry_session, recognition.consultation_id)
            validate_session_matches_resource(retry_session, recognition.session_id)
            validate_consultation_active(
                ConsultationRepository(db).get_by_id(recognition.consultation_id), now
            )
            if existing_request.created_message_id is None:
                raise HTTPException(
                    status_code=500,
                    detail={"code": "CONFIRMATION_MESSAGE_MISSING"},
                )
            return ConfirmInterpretationResponse(
                confirmation_id=existing_request.id,
                interpretation_id=interpretation_id,
                created_message_id=existing_request.created_message_id,
                created_at=existing_request.created_at,
            )

        if interp.status != "UNVERIFIED":
            raise HTTPException(
                status_code=409,
                detail={"code": "INTERPRETATION_NOT_CONFIRMABLE"},
            )

        if interp.quality_status != "ACCEPTABLE":
            raise HTTPException(
                status_code=409,
                detail={"code": "INTERPRETATION_NOT_CONFIRMABLE"},
            )

        candidate = db.execute(
            select(CandidateEntity).where(
                (CandidateEntity.id == request.candidate_id)
                & (CandidateEntity.interpretation_id == interpretation_id)
            )
        ).scalar_one_or_none()

        if not candidate:
            raise HTTPException(
                status_code=400,
                detail={"code": "CANDIDATE_NOT_FOUND"},
            )

        # Validate session active and belongs to consultation
        session_repo = ConsultationSessionRepository(db)
        session = session_repo.get_by_id(request.session_id)
        session = validate_session_active(session, now)
        if session.session_type != "PATIENT_DEVICE":
            raise HTTPException(status_code=403, detail={"code": "PATIENT_SESSION_REQUIRED"})

        # Get recognition to find consultation
        recognition = db.execute(
            select(RecognitionEntity).where(RecognitionEntity.id == interp.recognition_id)
        ).scalar_one_or_none()

        if recognition is None:
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            )

        validate_session_belongs_to_consultation(session, recognition.consultation_id)
        validate_session_matches_resource(session, recognition.session_id)

        # Validate consultation active
        cons_repo = ConsultationRepository(db)
        consultation = cons_repo.get_by_id(recognition.consultation_id)
        consultation = validate_consultation_active(consultation, now)

        # Atomic transaction
        try:
            confirmation_id = uuid4()
            message_id = uuid4()

            confirmation = ConfirmationEntity(
                id=confirmation_id,
                interpretation_id=interpretation_id,
                candidate_id=request.candidate_id,
                session_id=request.session_id,
                action="CONFIRM",
                correction_text=None,
                created_message_id=message_id,
                idempotency_key=request.idempotency_key,
                created_at=now,
            )
            db.add(confirmation)

            conversation = db.execute(
                select(ConversationEntity).where(
                    ConversationEntity.consultation_id == recognition.consultation_id
                )
            ).scalar_one_or_none()

            if not conversation:
                conversation = ConversationEntity(
                    id=uuid4(),
                    consultation_id=recognition.consultation_id,
                    status="ACTIVE",
                    created_at=now,
                )
                db.add(conversation)
                db.flush()

            message = MessageEntity(
                id=message_id,
                conversation_id=conversation.id,
                sender_type="PATIENT",
                sender_session_id=request.session_id,
                message_type="AI_TRANSLATION",
                text=candidate.text,
                provenance="AI_CANDIDATE_CONFIRMED",
                verification_status="PATIENT_CONFIRMED",
                confirmation_id=confirmation_id,
                created_at=now,
                confirmed_at=now,
            )
            db.add(message)
            candidate.selected = True

            interp.status = "CONFIRMED"
            interp.resolved_at = now
            db.merge(interp)

            audit = AuditEvent(
                id=uuid4(),
                consultation_id=recognition.consultation_id,
                session_id=request.session_id,
                action="CONFIRM_INTERPRETATION",
                resource_type="Interpretation",
                resource_id=interpretation_id,
                outcome="SUCCESS",
                correlation_id=getattr(req.state, "correlation_id", ""),
                event_metadata={
                    "confirmation_id": str(confirmation_id),
                    "candidate_id": str(request.candidate_id),
                },
                created_at=now,
            )
            db.add(audit)

            enqueue_notification(
                db,
                recognition.consultation_id,
                "confirmation.received",
                message_id,
                now,
            )
            db.commit()

            return ConfirmInterpretationResponse(
                confirmation_id=confirmation_id,
                interpretation_id=interpretation_id,
                created_message_id=message_id,
                created_at=now,
            )

        except SQLAlchemyError as error:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            ) from error

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from error


@router.post(
    "/interpretations/{interpretation_id}/correct",
    response_model=CorrectInterpretationResponse,
    status_code=201,
)
async def correct_interpretation(
    interpretation_id: UUID,
    request: CorrectInterpretationRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> CorrectInterpretationResponse:
    """Patient corrects AI interpretation (M1-007)."""
    now = datetime.now(UTC)

    try:
        interp = db.execute(
            select(InterpretationEntity).where(InterpretationEntity.id == interpretation_id)
        ).scalar_one_or_none()

        if not interp:
            raise HTTPException(
                status_code=404,
                detail={"code": "INTERPRETATION_NOT_FOUND"},
            )

        existing_request = db.execute(
            select(ConfirmationEntity).where(
                ConfirmationEntity.interpretation_id == interpretation_id,
                ConfirmationEntity.idempotency_key == request.idempotency_key,
            )
        ).scalar_one_or_none()
        if existing_request is not None:
            if (
                existing_request.action != "CORRECT"
                or existing_request.candidate_id != request.candidate_id
                or existing_request.correction_text != request.corrected_text
                or existing_request.session_id != request.session_id
            ):
                raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT"})
            retry_session = validate_session_active(
                ConsultationSessionRepository(db).get_by_id(request.session_id), now
            )
            if retry_session.session_type != "PATIENT_DEVICE":
                raise HTTPException(
                    status_code=403,
                    detail={"code": "PATIENT_SESSION_REQUIRED"},
                )
            recognition = db.get(RecognitionEntity, interp.recognition_id)
            if recognition is None:
                raise HTTPException(status_code=404, detail={"code": "RECOGNITION_NOT_FOUND"})
            validate_session_belongs_to_consultation(retry_session, recognition.consultation_id)
            validate_session_matches_resource(retry_session, recognition.session_id)
            validate_consultation_active(
                ConsultationRepository(db).get_by_id(recognition.consultation_id), now
            )
            if existing_request.created_message_id is None:
                raise HTTPException(
                    status_code=500,
                    detail={"code": "CONFIRMATION_MESSAGE_MISSING"},
                )
            return CorrectInterpretationResponse(
                confirmation_id=existing_request.id,
                interpretation_id=interpretation_id,
                created_message_id=existing_request.created_message_id,
                created_at=existing_request.created_at,
            )

        if interp.status != "UNVERIFIED" or interp.quality_status != "ACCEPTABLE":
            raise HTTPException(
                status_code=409,
                detail={"code": "INTERPRETATION_NOT_CONFIRMABLE"},
            )

        # Validate session active
        session_repo = ConsultationSessionRepository(db)
        session = session_repo.get_by_id(request.session_id)
        session = validate_session_active(session, now)
        if session.session_type != "PATIENT_DEVICE":
            raise HTTPException(status_code=403, detail={"code": "PATIENT_SESSION_REQUIRED"})

        # Get recognition and validate session belongs to consultation
        recognition = db.execute(
            select(RecognitionEntity).where(RecognitionEntity.id == interp.recognition_id)
        ).scalar_one_or_none()

        if recognition is None:
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            )

        validate_session_belongs_to_consultation(session, recognition.consultation_id)
        validate_session_matches_resource(session, recognition.session_id)

        candidate = db.execute(
            select(CandidateEntity).where(
                CandidateEntity.id == request.candidate_id,
                CandidateEntity.interpretation_id == interpretation_id,
            )
        ).scalar_one_or_none()
        if candidate is None:
            raise HTTPException(status_code=400, detail={"code": "CANDIDATE_NOT_FOUND"})

        # Validate consultation active
        cons_repo = ConsultationRepository(db)
        consultation = cons_repo.get_by_id(recognition.consultation_id)
        consultation = validate_consultation_active(consultation, now)

        try:
            confirmation_id = uuid4()
            message_id = uuid4()

            confirmation = ConfirmationEntity(
                id=confirmation_id,
                interpretation_id=interpretation_id,
                candidate_id=request.candidate_id,
                session_id=request.session_id,
                action="CORRECT",
                correction_text=request.corrected_text,
                created_message_id=message_id,
                idempotency_key=request.idempotency_key,
                created_at=now,
            )
            db.add(confirmation)

            conversation = db.execute(
                select(ConversationEntity).where(
                    ConversationEntity.consultation_id == recognition.consultation_id
                )
            ).scalar_one_or_none()

            if not conversation:
                conversation = ConversationEntity(
                    id=uuid4(),
                    consultation_id=recognition.consultation_id,
                    status="ACTIVE",
                    created_at=now,
                )
                db.add(conversation)
                db.flush()

            message = MessageEntity(
                id=message_id,
                conversation_id=conversation.id,
                sender_type="PATIENT",
                sender_session_id=request.session_id,
                message_type="AI_TRANSLATION",
                text=request.corrected_text,
                provenance="PATIENT_CORRECTED",
                verification_status="PATIENT_CONFIRMED",
                confirmation_id=confirmation_id,
                created_at=now,
                confirmed_at=now,
            )
            db.add(message)

            interp.status = "CONFIRMED"
            interp.resolved_at = now
            db.merge(interp)

            audit = AuditEvent(
                id=uuid4(),
                consultation_id=recognition.consultation_id,
                session_id=request.session_id,
                action="CORRECT_INTERPRETATION",
                resource_type="Interpretation",
                resource_id=interpretation_id,
                outcome="SUCCESS",
                correlation_id=getattr(req.state, "correlation_id", ""),
                event_metadata={"confirmation_id": str(confirmation_id)},
                created_at=now,
            )
            db.add(audit)

            enqueue_notification(
                db,
                recognition.consultation_id,
                "confirmation.received",
                message_id,
                now,
            )
            db.commit()

            return CorrectInterpretationResponse(
                confirmation_id=confirmation_id,
                interpretation_id=interpretation_id,
                created_message_id=message_id,
                created_at=now,
            )

        except SQLAlchemyError as error:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            ) from error

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from error


@router.post(
    "/interpretations/{interpretation_id}/reject",
    response_model=RejectInterpretationResponse,
    status_code=200,
)
async def reject_interpretation(
    interpretation_id: UUID,
    request: RejectInterpretationRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> RejectInterpretationResponse:
    """Patient rejects all candidates (M1-007)."""
    now = datetime.now(UTC)

    try:
        interp = db.execute(
            select(InterpretationEntity).where(InterpretationEntity.id == interpretation_id)
        ).scalar_one_or_none()

        if not interp:
            raise HTTPException(
                status_code=404,
                detail={"code": "INTERPRETATION_NOT_FOUND"},
            )

        if interp.status != "UNVERIFIED":
            raise HTTPException(
                status_code=409,
                detail={"code": "INTERPRETATION_NOT_CONFIRMABLE"},
            )

        # Validate session active
        session_repo = ConsultationSessionRepository(db)
        session = session_repo.get_by_id(request.session_id)
        session = validate_session_active(session, now)
        if session.session_type != "PATIENT_DEVICE":
            raise HTTPException(status_code=403, detail={"code": "PATIENT_SESSION_REQUIRED"})

        # Get recognition and validate session
        recognition = db.execute(
            select(RecognitionEntity).where(RecognitionEntity.id == interp.recognition_id)
        ).scalar_one_or_none()

        if recognition is None:
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            )

        validate_session_belongs_to_consultation(session, recognition.consultation_id)
        validate_session_matches_resource(session, recognition.session_id)

        # Validate consultation active
        cons_repo = ConsultationRepository(db)
        consultation = cons_repo.get_by_id(recognition.consultation_id)
        consultation = validate_consultation_active(consultation, now)

        try:
            confirmation_id = uuid4()

            confirmation = ConfirmationEntity(
                id=confirmation_id,
                interpretation_id=interpretation_id,
                candidate_id=None,
                session_id=request.session_id,
                action="REJECT",
                correction_text=request.reason,
                created_message_id=None,
                created_at=now,
            )
            db.add(confirmation)

            interp.status = "REJECTED"
            interp.resolved_at = now
            db.merge(interp)

            audit = AuditEvent(
                id=uuid4(),
                consultation_id=recognition.consultation_id,
                session_id=request.session_id,
                action="REJECT_INTERPRETATION",
                resource_type="Interpretation",
                resource_id=interpretation_id,
                outcome="SUCCESS",
                correlation_id="",
                event_metadata={"reason": request.reason},
                created_at=now,
            )
            db.add(audit)

            enqueue_notification(
                db,
                recognition.consultation_id,
                "confirmation.received",
                interpretation_id,
                now,
            )
            db.commit()

            return RejectInterpretationResponse(
                confirmation_id=confirmation_id,
                interpretation_id=interpretation_id,
                status=interp.status,
            )

        except SQLAlchemyError as error:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            ) from error

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from error


@router.post(
    "/interpretations/{interpretation_id}/request-clarification",
    response_model=ClarificationRequestResponse,
    status_code=200,
)
async def request_clarification(
    interpretation_id: UUID,
    request: ClarificationRequestRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> ClarificationRequestResponse:
    """Request patient re-sign or clarification (M1-007)."""
    now = datetime.now(UTC)

    try:
        interp = db.execute(
            select(InterpretationEntity).where(InterpretationEntity.id == interpretation_id)
        ).scalar_one_or_none()

        if not interp:
            raise HTTPException(
                status_code=404,
                detail={"code": "INTERPRETATION_NOT_FOUND"},
            )

        # Validate session active
        session_repo = ConsultationSessionRepository(db)
        session = session_repo.get_by_id(request.session_id)
        session = validate_session_active(session, now)
        if session.session_type != "PATIENT_DEVICE":
            raise HTTPException(status_code=403, detail={"code": "PATIENT_SESSION_REQUIRED"})

        # Get recognition and validate
        recognition = db.execute(
            select(RecognitionEntity).where(RecognitionEntity.id == interp.recognition_id)
        ).scalar_one_or_none()

        if recognition is None:
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            )

        validate_session_belongs_to_consultation(session, recognition.consultation_id)
        validate_session_matches_resource(session, recognition.session_id)

        # Validate consultation active
        cons_repo = ConsultationRepository(db)
        consultation = cons_repo.get_by_id(recognition.consultation_id)
        consultation = validate_consultation_active(consultation, now)

        try:
            confirmation_id = uuid4()

            confirmation = ConfirmationEntity(
                id=confirmation_id,
                interpretation_id=interpretation_id,
                candidate_id=None,
                session_id=request.session_id,
                action="CLARIFICATION_REQUESTED",
                correction_text=None,
                created_message_id=None,
                created_at=now,
            )
            db.add(confirmation)

            interp.status = "NEEDS_CLARIFICATION"
            interp.resolved_at = now
            db.merge(interp)
            db.add(
                AuditEvent(
                    id=uuid4(),
                    consultation_id=recognition.consultation_id,
                    session_id=request.session_id,
                    action="REQUEST_CLARIFICATION",
                    resource_type="Interpretation",
                    resource_id=interpretation_id,
                    outcome="SUCCESS",
                    correlation_id=getattr(req.state, "correlation_id", ""),
                    event_metadata={},
                    created_at=now,
                )
            )

            enqueue_notification(
                db,
                recognition.consultation_id,
                "interpretation.available",
                interpretation_id,
                now,
            )
            db.commit()

            return ClarificationRequestResponse(
                confirmation_id=confirmation_id,
                status="NEEDS_CLARIFICATION",
            )

        except SQLAlchemyError as error:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR"},
            ) from error

    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from error
