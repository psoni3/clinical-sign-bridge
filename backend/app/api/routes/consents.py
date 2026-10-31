"""Consultation consent endpoints."""

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
)
from app.api.schemas import CreateConsentRequest, CreateConsentResponse
from app.config import get_settings
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import AuditEvent
from app.infrastructure.persistence.entities import Consent as ConsentEntity

router = APIRouter(prefix="/api/v1", tags=["consents"])


def get_db() -> Generator[Session, None, None]:
    database_url = get_settings().database_url.get_secret_value()
    with Session(get_engine(database_url)) as session:
        yield session


@router.post(
    "/consultations/{consultation_id}/consents",
    response_model=CreateConsentResponse,
    status_code=201,
)
def create_consent(
    consultation_id: UUID,
    request: CreateConsentRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> CreateConsentResponse:
    now = datetime.now(UTC)
    try:
        consultation = ConsultationRepository(db).get_by_id(consultation_id)
        validate_consultation_active(consultation, now)
        session = ConsultationSessionRepository(db).get_by_id(request.session_id)
        session = validate_session_active(session, now)
        validate_session_belongs_to_consultation(session, consultation_id)
        if session.session_type != "PATIENT_DEVICE":
            raise HTTPException(
                status_code=403,
                detail={"code": "PATIENT_SESSION_REQUIRED"},
            )

        if not request.accepted:
            active_consents = db.scalars(
                select(ConsentEntity).where(
                    ConsentEntity.consultation_id == consultation_id,
                    ConsentEntity.session_id == request.session_id,
                    ConsentEntity.consent_type == request.consent_type,
                    ConsentEntity.accepted.is_(True),
                    ConsentEntity.withdrawn_at.is_(None),
                )
            ).all()
            for active_consent in active_consents:
                active_consent.withdrawn_at = now

        consent = ConsentEntity(
            id=uuid4(),
            consultation_id=consultation_id,
            session_id=request.session_id,
            consent_type=request.consent_type,
            policy_version=request.policy_version,
            accepted=request.accepted,
            captured_at=now,
        )
        db.add(consent)
        db.add(
            AuditEvent(
                id=uuid4(),
                consultation_id=consultation_id,
                session_id=request.session_id,
                action="RECORD_CONSENT" if request.accepted else "WITHDRAW_CONSENT",
                resource_type="Consent",
                resource_id=consent.id,
                outcome="SUCCESS",
                correlation_id=getattr(req.state, "correlation_id", ""),
                event_metadata={
                    "consent_type": request.consent_type,
                    "accepted": request.accepted,
                },
                created_at=now,
            )
        )
        db.commit()
        return CreateConsentResponse(
            consent_id=consent.id,
            consultation_id=consent.consultation_id,
            consent_type=consent.consent_type,
            policy_version=consent.policy_version,
            accepted=consent.accepted,
            captured_at=consent.captured_at,
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
