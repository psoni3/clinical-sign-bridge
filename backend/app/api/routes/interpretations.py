"""Interpretation candidates API (M1-006)."""

from collections.abc import Generator
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.helpers.validation import (
    validate_consultation_active,
    validate_session_active,
    validate_session_belongs_to_consultation,
    validate_session_matches_resource,
)
from app.api.schemas import CandidateDto, GetInterpretationResponse
from app.config import get_settings
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import (
    Interpretation as InterpretationEntity,
)
from app.infrastructure.persistence.entities import (
    InterpretationCandidate as CandidateEntity,
)
from app.infrastructure.persistence.entities import (
    ModelVersion,
)
from app.infrastructure.persistence.entities import (
    Recognition as RecognitionEntity,
)

router = APIRouter(prefix="/api/v1", tags=["interpretations"])


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    database_url = get_settings().database_url.get_secret_value()
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session


@router.get(
    "/interpretations/{interpretation_id}",
    response_model=GetInterpretationResponse,
)
async def get_interpretation(
    interpretation_id: UUID,
    req: Request,
    session_id: UUID = Query(..., alias="sessionId"),
    db: Session = Depends(get_db),
) -> GetInterpretationResponse:
    """Get unverified interpretation and candidates (M1-006).

    GET /api/v1/interpretations/{interpretation_id}

    Returns:
    - status: UNVERIFIED (always initially)
    - candidates: list of AI-generated alternatives
    - quality: assessment (ACCEPTABLE, LOW_QUALITY, NEEDS_CLARIFICATION)
    - availableActions: patient can CONFIRM, CORRECT, REJECT, RE_SIGN, etc.

    INVARIANT: Interpretation always starts UNVERIFIED.
    No doctor sees this before patient confirmation.
    Validates consultation is active (not expired/completed).
    """
    now = datetime.now(UTC)

    try:
        # Fetch interpretation
        interp = db.execute(
            select(InterpretationEntity).where(InterpretationEntity.id == interpretation_id)
        ).scalar_one_or_none()

        if not interp:
            raise HTTPException(
                status_code=404,
                detail={"code": "INTERPRETATION_NOT_FOUND"},
            )

        # Fetch recognition to get consultation
        recognition = db.execute(
            select(RecognitionEntity).where(RecognitionEntity.id == interp.recognition_id)
        ).scalar_one_or_none()

        if not recognition:
            raise HTTPException(
                status_code=404,
                detail={"code": "RECOGNITION_NOT_FOUND"},
            )

        session = validate_session_active(
            ConsultationSessionRepository(db).get_by_id(session_id), now
        )
        validate_session_belongs_to_consultation(session, recognition.consultation_id)
        validate_session_matches_resource(session, recognition.session_id)

        # Validate consultation is active
        cons_repo = ConsultationRepository(db)
        consultation = cons_repo.get_by_id(recognition.consultation_id)
        consultation = validate_consultation_active(consultation, now)

        # Fetch candidates sorted by rank
        candidates_list = (
            db.execute(
                select(CandidateEntity)
                .where(CandidateEntity.interpretation_id == interpretation_id)
                .order_by(CandidateEntity.rank)
            )
            .scalars()
            .all()
        )

        candidates = [
            CandidateDto(
                candidate_id=c.id,
                rank=c.rank,
                text=c.text,
                confidence=c.confidence,
            )
            for c in candidates_list
        ]

        model = db.get(ModelVersion, interp.model_version_id) if interp.model_version_id else None
        available_actions = []
        if interp.status in {"UNVERIFIED", "NEEDS_CLARIFICATION"}:
            available_actions = [
                "REJECT_ALL",
                "RESIGN",
                "TYPE_MESSAGE",
                "REQUEST_INTERPRETER",
            ]
            if interp.quality_status == "ACCEPTABLE" and candidates:
                available_actions[0:0] = ["CONFIRM", "SELECT_ALTERNATIVE", "CORRECT"]

        return GetInterpretationResponse(
            interpretation_id=interp.id,
            status=interp.status,
            model_version=(model.version if model else "unknown"),
            quality={
                "status": interp.quality_status,
                "messages": [],
            },
            candidates=candidates,
            available_actions=available_actions,
        )

    except HTTPException:
        raise
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR"},
        ) from error
