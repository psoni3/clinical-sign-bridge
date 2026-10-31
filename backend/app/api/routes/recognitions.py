"""Bounded video upload and deterministic mock recognition."""

from collections.abc import Generator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.helpers.validation import (
    validate_consultation_active,
    validate_session_active,
    validate_session_belongs_to_consultation,
    validate_session_matches_resource,
)
from app.api.schemas import RecognitionResponse
from app.api.websocket_handler import connection_manager
from app.config import get_settings
from app.domain.recognition import RecognitionConstraints
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import (
    AuditEvent,
    Consent,
    Interpretation,
    ModelVersion,
    Recognition,
)

router = APIRouter(prefix="/api/v1", tags=["recognitions"])
LIVE_CONSENT = "PROCESSING_FOR_LIVE_TRANSLATION"
MOCK_MODEL_NAME = "deterministic-mock"
MOCK_MODEL_VERSION = "mock-v1"


def get_db() -> Generator[Session, None, None]:
    database_url = get_settings().database_url.get_secret_value()
    with Session(get_engine(database_url)) as session:
        yield session


async def _read_limited_upload(video: UploadFile) -> tuple[int, bytes]:
    """Read at most the configured limit plus one byte, retaining only the header."""
    total = 0
    header = bytearray()
    while chunk := await video.read(64 * 1024):
        total += len(chunk)
        if total > RecognitionConstraints.MAX_SIZE_BYTES:
            raise HTTPException(status_code=413, detail={"code": "SEGMENT_TOO_LARGE"})
        if len(header) < 16:
            header.extend(chunk[: 16 - len(header)])
    if total == 0:
        raise HTTPException(status_code=400, detail={"code": "CAMERA_MEDIA_INVALID"})
    return total, bytes(header)


def _has_supported_container_signature(content_type: str | None, header: bytes) -> bool:
    if content_type == "video/mp4":
        return len(header) >= 8 and header[4:8] == b"ftyp"
    if content_type == "video/webm":
        return header.startswith(b"\x1a\x45\xdf\xa3")
    return False


@router.post(
    "/consultations/{consultation_id}/recognitions",
    response_model=RecognitionResponse,
    status_code=201,
)
async def create_recognition(
    consultation_id: UUID,
    request: Request,
    video: UploadFile = File(...),
    client_request_id: str = Form(..., min_length=1, max_length=128, alias="clientRequestId"),
    session_id: UUID = Form(..., alias="sessionId"),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    now = datetime.now(UTC)
    try:
        consultation = ConsultationRepository(db).get_by_id(consultation_id)
        validate_consultation_active(consultation, now)
        patient_session = ConsultationSessionRepository(db).get_by_id(session_id)
        patient_session = validate_session_active(patient_session, now)
        validate_session_belongs_to_consultation(patient_session, consultation_id)
        if patient_session.session_type != "PATIENT_DEVICE":
            raise HTTPException(
                status_code=403,
                detail={"code": "PATIENT_SESSION_REQUIRED"},
            )

        consent_exists = db.execute(
            select(Consent.id).where(
                Consent.consultation_id == consultation_id,
                Consent.session_id == session_id,
                Consent.consent_type == LIVE_CONSENT,
                Consent.accepted.is_(True),
                Consent.withdrawn_at.is_(None),
            )
        ).scalar_one_or_none()
        if consent_exists is None:
            raise HTTPException(status_code=409, detail={"code": "CONSENT_REQUIRED"})

        existing = db.execute(
            select(Recognition).where(
                Recognition.consultation_id == consultation_id,
                Recognition.session_id == session_id,
                Recognition.client_request_id == client_request_id,
            )
        ).scalar_one_or_none()
        if existing:
            return RecognitionResponse(
                recognition_id=existing.id,
                status=existing.status,
                created_at=existing.created_at,
                interpretation_id=db.execute(
                    select(Interpretation.id).where(Interpretation.recognition_id == existing.id)
                ).scalar_one_or_none(),
            )

        total_size, header = await _read_limited_upload(video)
        if not _has_supported_container_signature(video.content_type, header):
            raise HTTPException(status_code=400, detail={"code": "CAMERA_MEDIA_INVALID"})

        model = db.execute(
            select(ModelVersion).where(
                ModelVersion.model_name == MOCK_MODEL_NAME,
                ModelVersion.version == MOCK_MODEL_VERSION,
            )
        ).scalar_one_or_none()
        if model is None:
            model = ModelVersion(
                id=uuid4(),
                model_name=MOCK_MODEL_NAME,
                version=MOCK_MODEL_VERSION,
                metrics={"recognition": "not implemented"},
                approval_status="MOCK_ONLY",
                created_at=now,
            )
            db.add(model)
            db.flush()

        recognition_id = uuid4()
        recognition = Recognition(
            id=recognition_id,
            consultation_id=consultation_id,
            session_id=session_id,
            client_request_id=client_request_id,
            status="UNSUPPORTED",
            media_metadata={
                "size_bytes": total_size,
                "container": video.content_type,
                "duration_ms": None,
            },
            temporary_media_reference=None,
            created_at=now,
            completed_at=now,
        )
        db.add(recognition)
        db.flush()

        interpretation_id = uuid4()
        db.add(
            Interpretation(
                id=interpretation_id,
                recognition_id=recognition_id,
                status="NEEDS_CLARIFICATION",
                model_version_id=model.id,
                quality_status="NEEDS_CLARIFICATION",
                created_at=now,
            )
        )
        db.add(
            AuditEvent(
                id=uuid4(),
                consultation_id=consultation_id,
                session_id=session_id,
                action="RECOGNITION_UNSUPPORTED",
                resource_type="Recognition",
                resource_id=recognition_id,
                outcome="SUCCESS",
                correlation_id=getattr(request.state, "correlation_id", ""),
                event_metadata={
                    "size_bytes": total_size,
                    "interpretation_id": str(interpretation_id),
                },
                created_at=now,
            )
        )
        db.commit()
        await connection_manager.publish(
            consultation_id,
            "recognition.completed",
            recognition_id,
        )
        return RecognitionResponse(
            recognition_id=recognition_id,
            status="UNSUPPORTED",
            created_at=now,
            interpretation_id=interpretation_id,
        )
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR"}) from None
    finally:
        await video.close()


@router.get("/recognitions/{recognition_id}", response_model=RecognitionResponse)
def get_recognition(
    recognition_id: UUID,
    session_id: UUID = Query(..., alias="sessionId"),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    recognition = db.execute(
        select(Recognition).where(Recognition.id == recognition_id)
    ).scalar_one_or_none()
    if recognition is None:
        raise HTTPException(status_code=404, detail={"code": "RECOGNITION_NOT_FOUND"})
    session = validate_session_active(ConsultationSessionRepository(db).get_by_id(session_id))
    validate_session_belongs_to_consultation(session, recognition.consultation_id)
    validate_session_matches_resource(session, recognition.session_id)
    interpretation_id = db.execute(
        select(Interpretation.id).where(Interpretation.recognition_id == recognition_id)
    ).scalar_one_or_none()
    return RecognitionResponse(
        recognition_id=recognition.id,
        status=recognition.status,
        created_at=recognition.created_at,
        interpretation_id=interpretation_id,
    )
