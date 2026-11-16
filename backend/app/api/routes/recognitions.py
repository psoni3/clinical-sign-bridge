"""Bounded video upload and patient-scoped recognition orchestration."""

from collections.abc import Generator
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

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
from app.application.recognition_service import (
    RecognitionOrchestrationService,
    RecognitionOutcome,
    RecognitionPersistenceError,
    RecognitionRequestConflictError,
    RecognitionSubmission,
)
from app.config import get_settings
from app.domain.recognition import RecognitionConstraints
from app.infrastructure.ai_clients.video_inference_client import AIInferenceClient
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import Consent, Interpretation, Recognition

router = APIRouter(prefix="/api/v1", tags=["recognitions"])
LIVE_CONSENT = "PROCESSING_FOR_LIVE_TRANSLATION"
SUPPORTED_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}
MP4_MAJOR_BRANDS = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"iso7",
        b"iso8",
        b"iso9",
        b"mp41",
        b"mp42",
        b"avc1",
        b"M4V ",
        b"M4VH",
        b"M4VP",
        b"MSNV",
        b"F4V ",
        b"dash",
        b"cmfc",
        b"cmfs",
    }
)


def get_db() -> Generator[Session, None, None]:
    database_url = get_settings().database_url.get_secret_value()
    with Session(get_engine(database_url)) as session:
        yield session


def get_ai_inference_client() -> AIInferenceClient:
    settings = get_settings()
    return AIInferenceClient(
        base_url=str(settings.ai_service_base_url),
        connection_timeout_seconds=settings.ai_connection_timeout_seconds,
        response_timeout_seconds=settings.ai_response_timeout_seconds,
    )


async def _read_limited_upload(video: UploadFile) -> tuple[int, bytes, str]:
    """Read a bounded upload, retaining its signature and a streaming content digest."""
    total = 0
    header = bytearray()
    digest = sha256()
    while True:
        chunk = await video.read(
            min(64 * 1024, RecognitionConstraints.MAX_SIZE_BYTES - total + 1)
        )
        if not chunk:
            break
        total += len(chunk)
        if total > RecognitionConstraints.MAX_SIZE_BYTES:
            raise HTTPException(status_code=413, detail={"code": "SEGMENT_TOO_LARGE"})
        digest.update(chunk)
        if len(header) < 16:
            header.extend(chunk[: 16 - len(header)])
    if total == 0:
        raise HTTPException(status_code=400, detail={"code": "CAMERA_MEDIA_INVALID"})
    return total, bytes(header), digest.hexdigest()


def _supported_video_extension(
    filename: str | None,
    content_type: str | None,
) -> str | None:
    extension = Path(filename or "").suffix.lower()
    media_type = content_type.partition(";")[0].strip().lower() if content_type else ""
    if SUPPORTED_VIDEO_MEDIA_TYPES.get(extension) != media_type:
        return None
    return extension


def _has_supported_container_signature(extension: str, header: bytes) -> bool:
    if len(header) < 12 or header[4:8] != b"ftyp":
        return False

    major_brand = header[8:12]
    if extension == ".mov":
        return major_brand == b"qt  "
    if extension == ".mp4":
        return major_brand in MP4_MAJOR_BRANDS
    return False


def _validate_recognition_access(
    db: Session,
    consultation_id: UUID,
    session_id: UUID,
    now: datetime,
) -> None:
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


def _inference_file_details(extension: str) -> tuple[str, str]:
    return f"upload{extension}", SUPPORTED_VIDEO_MEDIA_TYPES[extension]


def _to_response(outcome: RecognitionOutcome) -> RecognitionResponse:
    return RecognitionResponse(
        recognition_id=outcome.recognition_id,
        status=outcome.status,
        created_at=outcome.created_at,
        interpretation_id=outcome.interpretation_id,
    )


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
    ai_client: AIInferenceClient = Depends(get_ai_inference_client),
) -> RecognitionResponse:
    try:
        _validate_recognition_access(db, consultation_id, session_id, datetime.now(UTC))
        db.commit()

        extension = _supported_video_extension(video.filename, video.content_type)
        if extension is None:
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": (
                        "Only MP4 (video/mp4) and MOV (video/quicktime) uploads are supported."
                    ),
                },
            )
        total_size, header, content_digest = await _read_limited_upload(video)
        if not _has_supported_container_signature(extension, header):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "CAMERA_MEDIA_INVALID",
                    "message": "The uploaded file does not match its declared MOV or MP4 format.",
                },
            )
        await video.seek(0)

        now = datetime.now(UTC)
        _validate_recognition_access(db, consultation_id, session_id, now)
        filename, inference_content_type = _inference_file_details(extension)
        settings = get_settings()
        outcome = await RecognitionOrchestrationService(
            db,
            ai_client,
            processing_lease_seconds=settings.recognition_processing_lease_seconds,
        ).execute(
            RecognitionSubmission(
                consultation_id=consultation_id,
                session_id=session_id,
                client_request_id=client_request_id,
                content_digest=content_digest,
                size_bytes=total_size,
                container=video.content_type or "",
                video=video.file,
                filename=filename,
                content_type=inference_content_type,
                created_at=now,
                correlation_id=getattr(request.state, "correlation_id", ""),
            )
        )
        return _to_response(outcome)
    except RecognitionRequestConflictError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "CLIENT_REQUEST_ID_CONFLICT"},
        ) from None
    except RecognitionPersistenceError:
        db.rollback()
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR"}) from None
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
