"""Recognition orchestration and persistence."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import BinaryIO, Protocol
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.infrastructure.ai_clients.video_inference_client import (
    AIInferenceError,
    AIInferenceResult,
)
from app.infrastructure.persistence.entities import (
    AuditEvent,
    Interpretation,
    InterpretationCandidate,
    ModelVersion,
    Recognition,
)
from app.infrastructure.persistence.outbox import enqueue_notification

logger = logging.getLogger(__name__)

MODEL_NAME = "bilstm"
MODEL_APPROVAL_STATUS = "RESEARCH_ONLY"
MODEL_VERSION_UNIQUE_CONSTRAINT = "uq_model_versions_model_name_version"
CANDIDATE_WARNING = (
    "Candidates are raw, unverified INCLUDE-50 class labels, not clinical phrases. "
    "Their uncalibrated scores are not confidence."
)


class RecognitionRequestConflictError(Exception):
    """Raised when an idempotency key is reused for different video content."""


class RecognitionPersistenceError(Exception):
    """Raised when a recognition state transition cannot be persisted."""


class VideoInferenceClient(Protocol):
    async def predict_video(
        self,
        video: BinaryIO,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> AIInferenceResult: ...


@dataclass(frozen=True)
class RecognitionSubmission:
    consultation_id: UUID
    session_id: UUID
    client_request_id: str
    content_digest: str
    size_bytes: int
    container: str
    video: BinaryIO
    filename: str
    content_type: str
    created_at: datetime
    correlation_id: str


@dataclass(frozen=True)
class RecognitionOutcome:
    recognition_id: UUID
    status: str
    created_at: datetime
    interpretation_id: UUID | None


@dataclass(frozen=True)
class _RecognitionClaim:
    recognition_id: UUID
    created_at: datetime
    processing_token: UUID


def _is_model_version_unique_violation(error: IntegrityError) -> bool:
    constraint_name = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
    if isinstance(constraint_name, str):
        return constraint_name == MODEL_VERSION_UNIQUE_CONSTRAINT
    return str(error.orig) == (
        "UNIQUE constraint failed: model_versions.model_name, model_versions.version"
    )


class RecognitionOrchestrationService:
    """Claim an idempotent recognition, invoke inference, and persist its result."""

    def __init__(
        self,
        db: Session,
        ai_client: VideoInferenceClient,
        processing_lease_seconds: int = 180,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db = db
        self._ai_client = ai_client
        self._processing_lease = timedelta(seconds=processing_lease_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(self, submission: RecognitionSubmission) -> RecognitionOutcome:
        try:
            claim_or_outcome = self._claim_recognition(submission)
        except RecognitionRequestConflictError:
            raise
        except SQLAlchemyError:
            self._db.rollback()
            logger.error("Recognition claim persistence failed error_type=SQLAlchemyError")
            raise RecognitionPersistenceError("Recognition claim could not be persisted.") from None

        if isinstance(claim_or_outcome, RecognitionOutcome):
            return claim_or_outcome

        try:
            prediction = await self._ai_client.predict_video(
                submission.video,
                submission.filename,
                submission.content_type,
            )
        except AIInferenceError as error:
            outcome = self._mark_failed(claim_or_outcome, self._clock())
            logger.warning(
                "Recognition inference failed recognition_id=%s error_type=%s",
                claim_or_outcome.recognition_id,
                type(error).__name__,
            )
            return outcome

        try:
            return self._complete_recognition(
                claim_or_outcome,
                submission,
                prediction,
            )
        except RecognitionPersistenceError:
            self._db.rollback()
            raise
        except SQLAlchemyError as error:
            self._db.rollback()
            logger.error(
                "Recognition completion persistence failed recognition_id=%s error_type=%s",
                claim_or_outcome.recognition_id,
                type(error).__name__,
            )
            self._mark_failed(claim_or_outcome, self._clock())
            raise RecognitionPersistenceError(
                "Recognition completion could not be persisted."
            ) from None

    def _claim_recognition(
        self,
        submission: RecognitionSubmission,
    ) -> _RecognitionClaim | RecognitionOutcome:
        existing = self._get_by_request_key(submission)
        if existing is not None:
            return self._handle_existing(existing, submission)

        recognition_id = uuid4()
        processing_token = uuid4()
        recognition = Recognition(
            id=recognition_id,
            consultation_id=submission.consultation_id,
            session_id=submission.session_id,
            client_request_id=submission.client_request_id,
            status="PROCESSING",
            media_metadata={
                "size_bytes": submission.size_bytes,
                "container": submission.container,
                "duration_ms": None,
                "sha256": submission.content_digest,
            },
            temporary_media_reference=None,
            processing_lease_expires_at=(
                submission.created_at + self._processing_lease
            ),
            processing_token=processing_token,
            created_at=submission.created_at,
            completed_at=None,
        )
        self._db.add(recognition)

        try:
            self._db.commit()
        except IntegrityError:
            self._db.rollback()
            existing = self._get_by_request_key(submission)
            if existing is None:
                raise
            return self._handle_existing(existing, submission)

        return _RecognitionClaim(
            recognition_id=recognition_id,
            created_at=submission.created_at,
            processing_token=processing_token,
        )

    def _get_by_request_key(self, submission: RecognitionSubmission) -> Recognition | None:
        return self._db.execute(
            select(Recognition).where(
                Recognition.consultation_id == submission.consultation_id,
                Recognition.session_id == submission.session_id,
                Recognition.client_request_id == submission.client_request_id,
            )
        ).scalar_one_or_none()

    def _handle_existing(
        self,
        recognition: Recognition,
        submission: RecognitionSubmission,
    ) -> _RecognitionClaim | RecognitionOutcome:
        if recognition.media_metadata.get("sha256") != submission.content_digest:
            raise RecognitionRequestConflictError

        if recognition.status == "COMPLETED":
            return self._outcome_for(recognition)

        if recognition.status == "PROCESSING":
            return self._claim_expired_processing(recognition, submission.created_at)

        if recognition.status != "FAILED":
            return self._outcome_for(recognition)

        processing_token = uuid4()
        claimed_id = self._db.execute(
            update(Recognition)
            .where(
                Recognition.id == recognition.id,
                Recognition.status == "FAILED",
            )
            .values(
                status="PROCESSING",
                completed_at=None,
                processing_lease_expires_at=(
                    submission.created_at + self._processing_lease
                ),
                processing_token=processing_token,
            )
            .returning(Recognition.id)
            .execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if claimed_id is None:
            self._db.rollback()
            return self._outcome_for_id(recognition.id)

        self._db.commit()
        return _RecognitionClaim(
            recognition_id=recognition.id,
            created_at=recognition.created_at,
            processing_token=processing_token,
        )

    def _claim_expired_processing(
        self,
        recognition: Recognition,
        now: datetime,
    ) -> _RecognitionClaim | RecognitionOutcome:
        processing_token = uuid4()
        claimed_id = self._db.execute(
            update(Recognition)
            .where(
                Recognition.id == recognition.id,
                Recognition.status == "PROCESSING",
                or_(
                    Recognition.processing_lease_expires_at.is_(None),
                    Recognition.processing_lease_expires_at <= now,
                    Recognition.processing_token.is_(None),
                ),
            )
            .values(
                processing_lease_expires_at=now + self._processing_lease,
                processing_token=processing_token,
                completed_at=None,
            )
            .returning(Recognition.id)
            .execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if claimed_id is None:
            self._db.rollback()
            return self._outcome_for_id(recognition.id)

        self._db.commit()
        return _RecognitionClaim(
            recognition_id=recognition.id,
            created_at=recognition.created_at,
            processing_token=processing_token,
        )

    def _outcome_for(self, recognition: Recognition) -> RecognitionOutcome:
        interpretation_id = self._db.execute(
            select(Interpretation.id).where(
                Interpretation.recognition_id == recognition.id
            )
        ).scalar_one_or_none()
        return RecognitionOutcome(
            recognition_id=recognition.id,
            status=recognition.status,
            created_at=recognition.created_at,
            interpretation_id=interpretation_id,
        )

    def _outcome_for_id(self, recognition_id: UUID) -> RecognitionOutcome:
        recognition = self._db.scalars(
            select(Recognition)
            .where(Recognition.id == recognition_id)
            .execution_options(populate_existing=True)
        ).one_or_none()
        if recognition is None:
            raise RecognitionPersistenceError("Recognition claim no longer exists.")
        return self._outcome_for(recognition)

    def _complete_recognition(
        self,
        claim: _RecognitionClaim,
        submission: RecognitionSubmission,
        prediction: AIInferenceResult,
    ) -> RecognitionOutcome:
        completed_at = self._clock()
        completed_id = self._db.execute(
            update(Recognition)
            .where(
                Recognition.id == claim.recognition_id,
                Recognition.status == "PROCESSING",
                Recognition.processing_token == claim.processing_token,
                Recognition.processing_lease_expires_at > completed_at,
            )
            .values(
                status="COMPLETED",
                completed_at=completed_at,
                processing_lease_expires_at=None,
            )
            .returning(Recognition.id)
            .execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if completed_id is None:
            self._db.rollback()
            return self._outcome_for_id(claim.recognition_id)

        model = self._get_or_create_model_version(prediction, completed_at)
        interpretation_id = uuid4()

        self._db.add(
            Interpretation(
                id=interpretation_id,
                recognition_id=claim.recognition_id,
                status="UNVERIFIED",
                model_version_id=model.id,
                # This enables Stage A patient confirmation; it does not calibrate model scores.
                quality_status="ACCEPTABLE",
                created_at=completed_at,
            )
        )
        self._db.flush()
        self._db.add_all(
            [
                InterpretationCandidate(
                    id=uuid4(),
                    interpretation_id=interpretation_id,
                    rank=candidate.rank,
                    text=candidate.label,
                    confidence=None,
                    selected=False,
                    created_at=completed_at,
                )
                for candidate in prediction.candidates
            ]
        )
        self._db.add(
            AuditEvent(
                id=uuid4(),
                consultation_id=submission.consultation_id,
                session_id=submission.session_id,
                action="RECOGNITION_COMPLETED",
                resource_type="Recognition",
                resource_id=claim.recognition_id,
                outcome="SUCCESS",
                correlation_id=submission.correlation_id,
                event_metadata={
                    "interpretation_id": str(interpretation_id),
                    "candidate_count": len(prediction.candidates),
                    "model_version": prediction.model_version,
                    "score_type": prediction.score_type,
                },
                created_at=completed_at,
            )
        )
        enqueue_notification(
            self._db,
            submission.consultation_id,
            "recognition.completed",
            claim.recognition_id,
            completed_at,
        )
        self._db.commit()
        return RecognitionOutcome(
            recognition_id=claim.recognition_id,
            status="COMPLETED",
            created_at=claim.created_at,
            interpretation_id=interpretation_id,
        )

    def _get_or_create_model_version(
        self,
        prediction: AIInferenceResult,
        created_at: datetime,
    ) -> ModelVersion:
        model = self._db.scalars(
            select(ModelVersion)
            .where(
                ModelVersion.model_name == MODEL_NAME,
                ModelVersion.version == prediction.model_version,
            )
        ).one_or_none()
        model_metrics = {
            "prediction_status": prediction.prediction_status,
            "score_type": prediction.score_type,
            "inference_warning": prediction.warning,
            "candidate_warning": CANDIDATE_WARNING,
        }
        if model is None:
            new_model = ModelVersion(
                id=uuid4(),
                model_name=MODEL_NAME,
                version=prediction.model_version,
                metrics=model_metrics,
                approval_status=MODEL_APPROVAL_STATUS,
                created_at=created_at,
            )
            try:
                with self._db.begin_nested():
                    self._db.add(new_model)
                    self._db.flush()
            except IntegrityError as error:
                if not _is_model_version_unique_violation(error):
                    raise
                model = self._db.scalars(
                    select(ModelVersion).where(
                        ModelVersion.model_name == MODEL_NAME,
                        ModelVersion.version == prediction.model_version,
                    )
                ).one_or_none()
                if model is None:
                    raise
            else:
                return new_model

        model.metrics = {**model.metrics, **model_metrics}
        return model

    def _mark_failed(
        self,
        claim: _RecognitionClaim,
        failed_at: datetime,
    ) -> RecognitionOutcome:
        try:
            self._db.execute(
                update(Recognition)
                .where(
                    Recognition.id == claim.recognition_id,
                    Recognition.status == "PROCESSING",
                    Recognition.processing_token == claim.processing_token,
                    Recognition.processing_lease_expires_at > failed_at,
                )
                .values(
                    status="FAILED",
                    completed_at=failed_at,
                    processing_lease_expires_at=None,
                )
                .returning(Recognition.id)
                .execution_options(synchronize_session=False)
            ).scalar_one_or_none()
            self._db.commit()
        except SQLAlchemyError:
            self._db.rollback()
            logger.error("Recognition failure state could not be persisted")
            raise RecognitionPersistenceError(
                "Recognition failure state could not be persisted."
            ) from None
        return self._outcome_for_id(claim.recognition_id)
