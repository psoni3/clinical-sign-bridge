"""M1 lifecycle, consent, messaging, recognition, and safety integration tests."""

import asyncio
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import BinaryIO
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, update
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.websockets import WebSocketDisconnect

import app.api.routes.confirmations as confirmation_routes
import app.api.routes.recognitions as recognition_routes
import app.application.recognition_service as recognition_service
from app.infrastructure.ai_clients.video_inference_client import (
    AIInferenceCandidate,
    AIInferenceError,
    AIInferenceHTTPError,
    AIInferenceResponseError,
    AIInferenceResult,
    AIInferenceUnavailableError,
)
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import (
    AuditEvent,
    Base,
    Confirmation,
    Consent,
    Consultation,
    Conversation,
    Interpretation,
    InterpretationCandidate,
    Message,
    ModelVersion,
    OutboxEvent,
    Recognition,
)
from app.infrastructure.persistence.entities import (
    ConsultationSession as ConsultationSessionEntity,
)
from app.infrastructure.persistence.outbox import OutboxDelivery, OutboxDispatcher
from app.main import app

TEST_DATABASE_URL = "sqlite+pysqlite:///:memory:"


def _successful_inference_result() -> AIInferenceResult:
    return AIInferenceResult(
        prediction_status="UNCONFIRMED",
        model_version="bilstm-v1",
        score_type="uncalibrated_softmax",
        frame_count=12,
        candidates=tuple(
            AIInferenceCandidate(rank=rank, label=f"INCLUDE-50-class-{rank}", score=0.9 - rank / 10)
            for rank in range(1, 6)
        ),
        warning="Predictions are unconfirmed.",
    )


class FakeInferenceClient:
    def __init__(
        self,
        result: AIInferenceResult | None = None,
        error: AIInferenceError | None = None,
        block_until_released: bool = False,
    ) -> None:
        self.result = result or _successful_inference_result()
        self.error = error
        self.block_until_released = block_until_released
        self.calls = 0
        self.filename: str | None = None
        self.content_type: str | None = None
        self.uploaded_video: bytes | None = None
        self.started = Event()
        self.release = Event()
        self._lock = Lock()

    async def predict_video(
        self,
        video: BinaryIO,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> AIInferenceResult:
        with self._lock:
            self.calls += 1
        self.filename = filename
        self.content_type = content_type
        self.uploaded_video = video.read()
        self.started.set()
        if self.block_until_released and not self.release.wait(timeout=10):
            raise TimeoutError("Test inference release timed out.")
        if self.error is not None:
            raise self.error
        return self.result


class ConcurrentFakeInferenceClient(FakeInferenceClient):
    def __init__(self) -> None:
        super().__init__()
        self._inference_barrier = Barrier(2)

    async def predict_video(
        self,
        video: BinaryIO,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> AIInferenceResult:
        with self._lock:
            self.calls += 1
        self.filename = filename
        self.content_type = content_type
        self.uploaded_video = video.read()
        self.started.set()
        await asyncio.to_thread(self._inference_barrier.wait, timeout=10)
        if self.error is not None:
            raise self.error
        return self.result


def _override_inference_client(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: FakeInferenceClient,
) -> None:
    monkeypatch.setitem(
        app.dependency_overrides,
        recognition_routes.get_ai_inference_client,
        lambda: fake_client,
    )


def _session(client: TestClient, consultation_id: str, join_code: str, role: str) -> str:
    response = client.post(
        f"/api/v1/consultations/{consultation_id}/join",
        json={
            "join_code": join_code,
            "device_label": role,
            "session_type": role,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["sessionId"]


def _consultation(client: TestClient) -> tuple[str, str, str, str, str]:
    created = client.post(
        "/api/v1/consultations",
        json={"display_label": "Synthetic test", "expires_in_minutes": 120},
    )
    assert created.status_code == 201, created.text
    data = created.json()
    patient_session_id = _session(
        client, data["consultationId"], data["joinCode"], "PATIENT_DEVICE"
    )
    doctor_session_id = _session(client, data["consultationId"], data["joinCode"], "DOCTOR_VIEW")
    started = client.post(f"/api/v1/consultations/{data['consultationId']}/start")
    assert started.status_code == 200, started.text
    status = client.get(f"/api/v1/consultations/{data['consultationId']}")
    assert status.status_code == 200, status.text
    return (
        data["consultationId"],
        data["joinCode"],
        patient_session_id,
        doctor_session_id,
        status.json()["conversationId"],
    )


def _give_processing_consent(client: TestClient, consultation_id: str, session_id: str) -> None:
    response = client.post(
        f"/api/v1/consultations/{consultation_id}/consents",
        json={
            "session_id": session_id,
            "consent_type": "PROCESSING_FOR_LIVE_TRANSLATION",
            "policy_version": "v1",
            "accepted": True,
        },
    )
    assert response.status_code == 201, response.text


def _video_upload(client_request_id: str, session_id: str) -> dict[str, tuple]:
    content = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 24
    return {
        "video": ("synthetic.mp4", BytesIO(content), "video/mp4"),
        "clientRequestId": (None, client_request_id),
        "sessionId": (None, session_id),
    }


def _mov_upload(client_request_id: str, session_id: str) -> dict[str, tuple]:
    content = b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 24
    return {
        "video": ("synthetic.mov", BytesIO(content), "video/quicktime"),
        "clientRequestId": (None, client_request_id),
        "sessionId": (None, session_id),
    }


def _seed_confirmable_interpretation(
    consultation_id: str,
    session_id: str,
    text: str = "Synthetic candidate",
) -> tuple[str, str]:
    recognition_id = uuid4()
    interpretation_id = uuid4()
    candidate_id = uuid4()
    now = datetime.now(UTC)
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        model = db.scalars(
            select(ModelVersion).where(
                ModelVersion.model_name == "test-only",
                ModelVersion.version == "test-v1",
            )
        ).one_or_none()
        if model is None:
            model = ModelVersion(
                id=uuid4(),
                model_name="test-only",
                version="test-v1",
                metrics={},
                approval_status="TEST_ONLY",
                created_at=now,
            )
            db.add(model)
            db.flush()
        db.add(
            Recognition(
                id=recognition_id,
                consultation_id=UUID(consultation_id),
                session_id=UUID(session_id),
                client_request_id=str(uuid4()),
                status="COMPLETED",
                media_metadata={"size_bytes": 1},
                temporary_media_reference=None,
                created_at=now,
                completed_at=now,
            )
        )
        db.add(
            Interpretation(
                id=interpretation_id,
                recognition_id=recognition_id,
                status="UNVERIFIED",
                model_version_id=model.id,
                quality_status="ACCEPTABLE",
                created_at=now,
            )
        )
        db.add(
            InterpretationCandidate(
                id=candidate_id,
                interpretation_id=interpretation_id,
                rank=1,
                text=text,
                confidence=0.5,
                selected=False,
                created_at=now,
            )
        )
        db.commit()
    return str(interpretation_id), str(candidate_id)


def _assert_confirmation_attempt_left_no_writes(
    consultation_id: str,
    interpretation_id: str,
    candidate_id: str,
    conversation_id: str,
) -> None:
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        interpretation_status = db.scalar(
            select(Interpretation.status).where(
                Interpretation.id == UUID(interpretation_id)
            )
        )
        candidate_selected = db.scalar(
            select(InterpretationCandidate.selected).where(
                InterpretationCandidate.id == UUID(candidate_id)
            )
        )
        confirmations = db.scalars(
            select(Confirmation).where(
                Confirmation.interpretation_id == UUID(interpretation_id)
            )
        ).all()
        messages = db.scalars(
            select(Message).where(Message.conversation_id == UUID(conversation_id))
        ).all()
        audits = db.scalars(
            select(AuditEvent).where(
                AuditEvent.resource_id == UUID(interpretation_id),
                AuditEvent.action == "CONFIRM_INTERPRETATION",
            )
        ).all()
        outbox_events = db.scalars(
            select(OutboxEvent).where(
                OutboxEvent.consultation_id == UUID(consultation_id),
                OutboxEvent.event_type == "confirmation.received",
            )
        ).all()

    assert interpretation_status == "UNVERIFIED"
    assert candidate_selected is False
    assert confirmations == []
    assert messages == []
    assert audits == []
    assert outbox_events == []


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def test_create_join_start_get_and_complete_consultation() -> None:
    with TestClient(app) as client:
        consultation_id, _, _, _, conversation_id = _consultation(client)
        assert conversation_id
        response = client.post(f"/api/v1/consultations/{consultation_id}/complete")
        assert response.status_code == 200
        assert response.json()["status"] == "COMPLETED"
        assert client.get("/api/v1/doctors/consultations").status_code == 200


def test_doctor_can_read_completed_consultation_history() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, doctor_session_id, conversation_id = _consultation(
            client
        )
        message = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "PATIENT",
                "session_id": patient_session_id,
                "message_type": "TEXT",
                "text": "Synthetic confirmed message",
            },
        )
        assert message.status_code == 201
        completed = client.post(f"/api/v1/consultations/{consultation_id}/complete")
        assert completed.status_code == 200

        active_queue = client.get("/api/v1/doctors/consultations")
        completed_history = client.get(
            "/api/v1/doctors/consultations",
            params={"includeCompleted": True},
        )
        assert all(
            row["consultationId"] != consultation_id for row in active_queue.json()["consultations"]
        )
        archived = next(
            row
            for row in completed_history.json()["consultations"]
            if row["consultationId"] == consultation_id
        )
        assert archived["status"] == "COMPLETED"
        assert archived["isActive"] is False

        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": doctor_session_id},
        )
        assert history.status_code == 200
        assert history.json()["messages"][0]["text"] == "Synthetic confirmed message"

        rejected_message = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "DOCTOR",
                "session_id": doctor_session_id,
                "message_type": "TEXT",
                "text": "This must not be sent",
            },
        )
        assert rejected_message.status_code == 410


def test_join_codes_are_hashed_and_attempts_are_rate_limited() -> None:
    with TestClient(app) as client:
        created = client.post("/api/v1/consultations", json={"expires_in_minutes": 120})
        consultation_id = created.json()["consultationId"]
        join_code = created.json()["joinCode"]
        assert len(join_code) == 12
        assert all(character in "0123456789ABCDEF" for character in join_code)
        for _ in range(5):
            rejected = client.post(
                f"/api/v1/consultations/{consultation_id}/join",
                json={"join_code": "incorrect"},
            )
            assert rejected.status_code == 401
        limited = client.post(
            f"/api/v1/consultations/{consultation_id}/join",
            json={"join_code": "incorrect"},
        )
        assert limited.status_code == 429
        with Session(get_engine(TEST_DATABASE_URL)) as db:
            row = db.execute(
                select(Base.metadata.tables["consultations"].c.join_code_hash).where(
                    Base.metadata.tables["consultations"].c.id == UUID(consultation_id)
                )
            ).scalar_one()
        assert "$" in row
        assert join_code not in row


def test_consent_is_bound_to_patient_session_and_separate_purpose() -> None:
    with TestClient(app) as client:
        consultation_id, join_code, patient_session_id, _, _ = _consultation(client)
        research = client.post(
            f"/api/v1/consultations/{consultation_id}/consents",
            json={
                "session_id": patient_session_id,
                "consent_type": "RESEARCH_VIDEO_COLLECTION",
                "policy_version": "v1",
                "accepted": False,
            },
        )
        assert research.status_code == 201
        assert research.json()["accepted"] is False
        other_consultation = client.post(
            "/api/v1/consultations", json={"expires_in_minutes": 120}
        ).json()
        other_patient_session = _session(
            client,
            other_consultation["consultationId"],
            other_consultation["joinCode"],
            "PATIENT_DEVICE",
        )
        client.post(f"/api/v1/consultations/{other_consultation['consultationId']}/start")
        mismatch = client.post(
            f"/api/v1/consultations/{other_consultation['consultationId']}/consents",
            json={
                "session_id": patient_session_id,
                "consent_type": "PROCESSING_FOR_LIVE_TRANSLATION",
                "policy_version": "v1",
                "accepted": True,
            },
        )
        assert mismatch.status_code == 401
        assert other_patient_session
        assert join_code


def test_withdrawing_live_consent_blocks_recognition() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        withdrawal = client.post(
            f"/api/v1/consultations/{consultation_id}/consents",
            json={
                "session_id": patient_session_id,
                "consent_type": "PROCESSING_FOR_LIVE_TRANSLATION",
                "policy_version": "v1",
                "accepted": False,
            },
        )
        assert withdrawal.status_code == 201
        assert withdrawal.json()["accepted"] is False
        recognition = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("after-withdrawal", patient_session_id),
        )
        assert recognition.status_code == 409
        assert recognition.json()["code"] == "CONSENT_REQUIRED"


def test_recognition_persists_unverified_candidates_for_stage_a_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        missing_consent = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("no-consent", patient_session_id),
        )
        assert missing_consent.status_code == 409
        assert fake_client.calls == 0

        _give_processing_consent(client, consultation_id, patient_session_id)
        uploaded = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("request-1", patient_session_id),
        )
        assert uploaded.status_code == 201, uploaded.text
        result = uploaded.json()
        assert result["status"] == "COMPLETED"
        assert result["interpretationId"]
        assert fake_client.calls == 1
        assert fake_client.filename == "upload.mp4"
        assert fake_client.content_type == "video/mp4"
        assert fake_client.uploaded_video is not None

        polled = client.get(
            f"/api/v1/recognitions/{result['recognitionId']}",
            params={"sessionId": patient_session_id},
        )
        assert polled.status_code == 200
        assert polled.json()["interpretationId"] == result["interpretationId"]
        interpretation = client.get(
            f"/api/v1/interpretations/{result['interpretationId']}",
            params={"sessionId": patient_session_id},
        )
        assert interpretation.status_code == 200
        interpretation_body = interpretation.json()
        assert interpretation_body["status"] == "UNVERIFIED"
        assert interpretation_body["modelVersion"] == "bilstm-v1"
        assert interpretation_body["quality"]["status"] == "ACCEPTABLE"
        assert "raw, unverified INCLUDE-50 class labels" in (
            interpretation_body["quality"]["messages"][0]
        )
        candidates = interpretation_body["candidates"]
        assert [candidate["rank"] for candidate in candidates] == [1, 2, 3, 4, 5]
        assert [candidate["text"] for candidate in candidates] == [
            f"INCLUDE-50-class-{rank}" for rank in range(1, 6)
        ]
        assert all(candidate["confidence"] is None for candidate in candidates)

        with Session(get_engine(TEST_DATABASE_URL)) as db:
            recognition = db.get(Recognition, UUID(result["recognitionId"]))
            confirmations = db.scalars(
                select(Confirmation)
                .join(Interpretation)
                .where(Interpretation.recognition_id == UUID(result["recognitionId"]))
            ).all()
            messages = db.scalars(
                select(Message).where(Message.conversation_id == UUID(conversation_id))
            ).all()
            persisted_candidates = db.scalars(
                select(InterpretationCandidate).where(
                    InterpretationCandidate.interpretation_id
                    == UUID(result["interpretationId"])
                )
            ).all()
        assert recognition is not None
        assert recognition.media_metadata["sha256"] == sha256(
            fake_client.uploaded_video
        ).hexdigest()
        assert confirmations == []
        assert messages == []
        assert len(persisted_candidates) == 5

        confirmed = client.post(
            f"/api/v1/interpretations/{result['interpretationId']}/confirm",
            json={
                "candidate_id": candidates[0]["candidateId"],
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        assert confirmed.status_code == 201, confirmed.text
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )
        assert len(history.json()["messages"]) == 1
        assert history.json()["messages"][0]["text"] == "INCLUDE-50-class-1"


def test_distinct_recognitions_reuse_the_same_model_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        responses = [
            client.post(
                f"/api/v1/consultations/{consultation_id}/recognitions",
                files=_video_upload(request_id, patient_session_id),
            )
            for request_id in ("model-version-first", "model-version-second")
        ]

    assert all(response.status_code == 201 for response in responses)
    assert all(response.json()["status"] == "COMPLETED" for response in responses)
    assert fake_client.calls == 2
    recognition_ids = [UUID(response.json()["recognitionId"]) for response in responses]
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        models = db.scalars(
            select(ModelVersion).where(
                ModelVersion.model_name == "bilstm",
                ModelVersion.version == "bilstm-v1",
            )
        ).all()
        model_version_ids = set(
            db.scalars(
                select(Interpretation.model_version_id).where(
                    Interpretation.recognition_id.in_(recognition_ids)
                )
            ).all()
        )
    assert len(models) == 1
    assert model_version_ids == {models[0].id}


def test_recognition_rejects_bad_container_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        invalid = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files={
                "video": ("bad.mp4", BytesIO(b"not video"), "video/mp4"),
                "clientRequestId": (None, "bad"),
                "sessionId": (None, patient_session_id),
            },
        )
        assert invalid.status_code == 400
        assert invalid.json()["message"] == (
            "The uploaded file does not match its declared MOV or MP4 format."
        )
        invalid_mov = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files={
                "video": (
                    "bad.mov",
                    BytesIO(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 24),
                    "video/quicktime",
                ),
                "clientRequestId": (None, "bad-mov"),
                "sessionId": (None, patient_session_id),
            },
        )
        assert invalid_mov.status_code == 400
        assert fake_client.calls == 0
        first = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("same-key", patient_session_id),
        )
        second = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("same-key", patient_session_id),
        )
        assert first.status_code == second.status_code == 201
        assert first.json()["recognitionId"] == second.json()["recognitionId"]
        assert first.json()["interpretationId"] == second.json()["interpretationId"]
        assert first.json()["status"] == second.json()["status"] == "COMPLETED"
        assert fake_client.calls == 1
        with Session(get_engine(TEST_DATABASE_URL)) as db:
            recognitions = db.scalars(
                select(Recognition).where(
                    Recognition.client_request_id == "same-key"
                )
            ).all()
            candidates = db.scalars(
                select(InterpretationCandidate)
                .join(Interpretation)
                .where(Interpretation.recognition_id == UUID(first.json()["recognitionId"]))
            ).all()
        assert len(recognitions) == 1
        assert len(candidates) == 5


def test_recognition_accepts_mov_and_forwards_matching_multipart_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        uploaded = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_mov_upload("valid-mov", patient_session_id),
        )

    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "COMPLETED"
    assert fake_client.calls == 1
    assert fake_client.filename == "upload.mov"
    assert fake_client.content_type == "video/quicktime"
    assert fake_client.uploaded_video == b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 24


def test_recognition_returns_conflict_for_different_content_with_same_request_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        first = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("same-key-different-content", patient_session_id),
        )
        changed = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files={
                "video": (
                    "synthetic.mp4",
                    BytesIO(b"\x00\x00\x00\x18ftypisom" + b"\x01" * 24),
                    "video/mp4",
                ),
                "clientRequestId": (None, "same-key-different-content"),
                "sessionId": (None, patient_session_id),
            },
        )

    assert first.status_code == 201
    assert changed.status_code == 409
    assert changed.json()["code"] == "CLIENT_REQUEST_ID_CONFLICT"
    assert fake_client.calls == 1


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("signing-segment.webm", "video/webm;codecs=vp8", b"\x1a\x45\xdf\xa3" + b"\x00" * 32),
        ("unsupported.avi", "video/x-msvideo", b"RIFF" + b"\x00" * 32),
        ("mismatched.mov", "video/mp4", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 24),
    ],
    ids=["webm", "avi", "extension-mime-mismatch"],
)
def test_recognition_rejects_unsupported_media_before_ai_invocation(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content_type: str,
    content: bytes,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        response = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files={
                "video": (
                    filename,
                    BytesIO(content),
                    content_type,
                ),
                "clientRequestId": (None, f"unsupported-{filename}"),
                "sessionId": (None, patient_session_id),
            },
        )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert response.json()["message"] == (
        "Only MP4 (video/mp4) and MOV (video/quicktime) uploads are supported."
    )
    assert fake_client.calls == 0
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        recognition = db.scalar(
            select(Recognition).where(
                Recognition.client_request_id == f"unsupported-{filename}"
            )
        )
    assert recognition is None


@pytest.mark.parametrize(
    "inference_error",
    [
        AIInferenceUnavailableError("AI service timeout."),
        AIInferenceHTTPError(415),
        AIInferenceResponseError("Invalid prediction response."),
    ],
    ids=["unavailable", "http-rejection", "invalid-response"],
)
def test_inference_errors_fail_recognition_without_persisting_candidates(
    monkeypatch: pytest.MonkeyPatch,
    inference_error: AIInferenceError,
) -> None:
    fake_client = FakeInferenceClient(error=inference_error)
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        response = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("inference-failure", patient_session_id),
        )

    assert response.status_code == 201
    assert response.json()["status"] == "FAILED"
    assert response.json()["interpretationId"] is None
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        recognition = db.get(Recognition, UUID(response.json()["recognitionId"]))
        interpretations = db.scalars(
            select(Interpretation).where(
                Interpretation.recognition_id == UUID(response.json()["recognitionId"])
            )
        ).all()
        candidates = db.scalars(
            select(InterpretationCandidate)
            .join(Interpretation)
            .where(Interpretation.recognition_id == UUID(response.json()["recognitionId"]))
        ).all()
        audits = db.scalars(
            select(AuditEvent).where(
                AuditEvent.resource_id == UUID(response.json()["recognitionId"]),
                AuditEvent.action == "RECOGNITION_COMPLETED",
            )
        ).all()
        outbox = db.scalars(
            select(OutboxEvent).where(
                OutboxEvent.resource_id == UUID(response.json()["recognitionId"]),
                OutboxEvent.event_type == "recognition.completed",
            )
        ).all()
        messages = db.scalars(
            select(Message).where(Message.conversation_id == UUID(conversation_id))
        ).all()

    assert recognition is not None
    assert recognition.status == "FAILED"
    assert interpretations == []
    assert candidates == []
    assert audits == []
    assert outbox == []
    assert messages == []


def test_failed_recognition_can_be_retried_without_duplicate_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient(error=AIInferenceUnavailableError("AI timeout."))
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        first = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("retry-after-failure", patient_session_id),
        )
        assert first.status_code == 201
        assert first.json()["status"] == "FAILED"

        fake_client.error = None
        retry = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("retry-after-failure", patient_session_id),
        )

    assert retry.status_code == 201
    assert retry.json()["status"] == "COMPLETED"
    assert retry.json()["recognitionId"] == first.json()["recognitionId"]
    assert retry.json()["interpretationId"]
    assert fake_client.calls == 2
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        recognitions = db.scalars(
            select(Recognition).where(
                Recognition.client_request_id == "retry-after-failure"
            )
        ).all()
        interpretations = db.scalars(
            select(Interpretation).where(
                Interpretation.recognition_id == UUID(retry.json()["recognitionId"])
            )
        ).all()
        candidates = db.scalars(
            select(InterpretationCandidate)
            .join(Interpretation)
            .where(Interpretation.recognition_id == UUID(retry.json()["recognitionId"]))
        ).all()

    assert len(recognitions) == 1
    assert len(interpretations) == 1
    assert len(candidates) == 5


def test_completion_write_failure_rolls_back_results_and_marks_recognition_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)

    def fail_outbox_enqueue(*args: object, **kwargs: object) -> None:
        raise SQLAlchemyError("Injected outbox persistence failure.")

    monkeypatch.setattr(recognition_service, "enqueue_notification", fail_outbox_enqueue)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        response = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("completion-rollback", patient_session_id),
        )

    assert response.status_code == 500
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        recognition = db.scalar(
            select(Recognition).where(
                Recognition.client_request_id == "completion-rollback"
            )
        )
        assert recognition is not None
        recognition_id = recognition.id
        interpretations = db.scalars(
            select(Interpretation).where(
                Interpretation.recognition_id == recognition_id
            )
        ).all()
        candidates = db.scalars(
            select(InterpretationCandidate)
            .join(Interpretation)
            .where(Interpretation.recognition_id == recognition_id)
        ).all()
        audits = db.scalars(
            select(AuditEvent).where(
                AuditEvent.resource_id == recognition_id,
                AuditEvent.action == "RECOGNITION_COMPLETED",
            )
        ).all()
        outbox = db.scalars(
            select(OutboxEvent).where(
                OutboxEvent.resource_id == recognition_id,
                OutboxEvent.event_type == "recognition.completed",
            )
        ).all()

    assert recognition.status == "FAILED"
    assert interpretations == []
    assert candidates == []
    assert audits == []
    assert outbox == []


@pytest.mark.parametrize(
    ("session_kind", "expected_status"),
    [
        ("expired", 401),
        ("revoked", 401),
        ("doctor", 403),
        ("missing-consent", 409),
    ],
)
def test_invalid_recognition_session_or_consent_never_calls_ai(
    monkeypatch: pytest.MonkeyPatch,
    session_kind: str,
    expected_status: int,
) -> None:
    fake_client = FakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, doctor_session_id, _ = _consultation(client)
        if session_kind in {"expired", "revoked"}:
            _give_processing_consent(client, consultation_id, patient_session_id)
            with Session(get_engine(TEST_DATABASE_URL)) as db:
                patient_session = db.get(
                    ConsultationSessionEntity, UUID(patient_session_id)
                )
                assert patient_session is not None
                if session_kind == "expired":
                    patient_session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
                else:
                    patient_session.revoked_at = datetime.now(UTC)
                db.commit()
        elif session_kind == "doctor":
            _give_processing_consent(client, consultation_id, patient_session_id)

        session_id = (
            doctor_session_id if session_kind == "doctor" else patient_session_id
        )
        response = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload(f"invalid-session-{session_kind}", session_id),
        )

    assert response.status_code == expected_status
    assert fake_client.calls == 0
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        recognition = db.scalar(
            select(Recognition).where(
                Recognition.client_request_id == f"invalid-session-{session_kind}"
            )
        )
    assert recognition is None


def test_concurrent_http_recognition_requests_share_one_inference_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(tmp_path / "concurrent-recognition.db")),
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    consultation_id = uuid4()
    session_id = uuid4()
    with Session(engine) as db:
        db.add_all(
            [
                Consultation(
                    id=consultation_id,
                    status="ACTIVE",
                    join_code_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
                ConsultationSessionEntity(
                    id=session_id,
                    consultation_id=consultation_id,
                    session_type="PATIENT_DEVICE",
                    session_token_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
                Consent(
                    id=uuid4(),
                    consultation_id=consultation_id,
                    session_id=session_id,
                    consent_type="PROCESSING_FOR_LIVE_TRANSLATION",
                    policy_version="test-v1",
                    accepted=True,
                    captured_at=now,
                ),
            ]
        )
        db.commit()

    def get_recognition_test_db() -> Generator[Session, None, None]:
        with Session(engine) as db:
            yield db

    monkeypatch.setitem(
        app.dependency_overrides,
        recognition_routes.get_db,
        get_recognition_test_db,
    )
    fake_client = FakeInferenceClient(block_until_released=True)
    _override_inference_client(monkeypatch, fake_client)
    request_key = "concurrent-identical-recognition"

    def post_recognition() -> tuple[int, dict[str, object]]:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/consultations/{consultation_id}/recognitions",
                files=_video_upload(request_key, str(session_id)),
            )
            return response.status_code, response.json()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_request = executor.submit(post_recognition)
            assert fake_client.started.wait(timeout=10)
            try:
                with TestClient(app) as client:
                    second_response = client.post(
                        f"/api/v1/consultations/{consultation_id}/recognitions",
                        files=_video_upload(request_key, str(session_id)),
                    )
            finally:
                fake_client.release.set()
            first_status, first_body = first_request.result(timeout=30)

        assert first_status == 201
        assert first_body["status"] == "COMPLETED"
        assert second_response.status_code == 201
        assert second_response.json()["status"] == "PROCESSING"
        assert second_response.json()["recognitionId"] == first_body["recognitionId"]
        assert fake_client.calls == 1
        with Session(engine) as db:
            recognitions = db.scalars(select(Recognition)).all()
            interpretations = db.scalars(select(Interpretation)).all()
            candidates = db.scalars(select(InterpretationCandidate)).all()
        assert len(recognitions) == 1
        assert len(interpretations) == 1
        assert len(candidates) == 5
    finally:
        fake_client.release.set()
        engine.dispose()


def test_concurrent_distinct_recognitions_share_one_model_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(tmp_path / "concurrent-model-version.db")),
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    consultation_id = uuid4()
    session_id = uuid4()
    with Session(engine) as db:
        db.add_all(
            [
                Consultation(
                    id=consultation_id,
                    status="ACTIVE",
                    join_code_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
                ConsultationSessionEntity(
                    id=session_id,
                    consultation_id=consultation_id,
                    session_type="PATIENT_DEVICE",
                    session_token_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
                Consent(
                    id=uuid4(),
                    consultation_id=consultation_id,
                    session_id=session_id,
                    consent_type="PROCESSING_FOR_LIVE_TRANSLATION",
                    policy_version="test-v1",
                    accepted=True,
                    captured_at=now,
                ),
            ]
        )
        db.commit()

    def get_recognition_test_db() -> Generator[Session, None, None]:
        with Session(engine) as db:
            yield db

    monkeypatch.setitem(
        app.dependency_overrides,
        recognition_routes.get_db,
        get_recognition_test_db,
    )
    fake_client = ConcurrentFakeInferenceClient()
    _override_inference_client(monkeypatch, fake_client)
    request_ids = ("concurrent-model-version-a", "concurrent-model-version-b")

    def post_recognition(request_id: str) -> tuple[int, dict[str, object]]:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/consultations/{consultation_id}/recognitions",
                files=_video_upload(request_id, str(session_id)),
            )
            return response.status_code, response.json()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(post_recognition, request_ids))

        assert [status for status, _ in outcomes] == [201, 201]
        assert all(body["status"] == "COMPLETED" for _, body in outcomes)
        assert fake_client.calls == 2
        recognition_ids = {UUID(body["recognitionId"]) for _, body in outcomes}
        assert len(recognition_ids) == 2
        with Session(engine) as db:
            models = db.scalars(
                select(ModelVersion).where(
                    ModelVersion.model_name == "bilstm",
                    ModelVersion.version == "bilstm-v1",
                )
            ).all()
            model_version_ids = set(
                db.scalars(
                    select(Interpretation.model_version_id).where(
                        Interpretation.recognition_id.in_(recognition_ids)
                    )
                ).all()
            )
        assert len(models) == 1
        assert model_version_ids == {models[0].id}
    finally:
        engine.dispose()


def test_patient_and_doctor_messages_preserve_provenance_and_acknowledgement() -> None:
    with TestClient(app) as client:
        _, _, patient_session_id, doctor_session_id, conversation_id = _consultation(client)
        patient_message = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "PATIENT",
                "session_id": patient_session_id,
                "message_type": "TEXT",
                "text": "Synthetic patient text",
            },
        )
        assert patient_message.status_code == 201, patient_message.text
        message_id = patient_message.json()["messageId"]
        acknowledged = client.post(
            f"/api/v1/messages/{message_id}/acknowledge",
            json={"session_id": doctor_session_id},
        )
        assert acknowledged.status_code == 200
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id, "limit": 1},
        )
        row = history.json()["messages"][0]
        assert row["provenance"] == "PATIENT_TYPED"
        assert row["verificationStatus"] == "PATIENT_CONFIRMED"
        assert row["confirmedAt"] == row["createdAt"]
        assert row["acknowledgedAt"] is not None
        doctor_message = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "DOCTOR",
                "session_id": doctor_session_id,
                "message_type": "TEXT",
                "text": "Synthetic doctor question",
            },
        )
        assert doctor_message.status_code == 201
        wrong_role = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "DOCTOR",
                "session_id": patient_session_id,
                "message_type": "TEXT",
                "text": "Not accepted",
            },
        )
        assert wrong_role.status_code == 403


def test_patient_message_status_is_derived_not_client_supplied() -> None:
    with TestClient(app) as client:
        _, _, patient_session_id, _, conversation_id = _consultation(client)
        posted = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "PATIENT",
                "session_id": patient_session_id,
                "message_type": "TEXT",
                "text": "Explicitly submitted patient text",
                "verification_status": "UNVERIFIED",
                "confirmed_at": None,
            },
        )
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )

    assert posted.status_code == 201
    assert len(history.json()["messages"]) == 1
    message = history.json()["messages"][0]
    assert message["verificationStatus"] == "PATIENT_CONFIRMED"
    assert message["confirmedAt"] == message["createdAt"]


def test_doctor_history_filters_unconfirmed_patient_messages_before_pagination() -> None:
    with TestClient(app) as client:
        _, _, patient_session_id, doctor_session_id, conversation_id = _consultation(client)
        patient_post = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "PATIENT",
                "session_id": patient_session_id,
                "message_type": "TEXT",
                "text": "Confirmed patient text",
            },
        )
        doctor_post = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={
                "sender_type": "DOCTOR",
                "session_id": doctor_session_id,
                "message_type": "TEXT",
                "text": "Doctor question",
            },
        )
        assert patient_post.status_code == doctor_post.status_code == 201

        unconfirmed_message_id = uuid4()
        with Session(get_engine(TEST_DATABASE_URL)) as db:
            db.add(
                Message(
                    id=unconfirmed_message_id,
                    conversation_id=UUID(conversation_id),
                    sender_type="PATIENT",
                    sender_session_id=UUID(patient_session_id),
                    message_type="TEXT",
                    text="Unconfirmed patient draft",
                    provenance="PATIENT_TYPED",
                    verification_status="UNVERIFIED",
                    created_at=datetime.now(UTC),
                )
            )
            db.commit()

        first_page = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": doctor_session_id, "limit": 1, "offset": 0},
        )
        second_page = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": doctor_session_id, "limit": 1, "offset": 1},
        )
        patient_history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )
        cannot_acknowledge = client.post(
            f"/api/v1/messages/{unconfirmed_message_id}/acknowledge",
            json={"session_id": doctor_session_id},
        )

    assert first_page.status_code == second_page.status_code == 200
    assert first_page.json()["total"] == second_page.json()["total"] == 2
    doctor_messages = first_page.json()["messages"] + second_page.json()["messages"]
    assert {message["text"] for message in doctor_messages} == {
        "Confirmed patient text",
        "Doctor question",
    }
    assert patient_history.status_code == 200
    assert len(patient_history.json()["messages"]) == 3
    assert cannot_acknowledge.status_code == 409
    assert cannot_acknowledge.json()["code"] == "MESSAGE_NOT_ACKNOWLEDGEABLE"


def test_confirmation_creates_exact_patient_confirmed_message_and_retries_idempotently() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id, "Exact candidate text"
        )
        idempotency_key = str(uuid4())
        payload = {
            "candidate_id": candidate_id,
            "session_id": patient_session_id,
            "idempotency_key": idempotency_key,
        }
        first = client.post(f"/api/v1/interpretations/{interpretation_id}/confirm", json=payload)
        retry = client.post(f"/api/v1/interpretations/{interpretation_id}/confirm", json=payload)
        assert first.status_code == 201, first.text
        assert retry.status_code == 201, retry.text
        assert first.json()["confirmationId"] == retry.json()["confirmationId"]
        assert first.json()["createdMessageId"] == retry.json()["createdMessageId"]
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )
        with Session(get_engine(TEST_DATABASE_URL)) as db:
            confirmation = db.get(Confirmation, UUID(first.json()["confirmationId"]))
            message_record = db.get(Message, UUID(first.json()["createdMessageId"]))
        assert len(history.json()["messages"]) == 1
        message = history.json()["messages"][0]
        assert message["text"] == "Exact candidate text"
        assert message["verificationStatus"] == "PATIENT_CONFIRMED"
        assert message["provenance"] == "AI_CANDIDATE_CONFIRMED"
        assert confirmation is not None
        assert message_record is not None
        assert message_record.confirmed_at is not None
        assert _as_utc(confirmation.created_at) == _as_utc(message_record.created_at)
        assert _as_utc(confirmation.created_at) == _as_utc(message_record.confirmed_at)


def test_concurrent_confirmation_http_requests_create_one_patient_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(
        URL.create(
            "sqlite+pysqlite",
            database=str(tmp_path / "concurrent-confirmation.db"),
        ),
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(engine)

    now = datetime.now(UTC)
    consultation_id = uuid4()
    session_id = uuid4()
    conversation_id = uuid4()
    recognition_id = uuid4()
    interpretation_id = uuid4()
    candidate_id = uuid4()
    with Session(engine) as db:
        db.add_all(
            [
                Consultation(
                    id=consultation_id,
                    status="ACTIVE",
                    join_code_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
                ConsultationSessionEntity(
                    id=session_id,
                    consultation_id=consultation_id,
                    session_type="PATIENT_DEVICE",
                    session_token_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
                Conversation(
                    id=conversation_id,
                    consultation_id=consultation_id,
                    status="ACTIVE",
                    created_at=now,
                ),
                Recognition(
                    id=recognition_id,
                    consultation_id=consultation_id,
                    session_id=session_id,
                    client_request_id="concurrent-http-confirmation",
                    status="COMPLETED",
                    media_metadata={},
                    created_at=now,
                ),
                Interpretation(
                    id=interpretation_id,
                    recognition_id=recognition_id,
                    status="UNVERIFIED",
                    quality_status="ACCEPTABLE",
                    created_at=now,
                ),
                InterpretationCandidate(
                    id=candidate_id,
                    interpretation_id=interpretation_id,
                    rank=1,
                    text="Concurrent candidate",
                    confidence=0.5,
                    selected=False,
                    created_at=now,
                ),
            ]
        )
        db.commit()

    def get_concurrent_test_db() -> Generator[Session, None, None]:
        with Session(engine) as db:
            yield db

    monkeypatch.setitem(
        app.dependency_overrides,
        confirmation_routes.get_db,
        get_concurrent_test_db,
    )
    request_payload = {
        "candidate_id": str(candidate_id),
        "session_id": str(session_id),
        "idempotency_key": str(uuid4()),
    }
    request_barrier = Barrier(2)

    def post_confirmation() -> tuple[int, dict[str, str]]:
        client = TestClient(app)
        try:
            request_barrier.wait(timeout=10)
            response = client.post(
                f"/api/v1/interpretations/{interpretation_id}/confirm",
                json=request_payload,
            )
            return response.status_code, response.json()
        finally:
            client.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_request = executor.submit(post_confirmation)
            second_request = executor.submit(post_confirmation)
            responses = (first_request.result(timeout=30), second_request.result(timeout=30))

        published_state: list[tuple[str, bool, bool]] = []

        async def inspect_published_confirmation(delivery: OutboxDelivery) -> None:
            with Session(engine) as publisher_db:
                confirmation = publisher_db.scalar(
                    select(Confirmation).where(
                        Confirmation.created_message_id == delivery.resource_id
                    )
                )
                message = publisher_db.get(Message, delivery.resource_id)
            published_state.append(
                (delivery.event_type, confirmation is not None, message is not None)
            )

        dispatcher = OutboxDispatcher(
            sessionmaker(bind=engine, class_=Session, expire_on_commit=False),
            inspect_published_confirmation,
        )
        dispatched_count = asyncio.run(
            dispatcher.dispatch_once(now + timedelta(seconds=30))
        )

        with Session(engine) as db:
            interpretation_status = db.scalar(
                select(Interpretation.status).where(Interpretation.id == interpretation_id)
            )
            confirmations = db.scalars(
                select(Confirmation).where(Confirmation.interpretation_id == interpretation_id)
            ).all()
            patient_messages = db.scalars(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.sender_type == "PATIENT",
                )
            ).all()
            confirmation_audits = db.scalars(
                select(AuditEvent).where(
                    AuditEvent.resource_id == interpretation_id,
                    AuditEvent.action == "CONFIRM_INTERPRETATION",
                )
            ).all()
            confirmation_outbox_events = db.scalars(
                select(OutboxEvent).where(
                    OutboxEvent.consultation_id == consultation_id,
                    OutboxEvent.event_type == "confirmation.received",
                )
            ).all()
    finally:
        engine.dispose()

    assert interpretation_status == "CONFIRMED"
    assert responses[0][0] == responses[1][0] == 201
    assert responses[0][1]["confirmationId"] == responses[1][1]["confirmationId"]
    assert responses[0][1]["createdMessageId"] == responses[1][1]["createdMessageId"]
    assert _as_utc(datetime.fromisoformat(responses[0][1]["createdAt"])) == _as_utc(
        datetime.fromisoformat(responses[1][1]["createdAt"])
    )
    assert dispatched_count == 1
    assert published_state == [("confirmation.received", True, True)]
    assert len(confirmations) == 1
    assert len(patient_messages) == 1
    assert len(confirmation_audits) == 1
    assert len(confirmation_outbox_events) == 1
    assert responses[0][1]["createdMessageId"] == str(patient_messages[0].id)


def test_confirmation_rejects_cross_session_and_candidate_mismatch() -> None:
    with TestClient(app) as client:
        consultation_id, join_code, patient_session_id, doctor_session_id, _ = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        second_session = _session(client, consultation_id, join_code, "PATIENT_DEVICE")
        mismatch = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": second_session,
                "idempotency_key": str(uuid4()),
            },
        )
        assert mismatch.status_code == 401

        other_consultation = client.post(
            "/api/v1/consultations",
            json={"expires_in_minutes": 120},
        ).json()
        foreign_session = _session(
            client,
            other_consultation["consultationId"],
            other_consultation["joinCode"],
            "PATIENT_DEVICE",
        )
        foreign = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": foreign_session,
                "idempotency_key": str(uuid4()),
            },
        )
        wrong_role = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": doctor_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        assert foreign.status_code == 401
        assert wrong_role.status_code == 403


@pytest.mark.parametrize(
    ("session_state", "expected_code"),
    [
        ("expired", "SESSION_EXPIRED"),
        ("revoked", "SESSION_REVOKED"),
    ],
)
def test_confirmation_rejects_expired_or_revoked_patient_sessions_without_writes(
    session_state: str,
    expected_code: str,
) -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        now = datetime.now(UTC)
        session_updates = {
            "expired": {"expires_at": now - timedelta(seconds=1)},
            "revoked": {"revoked_at": now},
        }
        with Session(get_engine(TEST_DATABASE_URL)) as db:
            db.execute(
                update(ConsultationSessionEntity)
                .where(ConsultationSessionEntity.id == UUID(patient_session_id))
                .values(**session_updates[session_state])
            )
            db.commit()

        rejected = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )

    assert rejected.status_code == 401
    assert rejected.json()["code"] == expected_code
    _assert_confirmation_attempt_left_no_writes(
        consultation_id,
        interpretation_id,
        candidate_id,
        conversation_id,
    )


def test_candidate_must_belong_to_interpretation() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        first_interpretation, first_candidate = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        _, second_candidate = _seed_confirmable_interpretation(consultation_id, patient_session_id)
        response = client.post(
            f"/api/v1/interpretations/{first_interpretation}/confirm",
            json={
                "candidate_id": second_candidate,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        assert first_candidate != second_candidate
        assert response.status_code == 400
        assert response.json()["code"] == "CANDIDATE_NOT_FOUND"


def test_confirmation_rejects_candidate_from_another_consultation_without_writes() -> None:
    with TestClient(app) as client:
        (
            consultation_id,
            _,
            patient_session_id,
            _,
            conversation_id,
        ) = _consultation(client)
        (
            foreign_consultation_id,
            _,
            foreign_patient_session_id,
            _,
            foreign_conversation_id,
        ) = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        foreign_interpretation_id, foreign_candidate_id = _seed_confirmable_interpretation(
            foreign_consultation_id, foreign_patient_session_id
        )
        rejected = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": foreign_candidate_id,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )

    assert rejected.status_code == 400
    assert rejected.json()["code"] == "CANDIDATE_NOT_FOUND"
    _assert_confirmation_attempt_left_no_writes(
        consultation_id,
        interpretation_id,
        candidate_id,
        conversation_id,
    )
    _assert_confirmation_attempt_left_no_writes(
        foreign_consultation_id,
        foreign_interpretation_id,
        foreign_candidate_id,
        foreign_conversation_id,
    )


def test_correction_rejects_candidate_from_another_interpretation() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, _ = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        _, foreign_candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        response = client.post(
            f"/api/v1/interpretations/{interpretation_id}/correct",
            json={
                "candidate_id": foreign_candidate_id,
                "corrected_text": "Patient correction",
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "CANDIDATE_NOT_FOUND"
    assert history.json()["messages"] == []


def test_confirmation_rejects_resolved_interpretation_and_correction() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        first = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        duplicate = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        correction = client.post(
            f"/api/v1/interpretations/{interpretation_id}/correct",
            json={
                "candidate_id": candidate_id,
                "corrected_text": "A later correction",
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )

    assert first.status_code == 201
    assert duplicate.status_code == correction.status_code == 409
    assert len(history.json()["messages"]) == 1


def test_confirmation_idempotency_key_conflicts_on_changed_payload() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        key = str(uuid4())
        original = {
            "candidate_id": candidate_id,
            "corrected_text": "First correction",
            "session_id": patient_session_id,
            "idempotency_key": key,
        }
        first = client.post(
            f"/api/v1/interpretations/{interpretation_id}/correct",
            json=original,
        )
        changed = client.post(
            f"/api/v1/interpretations/{interpretation_id}/correct",
            json={**original, "corrected_text": "Different correction"},
        )

    assert first.status_code == 201
    assert changed.status_code == 409
    assert changed.json()["code"] == "IDEMPOTENCY_CONFLICT"


def test_confirmation_database_failure_rolls_back_all_writes(monkeypatch) -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        enqueue_notification = confirmation_routes.enqueue_notification

        def enqueue_then_fail(
            db: Session,
            outbox_consultation_id: UUID,
            event_type: str,
            resource_id: UUID,
            created_at: datetime,
        ) -> None:
            enqueue_notification(
                db,
                outbox_consultation_id,
                event_type,
                resource_id,
                created_at,
            )
            raise SQLAlchemyError("injected outbox failure")

        monkeypatch.setattr(confirmation_routes, "enqueue_notification", enqueue_then_fail)
        response = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )

    assert response.status_code == 500
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        interpretation = db.get(Interpretation, UUID(interpretation_id))
        confirmations = db.scalars(
            select(Confirmation).where(Confirmation.interpretation_id == UUID(interpretation_id))
        ).all()
        messages = db.scalars(
            select(Message).where(Message.conversation_id == UUID(conversation_id))
        ).all()
        audits = db.scalars(
            select(AuditEvent).where(
                AuditEvent.resource_id == UUID(interpretation_id),
                AuditEvent.action == "CONFIRM_INTERPRETATION",
            )
        ).all()
        outbox_events = db.scalars(
            select(OutboxEvent).where(
                OutboxEvent.consultation_id == UUID(consultation_id),
                OutboxEvent.event_type == "confirmation.received",
            )
        ).all()

    assert interpretation is not None
    assert interpretation.status == "UNVERIFIED"
    assert confirmations == []
    assert messages == []
    assert audits == []
    assert outbox_events == []


def test_rejecting_unverified_interpretation_creates_no_message() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, _ = _seed_confirmable_interpretation(consultation_id, patient_session_id)
        rejected = client.post(
            f"/api/v1/interpretations/{interpretation_id}/reject",
            json={"session_id": patient_session_id, "reason": "NONE_OF_THESE"},
        )
        assert rejected.status_code == 200, rejected.text
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )
        assert history.status_code == 200
        assert history.json()["messages"] == []


def test_requesting_resign_moves_state_without_creating_a_message() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, _ = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        requested = client.post(
            f"/api/v1/interpretations/{interpretation_id}/request-clarification",
            json={"session_id": patient_session_id},
        )
        interpretation = client.get(
            f"/api/v1/interpretations/{interpretation_id}",
            params={"sessionId": patient_session_id},
        )
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )

    assert requested.status_code == 200
    assert requested.json()["status"] == "NEEDS_CLARIFICATION"
    assert interpretation.json()["status"] == "NEEDS_CLARIFICATION"
    assert history.json()["messages"] == []


def test_rejected_interpretation_cannot_later_be_confirmed() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id
        )
        rejected = client.post(
            f"/api/v1/interpretations/{interpretation_id}/reject",
            json={"session_id": patient_session_id},
        )
        confirmed = client.post(
            f"/api/v1/interpretations/{interpretation_id}/confirm",
            json={
                "candidate_id": candidate_id,
                "session_id": patient_session_id,
                "idempotency_key": str(uuid4()),
            },
        )
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )

    assert rejected.status_code == 200
    assert confirmed.status_code == 409
    assert history.json()["messages"] == []


def test_patient_correction_is_distinct_from_mock_text_and_reject_creates_no_message() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, conversation_id = _consultation(client)
        interpretation_id, candidate_id = _seed_confirmable_interpretation(
            consultation_id, patient_session_id, "Original candidate"
        )
        correction_payload = {
            "candidate_id": candidate_id,
            "corrected_text": "Patient supplied correction",
            "session_id": patient_session_id,
            "idempotency_key": str(uuid4()),
        }
        corrected = client.post(
            f"/api/v1/interpretations/{interpretation_id}/correct",
            json=correction_payload,
        )
        retry = client.post(
            f"/api/v1/interpretations/{interpretation_id}/correct",
            json=correction_payload,
        )
        assert corrected.status_code == 201, corrected.text
        assert retry.status_code == 201, retry.text
        assert corrected.json()["confirmationId"] == retry.json()["confirmationId"]
        assert corrected.json()["createdMessageId"] == retry.json()["createdMessageId"]
        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"sessionId": patient_session_id},
        )
        corrected_message = history.json()["messages"][0]
        assert corrected_message["text"] == "Patient supplied correction"
        assert corrected_message["provenance"] == "PATIENT_CORRECTED"

        unsupported = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("unsupported", patient_session_id),
        )
        assert unsupported.status_code == 409


def test_websocket_receives_post_commit_event_and_rest_remains_authoritative() -> None:
    with TestClient(app) as client:
        created = client.post("/api/v1/consultations", json={"expires_in_minutes": 120}).json()
        consultation_id = created["consultationId"]
        session_id = _session(client, consultation_id, created["joinCode"], "PATIENT_DEVICE")
        with client.websocket_connect(
            f"/api/v1/ws/consultations/{consultation_id}?sessionId={session_id}"
        ) as websocket:
            started = client.post(f"/api/v1/consultations/{consultation_id}/start")
            assert started.status_code == 200
            event = websocket.receive_json()
        assert event["type"] == "consultation.started"
        assert event["eventId"]
        assert event["consultationId"] == consultation_id
        assert "text" not in event
        assert client.get(f"/api/v1/consultations/{consultation_id}").status_code == 200


def test_state_change_commits_its_notification_to_the_outbox() -> None:
    with TestClient(app) as client:
        consultation_id, _, _, _, _ = _consultation(client)

    with Session(get_engine(TEST_DATABASE_URL)) as db:
        event = db.scalar(
            select(OutboxEvent).where(
                OutboxEvent.consultation_id == UUID(consultation_id),
                OutboxEvent.event_type == "consultation.started",
            )
        )

    assert event is not None
    assert event.status in {"PENDING", "PROCESSING", "PUBLISHED"}


def test_websocket_rejects_session_for_another_consultation() -> None:
    with TestClient(app) as client:
        first = client.post("/api/v1/consultations", json={"expiresInMinutes": 120}).json()
        second = client.post("/api/v1/consultations", json={"expiresInMinutes": 120}).json()
        session_id = _session(
            client,
            first["consultationId"],
            first["joinCode"],
            "PATIENT_DEVICE",
        )
        try:
            with client.websocket_connect(
                f"/api/v1/ws/consultations/{second['consultationId']}?sessionId={session_id}"
            ):
                raise AssertionError("Mismatched WebSocket session unexpectedly connected")
        except WebSocketDisconnect as error:
            assert error.code == 1008
