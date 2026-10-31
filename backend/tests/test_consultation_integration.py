"""M1 lifecycle, consent, messaging, recognition, and safety integration tests."""

from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.entities import (
    Base,
    Interpretation,
    InterpretationCandidate,
    ModelVersion,
    OutboxEvent,
    Recognition,
)
from app.main import app

TEST_DATABASE_URL = "sqlite+pysqlite:///:memory:"


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
    # Container signature is enough for the deterministic, non-recognizing mock.
    content = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 24
    return {
        "video": ("synthetic.mp4", BytesIO(content), "video/mp4"),
        "clientRequestId": (None, client_request_id),
        "sessionId": (None, session_id),
    }


def _seed_confirmable_interpretation(
    consultation_id: str,
    session_id: str,
    text: str = "Synthetic candidate",
) -> tuple[str, str]:
    recognition_id = uuid4()
    model_id = uuid4()
    interpretation_id = uuid4()
    candidate_id = uuid4()
    now = datetime.now(UTC)
    with Session(get_engine(TEST_DATABASE_URL)) as db:
        db.add(
            ModelVersion(
                id=model_id,
                model_name="test-only",
                version="test-v1",
                metrics={},
                approval_status="TEST_ONLY",
                created_at=now,
            )
        )
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
                model_version_id=model_id,
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


def test_recognition_requires_live_consent_and_abstains_without_candidates() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        missing_consent = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("no-consent", patient_session_id),
        )
        assert missing_consent.status_code == 409
        _give_processing_consent(client, consultation_id, patient_session_id)
        uploaded = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files=_video_upload("request-1", patient_session_id),
        )
        assert uploaded.status_code == 201, uploaded.text
        result = uploaded.json()
        assert result["status"] == "UNSUPPORTED"
        assert result["interpretationId"]
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
        assert interpretation.json()["status"] == "NEEDS_CLARIFICATION"
        assert interpretation.json()["candidates"] == []


def test_recognition_rejects_bad_container_and_is_idempotent() -> None:
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
            "The recorded segment is empty or not a supported video format."
        )
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


def test_recognition_accepts_webm_with_recorder_codec_parameter() -> None:
    with TestClient(app) as client:
        consultation_id, _, patient_session_id, _, _ = _consultation(client)
        _give_processing_consent(client, consultation_id, patient_session_id)
        uploaded = client.post(
            f"/api/v1/consultations/{consultation_id}/recognitions",
            files={
                "video": (
                    "signing-segment.webm",
                    BytesIO(b"\x1a\x45\xdf\xa3" + b"\x00" * 32),
                    "video/webm;codecs=vp8",
                ),
                "clientRequestId": (None, "webm-with-codec"),
                "sessionId": (None, patient_session_id),
            },
        )

    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "UNSUPPORTED"


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
        assert len(history.json()["messages"]) == 1
        message = history.json()["messages"][0]
        assert message["text"] == "Exact candidate text"
        assert message["verificationStatus"] == "PATIENT_CONFIRMED"
        assert message["provenance"] == "AI_CANDIDATE_CONFIRMED"


def test_confirmation_rejects_cross_session_and_candidate_mismatch() -> None:
    with TestClient(app) as client:
        consultation_id, join_code, patient_session_id, _, _ = _consultation(client)
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
