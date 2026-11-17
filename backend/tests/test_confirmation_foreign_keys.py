"""Exercise confirmation with immediate foreign keys, including opt-in PostgreSQL."""

import os
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.api.routes.confirmations as routes
from app.infrastructure.persistence.entities import (
    AuditEvent,
    Base,
    Confirmation,
    Consultation,
    ConsultationSession,
    Conversation,
    Interpretation,
    InterpretationCandidate,
    Message,
    OutboxEvent,
    Recognition,
)
from app.main import app


@pytest.fixture(params=["sqlite", "postgresql"])
def confirmation_engine(request, tmp_path) -> Generator[Engine, None, None]:
    if request.param == "postgresql":
        database_url = os.environ.get("CLINICAL_TEST_POSTGRES_URL")
        if not database_url:
            pytest.skip("Set CLINICAL_TEST_POSTGRES_URL to a dedicated PostgreSQL test database")
        assert make_url(database_url).get_backend_name() == "postgresql"
        engine = create_engine(database_url, hide_parameters=True)
    else:
        engine = create_engine(
            f"sqlite+pysqlite:///{(tmp_path / 'confirmation.db').as_posix()}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def confirmation_context(confirmation_engine) -> tuple[UUID, UUID, UUID]:
    now = datetime.now(UTC)
    consultation_id, session_id, recognition_id = uuid4(), uuid4(), uuid4()
    interpretation_id, candidate_id = uuid4(), uuid4()
    with Session(confirmation_engine) as db:
        db.add(
            Consultation(
                id=consultation_id,
                status="ACTIVE",
                join_code_hash="synthetic-hash",
                expires_at=now + timedelta(hours=1),
            )
        )
        db.flush()
        db.add(
            ConsultationSession(
                id=session_id,
                consultation_id=consultation_id,
                session_type="PATIENT_DEVICE",
                session_token_hash="synthetic-hash",
                expires_at=now + timedelta(hours=1),
            )
        )
        db.add(
            Conversation(
                id=uuid4(),
                consultation_id=consultation_id,
                status="ACTIVE",
            )
        )
        db.flush()
        db.add(
            Recognition(
                id=recognition_id,
                consultation_id=consultation_id,
                session_id=session_id,
                client_request_id=str(uuid4()),
                status="COMPLETED",
                media_metadata={},
            )
        )
        db.flush()
        db.add(
            Interpretation(
                id=interpretation_id,
                recognition_id=recognition_id,
                status="UNVERIFIED",
                quality_status="ACCEPTABLE",
            )
        )
        db.flush()
        db.add(
            InterpretationCandidate(
                id=candidate_id,
                interpretation_id=interpretation_id,
                rank=1,
                text="Synthetic candidate",
                selected=False,
            )
        )
        db.commit()
    return interpretation_id, candidate_id, session_id


@pytest.fixture
def confirmation_client(confirmation_engine, monkeypatch) -> Generator[TestClient, None, None]:
    def get_db():
        with Session(confirmation_engine) as db:
            yield db

    monkeypatch.setitem(app.dependency_overrides, routes.get_db, get_db)
    # Do not start the lifespan dispatcher: these tests inspect the committed outbox.
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()


def test_original_dangling_message_reference_fails_on_query_autoflush(
    confirmation_engine,
    confirmation_context,
) -> None:
    interpretation_id, candidate_id, session_id = confirmation_context
    with Session(confirmation_engine) as db:
        db.add(
            Confirmation(
                id=uuid4(),
                interpretation_id=interpretation_id,
                candidate_id=candidate_id,
                session_id=session_id,
                action="CONFIRM",
                created_message_id=uuid4(),
                idempotency_key=uuid4(),
            )
        )
        with pytest.raises(IntegrityError):
            db.execute(select(Conversation))
        db.rollback()


@pytest.mark.parametrize("action", ["confirm", "correct"])
def test_confirmation_links_both_foreign_keys_and_replays_once(
    confirmation_engine,
    confirmation_context,
    confirmation_client,
    action,
) -> None:
    interpretation_id, candidate_id, session_id = confirmation_context
    payload = {
        "candidateId": str(candidate_id),
        "sessionId": str(session_id),
        "idempotencyKey": str(uuid4()),
        "correctedText": "Synthetic correction",
    }
    url = f"/api/v1/interpretations/{interpretation_id}/{action}"
    first = confirmation_client.post(url, json=payload)
    retry = confirmation_client.post(url, json=payload)

    assert first.status_code == retry.status_code == 201, first.text
    assert first.json()["confirmationId"] == retry.json()["confirmationId"]
    assert first.json()["createdMessageId"] == retry.json()["createdMessageId"]
    assert datetime.fromisoformat(first.json()["createdAt"]).replace(
        tzinfo=UTC
    ) == datetime.fromisoformat(retry.json()["createdAt"]).replace(tzinfo=UTC)
    with Session(confirmation_engine) as db:
        confirmation = db.scalars(
            select(Confirmation).where(Confirmation.interpretation_id == interpretation_id)
        ).one()
        message = db.scalars(
            select(Message).where(Message.confirmation_id == confirmation.id)
        ).one()
        interpretation = db.get(Interpretation, interpretation_id)
        candidate = db.get(InterpretationCandidate, candidate_id)
        assert confirmation.created_message_id == message.id
        assert str(message.id) == first.json()["createdMessageId"]
        assert message.verification_status == "PATIENT_CONFIRMED"
        assert message.text == (
            "Synthetic candidate" if action == "confirm" else "Synthetic correction"
        )
        assert interpretation.status == "CONFIRMED"
        assert candidate.selected is True
        assert (
            db.scalars(select(AuditEvent).where(AuditEvent.resource_id == interpretation_id))
            .one()
            .outcome
            == "SUCCESS"
        )
        assert (
            db.scalars(select(OutboxEvent).where(OutboxEvent.resource_id == message.id))
            .one()
            .status
            == "PENDING"
        )


@pytest.mark.parametrize("action", ["confirm", "correct"])
def test_failure_after_flushed_message_rolls_back_and_logs_safe_traceback(
    confirmation_engine,
    confirmation_context,
    confirmation_client,
    monkeypatch,
    caplog,
    action,
) -> None:
    interpretation_id, candidate_id, session_id = confirmation_context
    correlation_id = str(uuid4())
    enqueue = routes.enqueue_notification
    error = IntegrityError(
        "PRIVATE_SQL",
        {"text": "PRIVATE_PATIENT_TEXT", "token": "PRIVATE_SESSION_TOKEN"},
        ValueError("PRIVATE_DRIVER_DETAILS"),
    )
    error.__cause__ = ValueError("PRIVATE_CHAINED_EXCEPTION")

    def enqueue_then_fail(db, consultation_id, event_type, resource_id, created_at):
        enqueue(db, consultation_id, event_type, resource_id, created_at)
        db.flush()
        raise error

    monkeypatch.setattr(routes, "enqueue_notification", enqueue_then_fail)
    response = confirmation_client.post(
        f"/api/v1/interpretations/{interpretation_id}/{action}",
        headers={"X-Correlation-ID": correlation_id},
        json={
            "candidateId": str(candidate_id),
            "sessionId": str(session_id),
            "idempotencyKey": str(uuid4()),
            "correctedText": "Synthetic correction",
        },
    )

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert response.json()["correlationId"] == correlation_id
    assert correlation_id in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "error_type=IntegrityError" in caplog.text
    assert "SQLAlchemyError" in caplog.text
    # The source line and the original exception message must both remain private.
    assert "PRIVATE_" not in caplog.text
    with Session(confirmation_engine) as db:
        interpretation = db.get(Interpretation, interpretation_id)
        candidate = db.get(InterpretationCandidate, candidate_id)
        consultation_id = db.get(Recognition, interpretation.recognition_id).consultation_id
        conversation_id = db.scalars(
            select(Conversation.id).where(Conversation.consultation_id == consultation_id)
        ).one()
        assert interpretation.status == "UNVERIFIED"
        assert interpretation.resolved_at is None
        assert candidate.selected is False
        assert (
            db.scalars(
                select(Confirmation).where(Confirmation.interpretation_id == interpretation_id)
            ).all()
            == []
        )
        assert (
            db.scalars(select(Message).where(Message.conversation_id == conversation_id)).all()
            == []
        )
        assert (
            db.scalars(select(AuditEvent).where(AuditEvent.resource_id == interpretation_id)).all()
            == []
        )
        assert (
            db.scalars(
                select(OutboxEvent).where(OutboxEvent.consultation_id == consultation_id)
            ).all()
            == []
        )
