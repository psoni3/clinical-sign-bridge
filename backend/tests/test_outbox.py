from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.persistence.entities import Base, OutboxEvent
from app.infrastructure.persistence.outbox import OutboxDelivery, OutboxDispatcher


@pytest.fixture
def outbox_sessions() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.mark.anyio
async def test_dispatcher_retries_failure_and_publishes_event(
    outbox_sessions: sessionmaker[Session],
) -> None:
    now = datetime(2026, 10, 31, tzinfo=UTC)
    event_id = uuid4()
    consultation_id = uuid4()
    resource_id = uuid4()
    with outbox_sessions.begin() as db:
        db.add(
            OutboxEvent(
                id=event_id,
                consultation_id=consultation_id,
                event_type="message.received",
                resource_id=resource_id,
                status="PENDING",
                attempts=0,
                available_at=now,
                created_at=now,
            )
        )

    delivered: list[OutboxDelivery] = []
    calls = 0

    async def publish(delivery: OutboxDelivery) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary websocket failure")
        delivered.append(delivery)

    dispatcher = OutboxDispatcher(outbox_sessions, publish)

    assert await dispatcher.dispatch_once(now) == 1
    with outbox_sessions() as db:
        retry_event = db.get(OutboxEvent, event_id)
        assert retry_event is not None
        assert retry_event.status == "PENDING"
        assert retry_event.attempts == 1
        assert retry_event.available_at.replace(tzinfo=UTC) == now + timedelta(seconds=1)

    assert await dispatcher.dispatch_once(now + timedelta(seconds=2)) == 1
    with outbox_sessions() as db:
        published_event = db.get(OutboxEvent, event_id)
        assert published_event is not None
        assert published_event.status == "PUBLISHED"
        assert published_event.attempts == 2
        assert published_event.claim_token is None

    assert [delivery.event_id for delivery in delivered] == [event_id]
    assert delivered[0].consultation_id == consultation_id
    assert delivered[0].resource_id == resource_id


def test_outbox_event_rolls_back_with_business_transaction(
    outbox_sessions: sessionmaker[Session],
) -> None:
    event_id = uuid4()
    with pytest.raises(RuntimeError, match="business transaction failed"):
        with outbox_sessions.begin() as db:
            db.add(
                OutboxEvent(
                    id=event_id,
                    consultation_id=uuid4(),
                    event_type="message.received",
                    resource_id=uuid4(),
                    status="PENDING",
                    attempts=0,
                    available_at=datetime.now(UTC),
                )
            )
            raise RuntimeError("business transaction failed")

    with outbox_sessions() as db:
        assert db.scalar(select(OutboxEvent).where(OutboxEvent.id == event_id)) is None


@pytest.mark.anyio
async def test_dispatcher_marks_event_dead_after_retry_limit(
    outbox_sessions: sessionmaker[Session],
) -> None:
    now = datetime(2026, 10, 31, tzinfo=UTC)
    event_id = uuid4()
    with outbox_sessions.begin() as db:
        db.add(
            OutboxEvent(
                id=event_id,
                consultation_id=uuid4(),
                event_type="message.received",
                resource_id=uuid4(),
                status="PENDING",
                attempts=9,
                available_at=now,
                created_at=now,
            )
        )

    async def fail(_delivery: OutboxDelivery) -> None:
        raise RuntimeError("terminal notification failure")

    dispatcher = OutboxDispatcher(outbox_sessions, fail)

    assert await dispatcher.dispatch_once(now) == 1
    with outbox_sessions() as db:
        dead_event = db.get(OutboxEvent, event_id)
        assert dead_event is not None
        assert dead_event.status == "DEAD"
        assert dead_event.last_error_code == "RETRY_LIMIT_EXCEEDED"
