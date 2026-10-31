"""Transactional notification outbox persistence and delivery."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.persistence.entities import OutboxEvent

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 10
MAX_BACKOFF_SECONDS = 300
CLAIM_LEASE_SECONDS = 30
DISPATCH_BATCH_SIZE = 50
DISPATCH_POLL_SECONDS = 1
MAX_POLL_BACKOFF_SECONDS = 30


@dataclass(frozen=True)
class OutboxDelivery:
    event_id: UUID
    consultation_id: UUID
    event_type: str
    resource_id: UUID
    created_at: datetime
    claim_token: UUID
    attempts: int


@dataclass(frozen=True)
class OutboxHealth:
    pending: int
    processing: int
    dead: int
    oldest_unpublished_seconds: int | None


NotificationPublisher = Callable[[OutboxDelivery], Awaitable[None]]


def enqueue_notification(
    db: Session,
    consultation_id: UUID,
    event_type: str,
    resource_id: UUID,
    created_at: datetime,
) -> None:
    db.add(
        OutboxEvent(
            id=uuid4(),
            consultation_id=consultation_id,
            event_type=event_type,
            resource_id=resource_id,
            status="PENDING",
            attempts=0,
            available_at=created_at,
            created_at=created_at,
        )
    )


class OutboxDispatcher:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        publisher: NotificationPublisher,
    ) -> None:
        self._session_factory = session_factory
        self._publisher = publisher

    async def dispatch_once(self, now: datetime | None = None) -> int:
        dispatch_time = now or datetime.now(UTC)
        deliveries = self._claim_due_events(dispatch_time)
        for delivery in deliveries:
            try:
                await self._publisher(delivery)
            except Exception as error:
                self._mark_failed(delivery, dispatch_time)
                log_failure = logger.error if delivery.attempts >= MAX_ATTEMPTS else logger.warning
                log_failure(
                    "Outbox notification delivery failed event_id=%s attempt=%s "
                    "terminal=%s error_type=%s",
                    delivery.event_id,
                    delivery.attempts,
                    delivery.attempts >= MAX_ATTEMPTS,
                    type(error).__name__,
                )
            else:
                self._mark_published(delivery, dispatch_time)
        return len(deliveries)

    async def run(self, stop_event: asyncio.Event) -> None:
        poll_delay = DISPATCH_POLL_SECONDS
        while not stop_event.is_set():
            try:
                await self.dispatch_once()
            except SQLAlchemyError as error:
                logger.error(
                    "Outbox database operation failed error_type=%s",
                    type(error).__name__,
                )
                poll_delay = min(poll_delay * 2, MAX_POLL_BACKOFF_SECONDS)
            else:
                poll_delay = DISPATCH_POLL_SECONDS
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_delay)
            except TimeoutError:
                continue

    def _claim_due_events(self, now: datetime) -> list[OutboxDelivery]:
        deliveries: list[OutboxDelivery] = []
        with self._session_factory.begin() as db:
            due_event = or_(
                and_(OutboxEvent.status == "PENDING", OutboxEvent.available_at <= now),
                and_(
                    OutboxEvent.status == "PROCESSING",
                    OutboxEvent.locked_until.is_not(None),
                    OutboxEvent.locked_until <= now,
                ),
            )
            events = db.scalars(
                select(OutboxEvent)
                .where(due_event)
                .order_by(OutboxEvent.created_at, OutboxEvent.id)
                .limit(DISPATCH_BATCH_SIZE)
                .with_for_update(skip_locked=True)
            ).all()
            for event in events:
                if event.attempts >= MAX_ATTEMPTS:
                    event.status = "DEAD"
                    event.locked_until = None
                    event.claim_token = None
                    event.last_error_code = "RETRY_LIMIT_EXCEEDED"
                    continue
                claim_token = uuid4()
                event.status = "PROCESSING"
                event.attempts += 1
                event.locked_until = now + timedelta(seconds=CLAIM_LEASE_SECONDS)
                event.claim_token = claim_token
                deliveries.append(
                    OutboxDelivery(
                        event_id=event.id,
                        consultation_id=event.consultation_id,
                        event_type=event.event_type,
                        resource_id=event.resource_id,
                        created_at=event.created_at,
                        claim_token=claim_token,
                        attempts=event.attempts,
                    )
                )
        return deliveries

    def _mark_published(self, delivery: OutboxDelivery, now: datetime) -> None:
        with self._session_factory.begin() as db:
            db.execute(
                update(OutboxEvent)
                .where(
                    OutboxEvent.id == delivery.event_id,
                    OutboxEvent.status == "PROCESSING",
                    OutboxEvent.claim_token == delivery.claim_token,
                )
                .values(
                    status="PUBLISHED",
                    published_at=now,
                    locked_until=None,
                    claim_token=None,
                    last_error_code=None,
                )
            )

    def _mark_failed(self, delivery: OutboxDelivery, now: datetime) -> None:
        retry_delay = min(2 ** (delivery.attempts - 1), MAX_BACKOFF_SECONDS)
        is_exhausted = delivery.attempts >= MAX_ATTEMPTS
        with self._session_factory.begin() as db:
            db.execute(
                update(OutboxEvent)
                .where(
                    OutboxEvent.id == delivery.event_id,
                    OutboxEvent.status == "PROCESSING",
                    OutboxEvent.claim_token == delivery.claim_token,
                )
                .values(
                    status="DEAD" if is_exhausted else "PENDING",
                    available_at=now + timedelta(seconds=retry_delay),
                    locked_until=None,
                    claim_token=None,
                    last_error_code=("RETRY_LIMIT_EXCEEDED" if is_exhausted else "DELIVERY_FAILED"),
                )
            )


def create_outbox_dispatcher(
    engine_url: str,
    publisher: NotificationPublisher,
) -> OutboxDispatcher:
    from app.infrastructure.persistence.database import get_engine

    session_factory = sessionmaker(
        bind=get_engine(engine_url),
        class_=Session,
        expire_on_commit=False,
    )
    return OutboxDispatcher(session_factory, publisher)


def read_outbox_health(db: Session) -> OutboxHealth:
    counts = db.execute(
        select(
            func.coalesce(
                func.sum(case((OutboxEvent.status == "PENDING", 1), else_=0)),
                0,
            ),
            func.coalesce(
                func.sum(case((OutboxEvent.status == "PROCESSING", 1), else_=0)),
                0,
            ),
            func.coalesce(
                func.sum(case((OutboxEvent.status == "DEAD", 1), else_=0)),
                0,
            ),
            func.min(
                case(
                    (OutboxEvent.status != "PUBLISHED", OutboxEvent.created_at),
                    else_=None,
                )
            ),
        )
    ).one()

    oldest_unpublished = counts[3]
    if oldest_unpublished is not None and oldest_unpublished.tzinfo is None:
        oldest_unpublished = oldest_unpublished.replace(tzinfo=UTC)
    oldest_age = (
        max(0, int((datetime.now(UTC) - oldest_unpublished).total_seconds()))
        if oldest_unpublished is not None
        else None
    )
    return OutboxHealth(
        pending=int(counts[0]),
        processing=int(counts[1]),
        dead=int(counts[2]),
        oldest_unpublished_seconds=oldest_age,
    )
