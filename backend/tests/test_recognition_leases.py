import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Barrier
from typing import BinaryIO
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select, update
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import Session

from app.application.recognition_service import (
    RecognitionOrchestrationService,
    RecognitionOutcome,
    RecognitionSubmission,
    _RecognitionClaim,
)
from app.infrastructure.ai_clients.video_inference_client import (
    AIInferenceCandidate,
    AIInferenceError,
    AIInferenceResult,
    AIInferenceUnavailableError,
)
from app.infrastructure.persistence.entities import (
    Base,
    Consultation,
    ConsultationSession,
    Interpretation,
    InterpretationCandidate,
    Recognition,
)
from migrations.versions import m7_recognition_lease_fencing

LEASE_SECONDS = 30
VIDEO = b"bounded synthetic video"
VIDEO_DIGEST = sha256(VIDEO).hexdigest()


class FrozenClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class ControlledInferenceClient:
    def __init__(
        self,
        blocked_calls: set[int] | None = None,
        errors: dict[int, AIInferenceError] | None = None,
    ) -> None:
        self.calls = 0
        self.blocked_calls = blocked_calls or set()
        self.errors = errors or {}
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]

    async def predict_video(
        self,
        video: BinaryIO,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> AIInferenceResult:
        call_number = self.calls + 1
        self.calls = call_number
        video.read()
        self.started[call_number - 1].set()
        if call_number in self.blocked_calls:
            await self.release[call_number - 1].wait()
        error = self.errors.get(call_number)
        if error is not None:
            raise error
        return _successful_result()


def _successful_result() -> AIInferenceResult:
    return AIInferenceResult(
        prediction_status="UNCONFIRMED",
        model_version="bilstm-v1",
        score_type="uncalibrated_softmax",
        frame_count=12,
        candidates=tuple(
            AIInferenceCandidate(
                rank=rank,
                label=f"INCLUDE-50-class-{rank}",
                score=0.9 - rank / 10,
            )
            for rank in range(1, 6)
        ),
        warning="Predictions are unconfirmed.",
    )


@pytest.fixture
def recognition_database(
    tmp_path: Path,
) -> Iterator[tuple[Engine, UUID, UUID, datetime]]:
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(tmp_path / "recognition-leases.db")),
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(engine)
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
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
                ConsultationSession(
                    id=session_id,
                    consultation_id=consultation_id,
                    session_type="PATIENT_DEVICE",
                    session_token_hash="test-hash",
                    expires_at=now + timedelta(hours=1),
                    created_at=now,
                ),
            ]
        )
        db.commit()
    try:
        yield engine, consultation_id, session_id, now
    finally:
        engine.dispose()


def _submission(
    consultation_id: UUID,
    session_id: UUID,
    now: datetime,
    request_id: str = "lease-test",
    content_digest: str = VIDEO_DIGEST,
) -> RecognitionSubmission:
    return RecognitionSubmission(
        consultation_id=consultation_id,
        session_id=session_id,
        client_request_id=request_id,
        content_digest=content_digest,
        size_bytes=len(VIDEO),
        container="video/mp4",
        video=BytesIO(VIDEO),
        filename="upload.mp4",
        content_type="video/mp4",
        created_at=now,
        correlation_id=str(uuid4()),
    )


def _seed_recognition(
    engine: Engine,
    consultation_id: UUID,
    session_id: UUID,
    now: datetime,
    status: str,
    lease_expires_at: datetime | None,
    processing_token: UUID | None,
) -> UUID:
    recognition_id = uuid4()
    with Session(engine) as db:
        db.add(
            Recognition(
                id=recognition_id,
                consultation_id=consultation_id,
                session_id=session_id,
                client_request_id="lease-test",
                status=status,
                media_metadata={"sha256": VIDEO_DIGEST},
                temporary_media_reference=None,
                processing_lease_expires_at=lease_expires_at,
                processing_token=processing_token,
                created_at=now,
                completed_at=now if status == "FAILED" else None,
            )
        )
        db.commit()
    return recognition_id


async def _execute(
    engine: Engine,
    client: ControlledInferenceClient,
    submission: RecognitionSubmission,
    clock: FrozenClock,
) -> RecognitionOutcome:
    with Session(engine) as db:
        service = RecognitionOrchestrationService(
            db,
            client,
            processing_lease_seconds=LEASE_SECONDS,
            clock=clock,
        )
        return await service.execute(submission)


def _read_recognition(engine: Engine, recognition_id: UUID) -> Recognition:
    with Session(engine) as db:
        recognition = db.get(Recognition, recognition_id)
        assert recognition is not None
        return recognition


def test_active_processing_lease_prevents_duplicate_inference(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    token = uuid4()
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        now + timedelta(seconds=LEASE_SECONDS),
        token,
    )
    client = ControlledInferenceClient()

    outcome = asyncio.run(
        _execute(engine, client, _submission(consultation_id, session_id, now), FrozenClock(now))
    )

    assert outcome.status == "PROCESSING"
    assert outcome.recognition_id == recognition_id
    assert client.calls == 0
    current = _read_recognition(engine, recognition_id)
    assert current.processing_token == token


def test_expired_processing_lease_is_reclaimed(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    old_token = uuid4()
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        now - timedelta(seconds=1),
        old_token,
    )
    client = ControlledInferenceClient()

    outcome = asyncio.run(
        _execute(engine, client, _submission(consultation_id, session_id, now), FrozenClock(now))
    )

    current = _read_recognition(engine, recognition_id)
    assert outcome.status == "COMPLETED"
    assert current.status == "COMPLETED"
    assert current.processing_token != old_token
    assert current.processing_lease_expires_at is None
    assert client.calls == 1


def test_legacy_processing_row_without_lease_is_reclaimed(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        None,
        None,
    )
    client = ControlledInferenceClient()

    outcome = asyncio.run(
        _execute(engine, client, _submission(consultation_id, session_id, now), FrozenClock(now))
    )

    current = _read_recognition(engine, recognition_id)
    assert outcome.status == "COMPLETED"
    assert current.status == "COMPLETED"
    assert current.processing_token is not None
    assert client.calls == 1


def test_concurrent_expired_lease_reclaims_have_one_inference_owner(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        now - timedelta(seconds=1),
        uuid4(),
    )
    client = ControlledInferenceClient(blocked_calls={1})
    submission = _submission(consultation_id, session_id, now)
    clock = FrozenClock(now)

    async def run_concurrent_reclaims() -> tuple[RecognitionOutcome, RecognitionOutcome]:
        first = asyncio.create_task(_execute(engine, client, submission, clock))
        second = asyncio.create_task(_execute(engine, client, submission, clock))
        await asyncio.wait_for(client.started[0].wait(), timeout=5)
        second_outcome = await asyncio.wait_for(second, timeout=5)
        active = _read_recognition(engine, recognition_id)
        client.release[0].set()
        first_outcome = await asyncio.wait_for(first, timeout=5)
        assert active.processing_lease_expires_at is not None
        assert active.processing_token is not None
        return first_outcome, second_outcome

    first_outcome, second_outcome = asyncio.run(run_concurrent_reclaims())

    assert client.calls == 1
    assert first_outcome.status == "COMPLETED"
    assert second_outcome.status == "PROCESSING"


def test_simultaneous_expired_lease_updates_grant_one_claim(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        now - timedelta(seconds=1),
        uuid4(),
    )
    barrier = Barrier(2)

    def claim_expired_row() -> _RecognitionClaim | RecognitionOutcome:
        with Session(engine) as db:
            recognition = db.get(Recognition, recognition_id)
            assert recognition is not None
            barrier.wait(timeout=5)
            service = RecognitionOrchestrationService(
                db,
                ControlledInferenceClient(),
                processing_lease_seconds=LEASE_SECONDS,
                clock=FrozenClock(now),
            )
            return service._claim_expired_processing(recognition, now)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: claim_expired_row(), range(2)))

    claims = [outcome for outcome in outcomes if isinstance(outcome, _RecognitionClaim)]
    in_progress = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, RecognitionOutcome) and outcome.status == "PROCESSING"
    ]
    assert len(claims) == 1
    assert len(in_progress) == 1
    current = _read_recognition(engine, recognition_id)
    assert current.processing_token == claims[0].processing_token


def test_failed_recognition_retry_uses_a_new_fencing_token(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    old_token = uuid4()
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "FAILED",
        None,
        old_token,
    )
    client = ControlledInferenceClient(blocked_calls={1})
    submission = _submission(consultation_id, session_id, now)
    clock = FrozenClock(now)

    async def run_retry() -> RecognitionOutcome:
        retry = asyncio.create_task(_execute(engine, client, submission, clock))
        await asyncio.wait_for(client.started[0].wait(), timeout=5)
        current = _read_recognition(engine, recognition_id)
        assert current.status == "PROCESSING"
        assert current.processing_token != old_token
        client.release[0].set()
        return await asyncio.wait_for(retry, timeout=5)

    outcome = asyncio.run(run_retry())

    assert outcome.status == "COMPLETED"
    assert client.calls == 1


def test_stale_success_cannot_overwrite_reclaimed_attempt(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        now - timedelta(seconds=1),
        uuid4(),
    )
    client = ControlledInferenceClient(blocked_calls={1, 2})
    submission = _submission(consultation_id, session_id, now)
    clock = FrozenClock(now)

    async def run_stale_success() -> tuple[RecognitionOutcome, RecognitionOutcome, UUID]:
        stale = asyncio.create_task(_execute(engine, client, submission, clock))
        await asyncio.wait_for(client.started[0].wait(), timeout=5)
        first_token = _read_recognition(engine, recognition_id).processing_token
        with Session(engine) as db:
            db.execute(
                update(Recognition)
                .where(Recognition.id == recognition_id)
                .values(processing_lease_expires_at=now - timedelta(seconds=1))
            )
            db.commit()

        current = asyncio.create_task(_execute(engine, client, submission, clock))
        await asyncio.wait_for(client.started[1].wait(), timeout=5)
        second_token = _read_recognition(engine, recognition_id).processing_token
        assert second_token is not None and second_token != first_token
        client.release[1].set()
        current_outcome = await asyncio.wait_for(current, timeout=5)
        client.release[0].set()
        stale_outcome = await asyncio.wait_for(stale, timeout=5)
        return stale_outcome, current_outcome, second_token

    stale_outcome, current_outcome, winning_token = asyncio.run(run_stale_success())

    assert client.calls == 2
    assert stale_outcome.status == current_outcome.status == "COMPLETED"
    assert stale_outcome.interpretation_id == current_outcome.interpretation_id
    current = _read_recognition(engine, recognition_id)
    assert current.processing_token == winning_token
    with Session(engine) as db:
        interpretations = db.scalars(
            select(Interpretation).where(Interpretation.recognition_id == recognition_id)
        ).all()
        candidates = db.scalars(
            select(InterpretationCandidate)
            .join(Interpretation)
            .where(Interpretation.recognition_id == recognition_id)
        ).all()
    assert len(interpretations) == 1
    assert len(candidates) == 5


def test_stale_failure_cannot_mark_reclaimed_attempt_failed(
    recognition_database: tuple[Engine, UUID, UUID, datetime],
) -> None:
    engine, consultation_id, session_id, now = recognition_database
    recognition_id = _seed_recognition(
        engine,
        consultation_id,
        session_id,
        now,
        "PROCESSING",
        now - timedelta(seconds=1),
        uuid4(),
    )
    client = ControlledInferenceClient(
        blocked_calls={1, 2},
        errors={1: AIInferenceUnavailableError("simulated stale inference failure")},
    )
    submission = _submission(consultation_id, session_id, now)
    clock = FrozenClock(now)

    async def run_stale_failure() -> tuple[RecognitionOutcome, RecognitionOutcome, UUID]:
        stale = asyncio.create_task(_execute(engine, client, submission, clock))
        await asyncio.wait_for(client.started[0].wait(), timeout=5)
        first_token = _read_recognition(engine, recognition_id).processing_token
        with Session(engine) as db:
            db.execute(
                update(Recognition)
                .where(Recognition.id == recognition_id)
                .values(processing_lease_expires_at=now - timedelta(seconds=1))
            )
            db.commit()

        current = asyncio.create_task(_execute(engine, client, submission, clock))
        await asyncio.wait_for(client.started[1].wait(), timeout=5)
        winning_token = _read_recognition(engine, recognition_id).processing_token
        assert winning_token is not None and winning_token != first_token
        client.release[1].set()
        current_outcome = await asyncio.wait_for(current, timeout=5)
        client.release[0].set()
        stale_outcome = await asyncio.wait_for(stale, timeout=5)
        return stale_outcome, current_outcome, winning_token

    stale_outcome, current_outcome, winning_token = asyncio.run(run_stale_failure())

    assert client.calls == 2
    assert stale_outcome.status == current_outcome.status == "COMPLETED"
    current = _read_recognition(engine, recognition_id)
    assert current.status == "COMPLETED"
    assert current.processing_token == winning_token
    with Session(engine) as db:
        candidates = db.scalars(
            select(InterpretationCandidate)
            .join(Interpretation)
            .where(Interpretation.recognition_id == recognition_id)
        ).all()
    assert len(candidates) == 5


def test_lease_migration_preserves_legacy_processing_rows_as_reclaimable(
    tmp_path: Path,
) -> None:
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(tmp_path / "legacy-recognition.db"))
    )
    metadata = sa.MetaData()
    legacy_recognitions = sa.Table(
        "recognitions",
        metadata,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
    )
    metadata.create_all(engine)
    recognition_id = uuid4()

    with engine.begin() as connection:
        connection.execute(
            legacy_recognitions.insert(),
            {"id": recognition_id, "status": "PROCESSING"},
        )
        with Operations.context(MigrationContext.configure(connection)):
            m7_recognition_lease_fencing.upgrade()

        migrated_recognitions = sa.Table(
            "recognitions",
            sa.MetaData(),
            autoload_with=connection,
        )
        row = connection.execute(
            select(migrated_recognitions).where(
                migrated_recognitions.c.id == recognition_id
            )
        ).one()
        columns = {
            column["name"]: column
            for column in sa.inspect(connection).get_columns("recognitions")
        }

    assert row.status == "PROCESSING"
    assert row.processing_lease_expires_at is None
    assert row.processing_token is None
    assert columns["processing_lease_expires_at"]["nullable"] is True
    assert columns["processing_token"]["nullable"] is True
    engine.dispose()
