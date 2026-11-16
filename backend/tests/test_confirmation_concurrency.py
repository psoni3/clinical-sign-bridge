from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api.routes.confirmations import _claim_confirmable_interpretation
from app.infrastructure.persistence.entities import (
    Base,
    Consultation,
    ConsultationSession,
    Interpretation,
    Recognition,
)


def test_concurrent_confirmation_claims_allow_only_one_transition(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'confirmation-race.db').as_posix()}"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(engine)

    now = datetime.now(UTC)
    consultation_id = uuid4()
    session_id = uuid4()
    recognition_id = uuid4()
    interpretation_id = uuid4()
    with Session(engine) as db:
        db.add(
            Consultation(
                id=consultation_id,
                status="ACTIVE",
                join_code_hash="test-hash",
                expires_at=now + timedelta(hours=1),
                created_at=now,
            )
        )
        db.add(
            ConsultationSession(
                id=session_id,
                consultation_id=consultation_id,
                session_type="PATIENT_DEVICE",
                session_token_hash="test-hash",
                expires_at=now + timedelta(hours=1),
                created_at=now,
            )
        )
        db.add(
            Recognition(
                id=recognition_id,
                consultation_id=consultation_id,
                session_id=session_id,
                client_request_id="concurrent-confirmation",
                status="COMPLETED",
                media_metadata={},
                created_at=now,
            )
        )
        db.add(
            Interpretation(
                id=interpretation_id,
                recognition_id=recognition_id,
                status="UNVERIFIED",
                quality_status="ACCEPTABLE",
                created_at=now,
            )
        )
        db.commit()

    barrier = Barrier(2)

    def claim_confirmation() -> bool:
        with Session(engine) as db:
            barrier.wait(timeout=10)
            claimed = _claim_confirmable_interpretation(db, interpretation_id, now)
            if claimed:
                db.commit()
            else:
                db.rollback()
            return claimed

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _index: claim_confirmation(), range(2)))

        assert sum(outcomes) == 1
        with Session(engine) as db:
            status = db.scalar(
                select(Interpretation.status).where(Interpretation.id == interpretation_id)
            )
        assert status == "CONFIRMED"
    finally:
        engine.dispose()
