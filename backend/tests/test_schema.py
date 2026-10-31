from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import IntegrityError

from app.infrastructure.persistence.entities import Base, Message


@pytest.fixture
def schema_engine() -> Iterator[Engine]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def test_ai_translation_message_requires_confirmation_record(schema_engine: Engine) -> None:
    with schema_engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                Message.__table__.insert().values(
                    id=uuid4(),
                    conversation_id=uuid4(),
                    sender_type="PATIENT",
                    message_type="AI_TRANSLATION",
                    text="Synthetic test",
                    provenance="AI_CANDIDATE_CONFIRMED",
                    verification_status="PATIENT_CONFIRMED",
                    created_at=datetime.now(UTC),
                )
            )
