from uuid import uuid4

import pytest
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    Uuid,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.infrastructure.persistence.entities import Base, ModelVersion
from migrations.versions.m6_model_version_unique import assert_no_duplicate_model_versions


def test_model_version_pair_has_a_database_unique_constraint() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            ModelVersion(
                model_name="bilstm",
                version="bilstm-v1",
                metrics={},
                approval_status="RESEARCH_ONLY",
            )
        )
        session.commit()
        session.add(
            ModelVersion(
                model_name="bilstm",
                version="bilstm-v1",
                metrics={},
                approval_status="RESEARCH_ONLY",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    engine.dispose()


def test_model_version_migration_refuses_duplicates_without_modifying_rows() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()
    model_versions = Table(
        "model_versions",
        metadata,
        Column("id", Uuid(), primary_key=True),
        Column("model_name", String(128), nullable=False),
        Column("version", String(64), nullable=False),
    )
    metadata.create_all(engine)
    duplicate_ids = [uuid4(), uuid4()]

    with engine.begin() as connection:
        assert_no_duplicate_model_versions(connection)
        connection.execute(
            insert(model_versions),
            [
                {"id": duplicate_id, "model_name": "bilstm", "version": "bilstm-v1"}
                for duplicate_id in duplicate_ids
            ],
        )

        with pytest.raises(RuntimeError, match="resolve these duplicate records explicitly"):
            assert_no_duplicate_model_versions(connection)

        row_count = connection.scalar(
            select(func.count()).select_from(model_versions)
        )

    assert row_count == 2
    engine.dispose()
