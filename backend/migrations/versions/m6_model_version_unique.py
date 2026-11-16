"""Prevent duplicate model version identities.

Revision ID: m6_model_version_unique
Revises: m5_transactional_outbox
Create Date: 2026-11-16 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.engine import Connection

revision: str = "m6_model_version_unique"
down_revision: str | None = "m5_transactional_outbox"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MODEL_VERSION_UNIQUE_CONSTRAINT = "uq_model_versions_model_name_version"


def assert_no_duplicate_model_versions(connection: Connection) -> None:
    model_versions = sa.table(
        "model_versions",
        sa.column("id", sa.Uuid()),
        sa.column("model_name", sa.String(length=128)),
        sa.column("version", sa.String(length=64)),
    )
    duplicate_keys = (
        sa.select(model_versions.c.model_name, model_versions.c.version)
        .group_by(model_versions.c.model_name, model_versions.c.version)
        .having(sa.func.count() > 1)
        .subquery()
    )
    duplicate_rows = connection.execute(
        sa.select(
            model_versions.c.model_name,
            model_versions.c.version,
            model_versions.c.id,
        )
        .join(
            duplicate_keys,
            sa.and_(
                model_versions.c.model_name == duplicate_keys.c.model_name,
                model_versions.c.version == duplicate_keys.c.version,
            ),
        )
        .order_by(
            model_versions.c.model_name,
            model_versions.c.version,
            model_versions.c.id,
        )
    ).all()
    if not duplicate_rows:
        return

    duplicates: dict[tuple[str, str], list[str]] = {}
    for row in duplicate_rows:
        key = (row.model_name, row.version)
        duplicates.setdefault(key, []).append(str(row.id))
    duplicate_details = "; ".join(
        f"model_name={model_name!r}, version={version!r}, ids={ids!r}"
        for (model_name, version), ids in duplicates.items()
    )
    raise RuntimeError(
        "Cannot add the ModelVersion unique constraint. Review and resolve these "
        f"duplicate records explicitly; no rows were changed: {duplicate_details}"
    )


def upgrade() -> None:
    if context.is_offline_mode():
        op.execute(
            """
            DO $migration$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM model_versions
                    GROUP BY model_name, version
                    HAVING COUNT(*) > 1
                ) THEN
                    RAISE EXCEPTION
                        'Duplicate model_versions exist; inspect and explicitly resolve '
                        'model_name/version groups before retrying this migration.';
                END IF;
            END;
            $migration$;
            """
        )
    else:
        assert_no_duplicate_model_versions(op.get_bind())

    op.create_unique_constraint(
        MODEL_VERSION_UNIQUE_CONSTRAINT,
        "model_versions",
        ["model_name", "version"],
    )


def downgrade() -> None:
    op.drop_constraint(
        MODEL_VERSION_UNIQUE_CONSTRAINT,
        "model_versions",
        type_="unique",
    )
