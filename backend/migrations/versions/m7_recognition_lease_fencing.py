"""Add leases and fencing tokens to recognition processing.

Revision ID: m7_recognition_lease_fencing
Revises: m6_model_version_unique
Create Date: 2026-11-16 00:00:00

Existing PROCESSING rows keep null lease fields and are reclaimed by the application.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "m7_recognition_lease_fencing"
down_revision: str | None = "m6_model_version_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "recognitions",
        sa.Column("processing_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "recognitions",
        sa.Column("processing_token", sa.Uuid(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("recognitions", "processing_token")
    op.drop_column("recognitions", "processing_lease_expires_at")
