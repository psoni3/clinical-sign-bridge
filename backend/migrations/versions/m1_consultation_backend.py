"""M1 consultation backend schema

Revision ID: m1_consultation_backend
Revises: 5363a6f1afc4
Create Date: 2026-10-31 13:45:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "m1_consultation_backend"
down_revision: str | None = "5363a6f1afc4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Track acknowledgements and confirmation idempotency independently."""
    op.add_column(
        "messages",
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "confirmations",
        sa.Column("idempotency_key", sa.Uuid(), nullable=True),
    )
    op.create_unique_constraint(
        "uq_confirmations_interpretation_idempotency",
        "confirmations",
        ["interpretation_id", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_confirmations_interpretation_idempotency",
        "confirmations",
        type_="unique",
    )
    op.drop_column("confirmations", "idempotency_key")
    op.drop_column("messages", "acknowledged_at")
