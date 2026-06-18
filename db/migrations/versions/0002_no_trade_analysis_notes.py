from __future__ import annotations

from alembic import op

revision = "0002_no_trade_analysis_notes"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE no_trade_log
            ADD COLUMN IF NOT EXISTS gate_reasons JSONB,
            ADD COLUMN IF NOT EXISTS analysis_notes TEXT;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE no_trade_log
            DROP COLUMN IF EXISTS gate_reasons,
            DROP COLUMN IF EXISTS analysis_notes;
        """
    )
