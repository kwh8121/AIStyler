"""Add RESTORATION operation type to operationtype enum

Revision ID: add_restoration_type
Revises: 222e5da5b323
Create Date: 2025-10-17 14:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_restoration_type'
down_revision: Union[str, None] = '222e5da5b323'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add RESTORATION value to operationtype enum
    PostgreSQL doesn't allow removing enum values easily, so we only implement upgrade
    """
    # Add new enum value to existing type
    # Using ALTER TYPE ... ADD VALUE (PostgreSQL 9.1+)
    op.execute("ALTER TYPE operationtype ADD VALUE IF NOT EXISTS 'RESTORATION'")


def downgrade() -> None:
    """
    PostgreSQL doesn't support removing enum values easily.
    To properly downgrade, you would need to:
    1. Create new enum without RESTORATION
    2. Migrate data
    3. Drop old enum and rename new one

    For safety, this downgrade is a no-op.
    If you need to remove RESTORATION, do it manually or create a new migration.
    """
    pass
