"""Add original_text column to text_correction_history

Revision ID: add_original_text_column
Revises: update_enum_uppercase
Create Date: 2025-09-11 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_original_text_column'
down_revision: Union[str, None] = 'update_enum_uppercase'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add original_text column to text_correction_history table
    op.add_column('text_correction_history', sa.Column('original_text', sa.Text(), nullable=True))


def downgrade() -> None:
    # Remove original_text column from text_correction_history table
    op.drop_column('text_correction_history', 'original_text')