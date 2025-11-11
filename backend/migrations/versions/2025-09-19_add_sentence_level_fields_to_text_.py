"""Add sentence-level fields to TextCorrectionHistoryStyle

Revision ID: 9150dac6f958
Revises: 484216bc0731
Create Date: 2025-09-19 12:18:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9150dac6f958'
down_revision: Union[str, None] = '484216bc0731'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add sentence-level fields to text_correction_history_styles table for tracking individual sentence corrections."""
    print("🔧 Adding sentence-level fields to text_correction_history_styles table...")
    
    # Add sentence-level fields to text_correction_history_styles table
    op.add_column('text_correction_history_styles', 
        sa.Column('sentence_index', sa.Integer(), nullable=True, comment='문장 순서 (0부터 시작)'))
    op.add_column('text_correction_history_styles', 
        sa.Column('before_text', sa.Text(), nullable=True, comment='원본 문장'))
    op.add_column('text_correction_history_styles', 
        sa.Column('after_text', sa.Text(), nullable=True, comment='교정된 문장'))
    op.add_column('text_correction_history_styles', 
        sa.Column('violations', postgresql.JSON(astext_type=sa.Text()), nullable=True, comment='해당 문장의 violations 리스트'))
    
    print("✅ Sentence-level fields added successfully!")


def downgrade() -> None:
    """Remove sentence-level fields from text_correction_history_styles table."""
    print("⚠️ Removing sentence-level fields from text_correction_history_styles table...")
    
    # Remove sentence-level fields from text_correction_history_styles table
    op.drop_column('text_correction_history_styles', 'violations')
    op.drop_column('text_correction_history_styles', 'after_text')
    op.drop_column('text_correction_history_styles', 'before_text')
    op.drop_column('text_correction_history_styles', 'sentence_index')
    
    print("✅ Sentence-level fields removed successfully!")