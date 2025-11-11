"""Fix history_style unique constraint for sentence-level tracking

Revision ID: a7f8c2d3e4b5
Revises: 9150dac6f958
Create Date: 2025-09-19 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7f8c2d3e4b5'
down_revision: Union[str, None] = '9150dac6f958'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old unique constraint that prevents multiple sentences
    op.drop_constraint('uq_history_style_once', 'text_correction_history_styles', type_='unique')
    
    # Add a new unique constraint that includes sentence_index
    # This allows multiple sentences per (history_id, style_id) but prevents duplicates
    # for the same sentence
    op.create_unique_constraint(
        'uq_history_style_sentence', 
        'text_correction_history_styles', 
        ['history_id', 'style_id', 'sentence_index']
    )
    
    # Add an index on sentence_index for better query performance
    op.create_index(
        'ix_hist_style_sentence', 
        'text_correction_history_styles', 
        ['sentence_index']
    )


def downgrade() -> None:
    # Remove the new index
    op.drop_index('ix_hist_style_sentence', table_name='text_correction_history_styles')
    
    # Drop the new unique constraint
    op.drop_constraint('uq_history_style_sentence', 'text_correction_history_styles', type_='unique')
    
    # Restore the original unique constraint
    op.create_unique_constraint(
        'uq_history_style_once', 
        'text_correction_history_styles', 
        ['history_id', 'style_id']
    )