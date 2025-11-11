"""add news_key category to text_correction_history

Revision ID: 33e02e84f6c1
Revises: 7d6787188cfa
Create Date: 2025-09-12 10:45:49.635372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '33e02e84f6c1'
down_revision: Union[str, None] = '7d6787188cfa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add news_key and category columns to text_correction_history
    op.add_column('text_correction_history', 
        sa.Column('news_key', sa.String(36), nullable=True))
    op.add_column('text_correction_history',
        sa.Column('category', sa.Enum('TITLE', 'BODY', 'CAPTION', name='article_category'), nullable=True))
    
    # Update existing data from articles table
    op.execute("""
        UPDATE text_correction_history tch
        SET news_key = a.news_key,
            category = a.category
        FROM articles a
        WHERE tch.article_id = a.id
    """)
    
    # Make columns NOT NULL after data migration
    op.alter_column('text_correction_history', 'news_key', nullable=False)
    op.alter_column('text_correction_history', 'category', nullable=False)
    
    # Create indexes for better query performance
    op.create_index('ix_tch_news_key_category_created', 
                    'text_correction_history', 
                    ['news_key', 'category', 'created_at'])
    op.create_index('ix_tch_news_key_created', 
                    'text_correction_history', 
                    ['news_key', 'created_at'])
    op.create_index('ix_tch_news_key', 
                    'text_correction_history', 
                    ['news_key'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_tch_news_key', table_name='text_correction_history')
    op.drop_index('ix_tch_news_key_created', table_name='text_correction_history')
    op.drop_index('ix_tch_news_key_category_created', table_name='text_correction_history')
    
    # Drop columns
    op.drop_column('text_correction_history', 'category')
    op.drop_column('text_correction_history', 'news_key')
