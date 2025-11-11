"""Increase news_key column size to 255 characters

Revision ID: 222e5da5b323
Revises: cb7cfee3e90a
Create Date: 2025-09-22 15:02:54.577333

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '222e5da5b323'
down_revision: Union[str, None] = 'cb7cfee3e90a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Increase news_key column size from 36 to 255 characters in all related tables"""
    # Articles table
    op.alter_column('articles', 'news_key',
                    type_=sa.String(255),
                    existing_type=sa.String(36),
                    existing_nullable=False)
    
    # TextCorrectionHistory table
    op.alter_column('text_correction_history', 'news_key',
                    type_=sa.String(255),
                    existing_type=sa.String(36),
                    existing_nullable=False)
    
    # SEOGenerationHistory table
    op.alter_column('seo_generation_history', 'news_key',
                    type_=sa.String(255),
                    existing_type=sa.String(36),
                    existing_nullable=True)


def downgrade() -> None:
    """Revert news_key column size back to 36 characters"""
    # NOTE: This may fail if there are values longer than 36 characters
    # SEOGenerationHistory table
    op.alter_column('seo_generation_history', 'news_key',
                    type_=sa.String(36),
                    existing_type=sa.String(255),
                    existing_nullable=True)
    
    # TextCorrectionHistory table
    op.alter_column('text_correction_history', 'news_key',
                    type_=sa.String(36),
                    existing_type=sa.String(255),
                    existing_nullable=False)
    
    # Articles table
    op.alter_column('articles', 'news_key',
                    type_=sa.String(36),
                    existing_type=sa.String(255),
                    existing_nullable=False)
