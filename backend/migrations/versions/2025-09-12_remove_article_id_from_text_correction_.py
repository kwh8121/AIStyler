"""remove article_id from text_correction_history

Revision ID: cda207fa69c1
Revises: 33e02e84f6c1
Create Date: 2025-09-12 11:08:16.570139

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cda207fa69c1'
down_revision: Union[str, None] = '33e02e84f6c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop foreign key constraint first
    op.drop_constraint('text_correction_history_article_id_fkey', 'text_correction_history', type_='foreignkey')
    
    # Drop the unique index that includes article_id
    op.drop_index('ix_text_corr_article_version', table_name='text_correction_history')
    
    # Drop the article_id column
    op.drop_column('text_correction_history', 'article_id')
    
    # Create new index for version tracking per news_key + category
    op.create_index('ix_tch_news_key_category_version', 
                    'text_correction_history', 
                    ['news_key', 'category', 'version'])


def downgrade() -> None:
    # Re-add article_id column
    op.add_column('text_correction_history',
        sa.Column('article_id', sa.Integer(), nullable=True))
    
    # Update article_id from articles table (if needed for rollback)
    # This would need custom logic based on your needs
    
    # Re-create the unique index
    op.create_index('ix_text_corr_article_version', 
                    'text_correction_history', 
                    ['article_id', 'version'], 
                    unique=True)
    
    # Re-add foreign key constraint
    op.create_foreign_key('text_correction_history_article_id_fkey',
                          'text_correction_history', 'articles',
                          ['article_id'], ['id'],
                          ondelete='CASCADE')
    
    # Drop the new index
    op.drop_index('ix_tch_news_key_category_version', table_name='text_correction_history')
