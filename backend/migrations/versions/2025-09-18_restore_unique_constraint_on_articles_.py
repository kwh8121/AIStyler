"""Restore unique constraint on articles news_key and category

Revision ID: 484216bc0731
Revises: 5b041c83b356
Create Date: 2025-09-18 11:21:19.767974

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '484216bc0731'
down_revision: Union[str, None] = '5b041c83b356'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Restore unique constraint on (news_key, category) to prevent duplicate articles.
    
    This ensures data integrity for CMS operations where the same news_key + category
    combination should only exist once, supporting proper upsert behavior.
    """
    print("🔧 Restoring unique constraint on articles (news_key, category)...")
    
    # First, remove any potential duplicates that might exist
    # Keep the most recent version of each (news_key, category) combination
    op.execute("""
        DELETE FROM articles a1 
        USING articles a2 
        WHERE a1.news_key = a2.news_key 
          AND a1.category = a2.category 
          AND a1.id < a2.id
    """)
    
    # Add the unique constraint back
    op.create_unique_constraint(
        'uq_articles_news_key_category', 
        'articles', 
        ['news_key', 'category']
    )
    
    print("✅ Unique constraint restored successfully!")


def downgrade() -> None:
    """
    Remove unique constraint to allow multiple articles with same news_key + category.
    """
    print("⚠️ Removing unique constraint on articles (news_key, category)...")
    op.drop_constraint('uq_articles_news_key_category', 'articles', type_='unique')
    print("✅ Unique constraint removed!")
