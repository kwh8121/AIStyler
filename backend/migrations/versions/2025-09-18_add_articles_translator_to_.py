"""Add ARTICLES_TRANSLATOR to ArticleCategory enum

Revision ID: 5b041c83b356
Revises: 20b3f1eee28a
Create Date: 2025-09-18 10:50:32.211059

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b041c83b356'
down_revision: Union[str, None] = '20b3f1eee28a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add ARTICLES_TRANSLATOR value to article_category enum.
    
    This is a safe operation that adds a new enum value without affecting existing data.
    Following the same pattern as SEO_TITLE addition.
    """
    print("🔧 Adding ARTICLES_TRANSLATOR to article_category enum...")
    
    # Add ARTICLES_TRANSLATOR to the article_category enum
    # This is safe in PostgreSQL - adding enum values doesn't break existing data
    op.execute("ALTER TYPE article_category ADD VALUE 'ARTICLES_TRANSLATOR'")
    
    print("✅ ARTICLES_TRANSLATOR added to article_category enum successfully!")


def downgrade() -> None:
    """
    Downgrade is complex for enum values in PostgreSQL.
    
    In production, removing enum values requires recreating the enum type,
    which is risky if data exists. For safety, we'll leave a comment
    explaining the manual process if needed.
    """
    # WARNING: Removing enum values in PostgreSQL is complex and risky.
    # If you need to remove ARTICLES_TRANSLATOR, you would need to:
    # 1. Ensure no data uses ARTICLES_TRANSLATOR value
    # 2. Create a new enum type without ARTICLES_TRANSLATOR
    # 3. Update all columns to use the new type
    # 4. Drop the old enum type
    # 
    # For safety, this downgrade does nothing.
    # Manual intervention required if rollback is necessary.
    print("⚠️ WARNING: Cannot safely remove ARTICLES_TRANSLATOR enum value")
    print("ℹ️ Manual intervention required for enum value removal")
    pass
