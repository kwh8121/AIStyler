"""Add SEO_TITLE to ArticleCategory enum

Revision ID: f93995cc59f6
Revises: 27ff3b8a13ec
Create Date: 2025-09-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f93995cc59f6'
down_revision: Union[str, None] = '27ff3b8a13ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add SEO_TITLE value to article_category enum.
    
    This is a safe operation that adds a new enum value without affecting existing data.
    """
    # Add SEO_TITLE to the article_category enum
    # This is safe in PostgreSQL - adding enum values doesn't break existing data
    op.execute("ALTER TYPE article_category ADD VALUE 'SEO_TITLE'")


def downgrade() -> None:
    """
    Downgrade is complex for enum values in PostgreSQL.
    
    In production, removing enum values requires recreating the enum type,
    which is risky if data exists. For safety, we'll leave a comment
    explaining the manual process if needed.
    """
    # WARNING: Removing enum values in PostgreSQL is complex and risky.
    # If you need to remove SEO_TITLE, you would need to:
    # 1. Ensure no data uses SEO_TITLE value
    # 2. Create a new enum type without SEO_TITLE
    # 3. Update all columns to use the new type
    # 4. Drop the old enum type
    # 
    # For safety, this downgrade does nothing.
    # Manual intervention required if rollback is necessary.
    pass