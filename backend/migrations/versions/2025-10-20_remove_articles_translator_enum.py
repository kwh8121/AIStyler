"""Remove ARTICLES_TRANSLATOR category

Revision ID: remove_articles_translator_enum
Revises: add_restoration_type
Create Date: 2025-10-20 13:10:00.000000

"""
from typing import Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "remove_articles_translator_enum"
down_revision: Union[str, None] = "add_restoration_type"
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Normalize existing rows to BODY before changing enum definition
    op.execute("UPDATE articles SET category='BODY' WHERE category::text='ARTICLES_TRANSLATOR'")
    op.execute("UPDATE text_correction_history SET category='BODY' WHERE category::text='ARTICLES_TRANSLATOR'")
    # article_prompts_category enum never included ARTICLES_TRANSLATOR in production, but guard just in case
    op.execute("UPDATE article_prompts SET category='BODY' WHERE category::text='ARTICLES_TRANSLATOR'")

    # Update article_category enum (used by articles & text_correction_history)
    op.execute("ALTER TYPE article_category RENAME TO article_category_old")
    op.execute("CREATE TYPE article_category AS ENUM('SEO','TITLE','BODY','CAPTION')")
    op.execute(
        "ALTER TABLE articles ALTER COLUMN category TYPE article_category USING category::text::article_category"
    )
    op.execute(
        "ALTER TABLE text_correction_history ALTER COLUMN category TYPE article_category USING category::text::article_category"
    )
    op.execute("DROP TYPE article_category_old")

    # Update article_prompts_category enum
    op.execute("ALTER TYPE article_prompts_category RENAME TO article_prompts_category_old")
    op.execute("CREATE TYPE article_prompts_category AS ENUM('SEO','TITLE','BODY','CAPTION')")
    op.execute(
        "ALTER TABLE article_prompts ALTER COLUMN category TYPE article_prompts_category USING category::text::article_prompts_category"
    )
    op.execute("DROP TYPE article_prompts_category_old")


def downgrade() -> None:
    """Downgrade creating enum without ARTICLES_TRANSLATOR is not supported."""
    raise RuntimeError(
        "Downgrade not supported for remove_articles_translator_enum migration."
    )
