"""Normalize SEO_TITLE categories to SEO

Revision ID: normalize_seo_category
Revises: remove_articles_translator_enum
Create Date: 2025-10-21 07:10:00.000000

"""
from typing import Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "normalize_seo_category"
down_revision: Union[str, None] = "remove_articles_translator_enum"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE articles SET category='SEO' WHERE category::text='SEO_TITLE'")
    op.execute("UPDATE text_correction_history SET category='SEO' WHERE category::text='SEO_TITLE'")
    op.execute("UPDATE article_prompts SET category='SEO' WHERE category::text='SEO_TITLE'")


def downgrade() -> None:
    # 안전한 다운그레이드를 지원하지 않습니다.
    raise RuntimeError("Downgrade not supported for normalize_seo_category migration.")
