"""remove articles unique constraint news_key category

Revision ID: 7d6787188cfa
Revises: add_original_text_column
Create Date: 2025-09-12 10:05:58.806311

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7d6787188cfa'
down_revision: Union[str, None] = 'add_original_text_column'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove unique constraint on (news_key, category) to allow multiple articles with same news_key + category
    op.drop_constraint('uq_articles_news_key_category', 'articles', type_='unique')


def downgrade() -> None:
    # Re-add unique constraint on (news_key, category)
    op.create_unique_constraint('uq_articles_news_key_category', 'articles', ['news_key', 'category'])
