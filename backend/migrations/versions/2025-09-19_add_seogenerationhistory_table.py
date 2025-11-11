"""Add SEOGenerationHistory table

Revision ID: cb7cfee3e90a
Revises: a7f8c2d3e4b5
Create Date: 2025-09-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "cb7cfee3e90a"
down_revision: Union[str, None] = "a7f8c2d3e4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create SEOGenerationHistory table
    op.create_table(
        "seo_generation_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("news_key", sa.String(length=36), nullable=True),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("edited_title", sa.Text(), nullable=False),
        sa.Column("seo_titles", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("prompt_used", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=50), nullable=True),
        sa.Column("data_type", sa.Text(), nullable=True),
        sa.Column("guideline_text", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ),
        sa.PrimaryKeyConstraint("id")
    )
    op.create_index(op.f("ix_seo_generation_history_id"), "seo_generation_history", ["id"], unique=False)
    op.create_index("ix_seo_history_created_at", "seo_generation_history", ["created_at"], unique=False)
    op.create_index("ix_seo_history_news_key", "seo_generation_history", ["news_key"], unique=False)


def downgrade() -> None:
    # Drop SEOGenerationHistory table
    op.drop_index("ix_seo_history_news_key", table_name="seo_generation_history")
    op.drop_index("ix_seo_history_created_at", table_name="seo_generation_history")
    op.drop_index(op.f("ix_seo_generation_history_id"), table_name="seo_generation_history")
    op.drop_table("seo_generation_history")
