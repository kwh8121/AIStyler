"""Update enum values to uppercase for consistency

Revision ID: update_enum_uppercase
Revises: 35d66ba6638c
Create Date: 2025-09-11

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'update_enum_uppercase'
down_revision: Union[str, None] = '35d66ba6638c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Update operationtype enum values from lowercase to uppercase
    and convert article.status from VARCHAR to ENUM
    """
    
    # 1. Handle operationtype enum update
    # Create temporary column
    op.add_column('text_correction_history', 
        sa.Column('operation_type_temp', sa.String(50), nullable=True)
    )
    
    # Copy data with uppercase values
    op.execute("""
        UPDATE text_correction_history 
        SET operation_type_temp = CASE 
            WHEN operation_type::text = 'correction' THEN 'CORRECTION'
            WHEN operation_type::text = 'translation' THEN 'TRANSLATION'
            WHEN operation_type::text = 'translation_correction' THEN 'TRANSLATION_CORRECTION'
            ELSE operation_type::text
        END
    """)
    
    # Drop old column and type
    op.drop_column('text_correction_history', 'operation_type')
    op.execute('DROP TYPE IF EXISTS operationtype')
    
    # Create new enum type with uppercase values
    operationtype_enum = postgresql.ENUM(
        'CORRECTION', 'TRANSLATION', 'TRANSLATION_CORRECTION',
        name='operationtype'
    )
    operationtype_enum.create(op.get_bind())
    
    # Add new column with correct enum
    op.add_column('text_correction_history',
        sa.Column('operation_type', operationtype_enum, 
                  nullable=False, server_default='CORRECTION')
    )
    
    # Restore data
    op.execute("""
        UPDATE text_correction_history 
        SET operation_type = operation_type_temp::operationtype
    """)
    
    # Drop temporary column
    op.drop_column('text_correction_history', 'operation_type_temp')
    
    # Recreate index
    op.create_index('ix_text_correction_history_operation_type', 
                    'text_correction_history', ['operation_type'])
    
    # 2. Convert article.status from VARCHAR to ENUM
    # First, create the enum type
    articlestatus_enum = postgresql.ENUM(
        'DRAFT', 'TRANSLATING', 'TRANSLATED', 'CORRECTING', 'COMPLETED',
        name='articlestatus'
    )
    articlestatus_enum.create(op.get_bind())
    
    # Add temporary column
    op.add_column('articles',
        sa.Column('status_temp', articlestatus_enum, nullable=True)
    )
    
    # Convert existing data to uppercase
    op.execute("""
        UPDATE articles 
        SET status_temp = CASE 
            WHEN status = 'draft' THEN 'DRAFT'::articlestatus
            WHEN status = 'translating' THEN 'TRANSLATING'::articlestatus
            WHEN status = 'translated' THEN 'TRANSLATED'::articlestatus
            WHEN status = 'correcting' THEN 'CORRECTING'::articlestatus
            WHEN status = 'completed' THEN 'COMPLETED'::articlestatus
            ELSE 'DRAFT'::articlestatus
        END
    """)
    
    # Drop old column
    op.drop_column('articles', 'status')
    
    # Rename new column
    op.alter_column('articles', 'status_temp', 
                    new_column_name='status',
                    nullable=False,
                    server_default='DRAFT')
    
    # 3. Update article_category enum if needed (already uppercase, so skip)
    # article_category is already TITLE, BODY, CAPTION - no change needed


def downgrade() -> None:
    """
    Revert enum values back to lowercase
    """
    
    # 1. Revert operationtype enum
    op.add_column('text_correction_history', 
        sa.Column('operation_type_temp', sa.String(50), nullable=True)
    )
    
    op.execute("""
        UPDATE text_correction_history 
        SET operation_type_temp = CASE 
            WHEN operation_type::text = 'CORRECTION' THEN 'correction'
            WHEN operation_type::text = 'TRANSLATION' THEN 'translation'
            WHEN operation_type::text = 'TRANSLATION_CORRECTION' THEN 'translation_correction'
            ELSE operation_type::text
        END
    """)
    
    op.drop_column('text_correction_history', 'operation_type')
    op.execute('DROP TYPE IF EXISTS operationtype')
    
    # Recreate with lowercase
    operationtype_enum = postgresql.ENUM(
        'correction', 'translation', 'translation_correction',
        name='operationtype'
    )
    operationtype_enum.create(op.get_bind())
    
    op.add_column('text_correction_history',
        sa.Column('operation_type', operationtype_enum, 
                  nullable=False, server_default='correction')
    )
    
    op.execute("""
        UPDATE text_correction_history 
        SET operation_type = operation_type_temp::operationtype
    """)
    
    op.drop_column('text_correction_history', 'operation_type_temp')
    op.create_index('ix_text_correction_history_operation_type', 
                    'text_correction_history', ['operation_type'])
    
    # 2. Revert article.status to VARCHAR
    op.add_column('articles',
        sa.Column('status_temp', sa.String(32), nullable=True)
    )
    
    op.execute("""
        UPDATE articles 
        SET status_temp = CASE 
            WHEN status::text = 'DRAFT' THEN 'draft'
            WHEN status::text = 'TRANSLATING' THEN 'translating'
            WHEN status::text = 'TRANSLATED' THEN 'translated'
            WHEN status::text = 'CORRECTING' THEN 'correcting'
            WHEN status::text = 'COMPLETED' THEN 'completed'
            ELSE 'draft'
        END
    """)
    
    op.drop_column('articles', 'status')
    op.alter_column('articles', 'status_temp', 
                    new_column_name='status',
                    nullable=False)
    
    # Drop the enum type
    op.execute('DROP TYPE IF EXISTS articlestatus')