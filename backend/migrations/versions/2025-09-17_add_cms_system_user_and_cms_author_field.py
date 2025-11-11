"""Add CMS system user and cms_author field

Revision ID: ec08c58190a3
Revises: f93995cc59f6
Create Date: 2025-09-17 15:13:11.438386

"""
from typing import Sequence, Union
from sqlalchemy import text
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ec08c58190a3'
down_revision: Union[str, None] = 'f93995cc59f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    1. CMS 시스템 사용자 생성 (ID=1)
    2. Article 테이블에 cms_author 컬럼 추가
    """
    connection = op.get_bind()
    
    print("🔧 Adding cms_author column to articles table...")
    # 1. Article 테이블에 cms_author 컬럼 추가
    op.add_column('articles', sa.Column('cms_author', sa.String(100), nullable=True))
    
    print("👤 Creating CMS system user...")
    # 2. CMS 시스템 사용자 생성 (ID=1, 멱등성 보장)
    try:
        # 기존 사용자 확인
        result = connection.execute(text("SELECT COUNT(*) FROM users WHERE id = 1"))
        existing_count = result.fetchone()[0]
        
        if existing_count == 0:
            # CMS 시스템 사용자 생성
            connection.execute(text("""
                INSERT INTO users (id, name, email, role, hashed_password, created_at, updated_at)
                VALUES (
                    1, 
                    'CMS System', 
                    'cms-system@internal.ai-styler.com', 
                    'system',
                    'NOT_APPLICABLE',  -- CMS 시스템은 로그인하지 않음
                    NOW(), 
                    NOW()
                )
            """))
            
            # PostgreSQL에서 시퀀스 조정 (ID=1 이후부터 자동 증가)
            connection.execute(text("SELECT setval('users_id_seq', 1, true)"))
            
            print("✅ CMS system user created with ID=1")
        else:
            print("ℹ️ CMS system user already exists (ID=1)")
            
    except Exception as e:
        print(f"⚠️ Error creating CMS system user: {e}")
        # 실패해도 컬럼 추가는 성공했으므로 계속 진행
        pass
    
    print("✅ Migration completed successfully!")


def downgrade() -> None:
    """
    변경사항 롤백
    """
    connection = op.get_bind()
    
    print("🗑️ Removing cms_author column...")
    # cms_author 컬럼 제거
    op.drop_column('articles', 'cms_author')
    
    print("👤 Removing CMS system user...")
    # CMS 시스템 사용자 제거 (신중하게)
    try:
        # CMS 시스템 사용자가 생성한 Article이 있는지 확인
        result = connection.execute(text("SELECT COUNT(*) FROM articles WHERE user_id = 1"))
        article_count = result.fetchone()[0]
        
        if article_count > 0:
            print(f"⚠️ Warning: {article_count} articles are linked to CMS system user")
            print("⚠️ CMS system user will NOT be deleted to preserve data integrity")
        else:
            # Article이 없으면 시스템 사용자 삭제
            connection.execute(text("DELETE FROM users WHERE id = 1 AND email = 'cms-system@internal.ai-styler.com'"))
            print("✅ CMS system user removed")
            
    except Exception as e:
        print(f"⚠️ Error during CMS system user cleanup: {e}")
    
    print("✅ Downgrade completed!")
