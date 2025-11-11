"""Add public_id UUID field to articles

Revision ID: 20b3f1eee28a
Revises: ec08c58190a3
Create Date: 2025-09-17 15:25:52.246932

"""
from typing import Sequence, Union
import uuid
from sqlalchemy import text
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20b3f1eee28a'
down_revision: Union[str, None] = 'ec08c58190a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Article 테이블에 public_id UUID 컬럼 추가 및 기존 데이터 마이그레이션
    """
    connection = op.get_bind()
    
    print("🆔 Adding public_id UUID column to articles table...")
    
    # 1. public_id 컬럼 추가 (nullable=True로 먼저 생성)
    op.add_column('articles', sa.Column('public_id', sa.String(36), nullable=True))
    
    # 2. 기존 데이터에 UUID 할당
    print("🔄 Generating UUIDs for existing articles...")
    try:
        # 기존 Article 개수 확인
        result = connection.execute(text("SELECT COUNT(*) FROM articles WHERE public_id IS NULL"))
        count = result.fetchone()[0]
        
        if count > 0:
            print(f"📦 Found {count} articles without public_id")
            
            # PostgreSQL의 gen_random_uuid() 함수 사용 (더 효율적)
            # 만약 gen_random_uuid()가 없다면 uuid-ossp 확장 설치 필요
            try:
                connection.execute(text("""
                    UPDATE articles 
                    SET public_id = gen_random_uuid()::text 
                    WHERE public_id IS NULL
                """))
                print(f"✅ Generated UUIDs for {count} articles using PostgreSQL gen_random_uuid()")
            except Exception as pg_error:
                print(f"⚠️ PostgreSQL gen_random_uuid() not available: {pg_error}")
                print("🔄 Falling back to Python UUID generation...")
                
                # Python UUID 생성으로 폴백 (배치 처리)
                updates = []
                result = connection.execute(text("SELECT id FROM articles WHERE public_id IS NULL"))
                articles = result.fetchall()
                
                for article in articles:
                    article_id = article[0]
                    new_uuid = str(uuid.uuid4())
                    updates.append({"public_id": new_uuid, "id": article_id})
                
                # 배치로 업데이트
                for update_data in updates:
                    connection.execute(text("""
                        UPDATE articles 
                        SET public_id = :public_id 
                        WHERE id = :id
                    """), update_data)
                
                print(f"✅ Generated UUIDs for {len(updates)} articles using Python UUID")
        else:
            print("ℹ️ No existing articles found")
            
    except Exception as e:
        print(f"⚠️ Error during UUID generation: {e}")
        raise
    
    # 3. public_id를 NOT NULL로 변경
    print("🔒 Making public_id column NOT NULL...")
    op.alter_column('articles', 'public_id', nullable=False)
    
    # 4. public_id에 UNIQUE 제약조건 및 인덱스 추가
    print("🔍 Adding unique constraint and index...")
    op.create_unique_constraint('uq_articles_public_id', 'articles', ['public_id'])
    op.create_index('ix_articles_public_id', 'articles', ['public_id'])
    
    print("✅ public_id UUID field added successfully!")


def downgrade() -> None:
    """
    public_id 컬럼 제거
    """
    print("🗑️ Removing public_id UUID column...")
    
    # 인덱스 및 제약조건 제거
    try:
        op.drop_constraint('uq_articles_public_id', 'articles', type_='unique')
        op.drop_index('ix_articles_public_id', 'articles')
    except Exception as e:
        print(f"⚠️ Error dropping constraints: {e}")
    
    # 컬럼 제거
    op.drop_column('articles', 'public_id')
    
    print("✅ public_id column removed!")
