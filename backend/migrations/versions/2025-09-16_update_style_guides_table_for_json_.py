"""Update style_guides table for JSON compatibility

Revision ID: 27ff3b8a13ec
Revises: cda207fa69c1
Create Date: 2025-09-16 16:10:42.510924

"""
from typing import Sequence, Union
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = '27ff3b8a13ec'
down_revision: Union[str, None] = 'cda207fa69c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    StyleGuide 테이블을 JSON 형식과 호환되도록 업데이트
    기존 데이터는 보존하고 새로운 필드들을 추가
    """
    connection = op.get_bind()
    
    # 1. 기존 데이터 백업 및 확인
    print("🔍 Checking existing data...")
    result = connection.execute(text("SELECT COUNT(*) as count FROM style_guides WHERE deleted_at IS NULL"))
    existing_count = result.fetchone()[0]  # 인덱스로 접근
    print(f"📊 Found {existing_count} existing style guides")
    
    # 2. 새로운 JSON 컬럼들 추가
    print("➕ Adding new JSON-compatible columns...")
    
    # number 컬럼 추가 (nullable, 기존 데이터는 나중에 마이그레이션)
    op.add_column('style_guides', sa.Column('number', sa.Integer(), nullable=True))
    
    # content 컬럼 추가 (JSON 배열)
    op.add_column('style_guides', sa.Column('content', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # examples_correct 컬럼 추가 (JSON 배열)
    op.add_column('style_guides', sa.Column('examples_correct', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # examples_incorrect 컬럼 추가 (JSON 배열)
    op.add_column('style_guides', sa.Column('examples_incorrect', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # 3. category 컬럼 타입 변경 (ENUM -> VARCHAR)
    print("🔄 Converting category column from ENUM to VARCHAR...")
    
    # 기존 category 데이터 백업
    if existing_count > 0:
        print("💾 Backing up existing category data...")
        result = connection.execute(text("""
            SELECT id, category, name, docs 
            FROM style_guides 
            WHERE deleted_at IS NULL
            ORDER BY id
        """))
        existing_data = result.fetchall()
        
        # 카테고리 값은 기존 형식 유지 (TITLE, BODY, CAPTION)
        # ENUM에서 VARCHAR로 타입만 변경
        print("🔄 Converting category column type only...")
        
        # 임시 컬럼 생성
        op.add_column('style_guides', sa.Column('category_new', sa.String(50), nullable=True))
        
        # 기존 값 그대로 복사 (변환하지 않음)
        for row in existing_data:
            connection.execute(text("""
                UPDATE style_guides 
                SET category_new = :category 
                WHERE id = :id
            """), {"category": row['category'], "id": row['id']})
        
        # 기존 category 컬럼 삭제
        op.drop_column('style_guides', 'category')
        
        # 새 컬럼을 원래 이름으로 변경
        op.alter_column('style_guides', 'category_new', new_column_name='category')
    else:
        # 데이터가 없으면 직접 컬럼 타입 변경
        op.drop_column('style_guides', 'category')
        op.add_column('style_guides', sa.Column('category', sa.String(50), nullable=False, server_default='BODY'))
    
    # 4. 기존 필드들을 nullable로 변경 (하위 호환성)
    print("🔧 Making legacy columns nullable...")
    op.alter_column('style_guides', 'name', nullable=True)
    op.alter_column('style_guides', 'docs', nullable=True)
    
    # 5. 기존 데이터를 새로운 형식으로 마이그레이션
    if existing_count > 0:
        print("📦 Migrating existing data to new format...")
        
        # 기존 데이터 다시 조회 (category가 변경되었으므로)
        result = connection.execute(text("""
            SELECT id, name, docs, category 
            FROM style_guides 
            WHERE deleted_at IS NULL AND docs IS NOT NULL
            ORDER BY id
        """))
        existing_data = result.fetchall()
        
        for i, row in enumerate(existing_data, 1):
            # docs를 content 배열로 변환
            docs_text = row['docs'] or ""
            content_array = [docs_text] if docs_text.strip() else []
            
            # number를 순차적으로 할당
            number_value = i
            
            connection.execute(text("""
                UPDATE style_guides 
                SET 
                    number = :number,
                    content = :content,
                    examples_correct = :examples_correct,
                    examples_incorrect = :examples_incorrect
                WHERE id = :id
            """), {
                "number": number_value,
                "content": json.dumps(content_array),
                "examples_correct": json.dumps([]),  # 빈 배열로 초기화
                "examples_incorrect": json.dumps([]),  # 빈 배열로 초기화
                "id": row['id']
            })
    
    # 6. 새로운 인덱스 생성
    print("🔍 Creating new indexes...")
    
    # 기존 constraint 삭제
    try:
        op.drop_constraint('uq_style_guides_name_version', 'style_guides', type_='unique')
    except Exception as e:
        print(f"⚠️ Could not drop old unique constraint: {e}")
    
    # 새로운 unique constraint 생성 (number + category)
    op.create_unique_constraint('uq_style_guides_number_category', 'style_guides', ['number', 'category'])
    
    # number 인덱스 생성
    op.create_index('ix_style_guides_number', 'style_guides', ['number'])
    
    # 7. ENUM 타입 정리 (사용하지 않는 경우)
    print("🧹 Cleaning up unused ENUM type...")
    try:
        op.execute("DROP TYPE IF EXISTS style_category")
    except Exception as e:
        print(f"⚠️ Could not drop ENUM type (may still be in use): {e}")
    
    print(f"✅ Migration completed! Processed {existing_count} existing records.")


def downgrade() -> None:
    """
    변경사항을 롤백 (주의: 데이터 손실 가능)
    """
    connection = op.get_bind()
    
    print("⚠️ WARNING: Downgrade will lose JSON format data!")
    
    # 1. 새로운 컬럼들 삭제
    print("🗑️ Removing JSON columns...")
    op.drop_column('style_guides', 'examples_incorrect')
    op.drop_column('style_guides', 'examples_correct')
    op.drop_column('style_guides', 'content')
    op.drop_column('style_guides', 'number')
    
    # 2. 새로운 인덱스 및 제약조건 삭제
    try:
        op.drop_constraint('uq_style_guides_number_category', 'style_guides', type_='unique')
        op.drop_index('ix_style_guides_number', 'style_guides')
    except Exception as e:
        print(f"⚠️ Error dropping constraints: {e}")
    
    # 3. ENUM 타입 재생성
    print("🔄 Recreating ENUM type...")
    style_category_enum = postgresql.ENUM('TITLE', 'BODY', 'CAPTION', name='style_category')
    style_category_enum.create(connection)
    
    # 4. category 컬럼을 다시 ENUM으로 변경
    print("🔄 Converting category back to ENUM...")
    
    # 임시 컬럼 생성
    op.add_column('style_guides', sa.Column('category_enum', style_category_enum, nullable=True))
    
    # 데이터는 이미 TITLE, BODY, CAPTION 형식이므로 그대로 복사
    result = connection.execute(text("SELECT id, category FROM style_guides WHERE deleted_at IS NULL"))
    for row in result.fetchall():
        category = row['category']
        # 이미 올바른 형식이므로 그대로 사용
        connection.execute(text("""
            UPDATE style_guides 
            SET category_enum = :category 
            WHERE id = :id
        """), {"category": category, "id": row['id']})
    
    # 기존 컬럼 삭제 후 이름 변경
    op.drop_column('style_guides', 'category')
    op.alter_column('style_guides', 'category_enum', new_column_name='category')
    op.alter_column('style_guides', 'category', nullable=False)
    
    # 5. 필수 필드로 되돌리기
    print("🔒 Making legacy columns required...")
    op.alter_column('style_guides', 'name', nullable=False)
    op.alter_column('style_guides', 'docs', nullable=False)
    
    # 6. 기존 제약조건 재생성
    op.create_unique_constraint('uq_style_guides_name_version', 'style_guides', ['name', 'version'])
    
    print("✅ Downgrade completed!")
