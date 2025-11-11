import re
import logging
from typing import Iterable, Optional, List, Tuple

from fastapi import HTTPException, status
from sqlalchemy import and_, func, select, update, or_
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import StyleGuide, StyleCategory
from .schemas import BulkResult, StyleGuideCreate, StyleGuideUpdate

logger = logging.getLogger(__name__)


def map_json_category_to_db(category: str) -> str:
    """JSON 카테고리를 DB 형식으로 변환"""
    mapping = {
        "headlines": "TITLE",    # headlines → TITLE
        "articles": "BODY",      # articles → BODY  
        "captions": "CAPTION"    # captions → CAPTION
    }
    # 이미 DB 형식이면 그대로 반환
    if category in ["TITLE", "BODY", "CAPTION"]:
        return category
    return mapping.get(category.lower(), "BODY")  # 기본값 BODY


def map_db_category_to_json(category: str) -> str:
    """DB 카테고리를 JSON 형식으로 변환 (필요시)"""
    mapping = {
        "TITLE": "headlines",
        "BODY": "articles", 
        "CAPTION": "captions"
    }
    return mapping.get(category, category.lower())


def convert_json_to_styleguide_create(json_item: dict) -> StyleGuideCreate:
    """JSON 형식을 StyleGuideCreate로 변환"""
    # 카테고리 매핑 (JSON → DB 형식)
    category = map_json_category_to_db(json_item.get("category", ""))
    
    # content 배열을 docs로 변환 (하위 호환성)
    content_list = json_item.get("content", [])
    docs = "\n".join(content_list) if content_list else ""
    
    # examples 처리
    examples = json_item.get("examples", {})
    examples_correct = examples.get("correct", [])
    examples_incorrect = examples.get("incorrect", [])
    
    # name 자동 생성
    number = json_item.get("number")
    name = f"Style Guide #{number}" if number else "Style Guide"
    
    return StyleGuideCreate(
        number=number,
        name=name,
        category=category,
        docs=docs,
        content=content_list,
        examples_correct=examples_correct,
        examples_incorrect=examples_incorrect
    )


async def get_by_id(db: AsyncSession, style_id: int, *, include_deleted: bool = False) -> Optional[StyleGuide]:
    stmt = select(StyleGuide).where(StyleGuide.id == style_id)
    if not include_deleted:
        stmt = stmt.where(StyleGuide.deleted_at.is_(None))
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_styleguides(
    db: AsyncSession,
    *,
    category: Optional[StyleCategory] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
    q: Optional[str] = None,
    include_deleted: bool = False,
    skip: int = 0,
    limit: int = 50,
):
    conditions = []
    if not include_deleted:
        conditions.append(StyleGuide.deleted_at.is_(None))
    if category:
        conditions.append(StyleGuide.category == category)
    if name:
        conditions.append(StyleGuide.name == name)
    if version is not None:
        conditions.append(StyleGuide.version == version)
    if q:
        ilike = f"%{q}%"
        conditions.append(StyleGuide.name.ilike(ilike))
    stmt = (
        select(StyleGuide)
        .where(and_(*conditions) if conditions else True)
        .offset(skip)
        .limit(limit)
        .order_by(StyleGuide.id.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_styleguide(db: AsyncSession, data: StyleGuideCreate) -> StyleGuide:
    # 카테고리 정규화 (JSON 형식을 DB 형식으로 변환)
    db_category = map_json_category_to_db(data.category)
    
    # JSON 형식 데이터 처리
    import json
    content_json = json.dumps(data.content) if data.content else None
    examples_correct_json = json.dumps(data.examples_correct) if data.examples_correct else None
    examples_incorrect_json = json.dumps(data.examples_incorrect) if data.examples_incorrect else None
    
    # name이 없으면 자동 생성
    name = data.name
    if not name and data.number:
        name = f"Style Guide #{data.number}"
    elif not name:
        name = "Style Guide"
    
    # docs가 없으면 content에서 생성
    docs = data.docs
    if not docs and data.content:
        docs = "\n".join(data.content)
    
    sg = StyleGuide(
        # 기존 필드들
        name=name,
        category=db_category,  # TITLE, BODY, CAPTION 저장
        docs=docs,
        version=1,
        # 새로운 JSON 형식 필드들
        number=data.number,
        content=content_json,
        examples_correct=examples_correct_json,
        examples_incorrect=examples_incorrect_json,
    )
    db.add(sg)
    try:
        await db.commit()
        await db.refresh(sg)
    except IntegrityError:
        await db.rollback()
        # (number, category) 또는 (name, version) 중복 시 409
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="StyleGuide already exists with same number and category")
    return sg


async def update_styleguide(db: AsyncSession, db_obj: StyleGuide, data: StyleGuideUpdate) -> StyleGuide:
    update_data = data.model_dump(exclude_unset=True)

    # version 변경은 허용하지 않음
    if "version" in update_data:
        update_data.pop("version")

    for field, value in update_data.items():
        setattr(db_obj, field, value)

    try:
        await db.commit()
        await db.refresh(db_obj)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Conflict while updating StyleGuide")
    return db_obj


async def soft_delete(db: AsyncSession, db_obj: StyleGuide) -> None:
    await db.execute(
        update(StyleGuide)
        .where(StyleGuide.id == db_obj.id)
        .values(deleted_at=func.now())
    )
    await db.commit()


async def restore(db: AsyncSession, db_obj: StyleGuide) -> StyleGuide:
    db_obj.deleted_at = None
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def hard_delete(db: AsyncSession, db_obj: StyleGuide) -> None:
    await db.delete(db_obj)
    await db.commit()


async def bulk_import(db: AsyncSession, items: Iterable[StyleGuideCreate], *, mode: str = "error") -> BulkResult:
    items = list(items)
    if len(items) > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bulk import supports up to 100 items per request")
    created = 0
    skipped = 0
    errors = []
    
    for i, item in enumerate(items):
        try:
            await create_styleguide(db, item)
            created += 1
        except HTTPException as e:
            if e.status_code == status.HTTP_409_CONFLICT and mode == "skip":
                skipped += 1
                continue
            # 그 외는 에러로 기록하고 계속 진행 (전체 실패 방지)
            error_detail = f"Item {i+1}: {e.detail}"
            errors.append(error_detail)
            skipped += 1
        except Exception as e:
            # 예상치 못한 에러도 기록
            error_detail = f"Item {i+1}: Unexpected error - {str(e)}"
            errors.append(error_detail)
            skipped += 1
    
    details = {"errors": errors} if errors else None
    return BulkResult(created=created, skipped=skipped, total=created + skipped, details=details)


async def bulk_import_json(db: AsyncSession, json_data: dict, *, mode: str = "error") -> BulkResult:
    """JSON 형식 데이터의 대량 임포트"""
    style_guides_data = json_data.get("style_guides", [])
    
    if not isinstance(style_guides_data, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'style_guides' must be an array")
    
    if len(style_guides_data) > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bulk import supports up to 100 items per request")
    
    # JSON 형식을 StyleGuideCreate로 변환
    converted_items = []
    for item in style_guides_data:
        try:
            converted_item = convert_json_to_styleguide_create(item)
            converted_items.append(converted_item)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid JSON format: {str(e)}")
    
    # 기존 bulk_import 함수 재사용
    return await bulk_import(db, converted_items, mode=mode)


def parse_rule_id(rule_id: str) -> Optional[Tuple[str, int]]:
    """rule_id 문자열을 (DB_CATEGORY, number) 튜플로 파싱"""

    if not rule_id:
        return None

    raw = str(rule_id).strip()
    if not raw:
        return None

    normalized = raw.upper().replace("-", "_")

    # 신형: A/H/C 접두, 공백/0 패딩 허용 (예: "A36", "C 007", "h05")
    m_new = re.search(r"([AHC])\s*0*(\d{1,3})", normalized)
    if m_new:
        prefix = m_new.group(1)
        number = int(m_new.group(2))
        code_map = {"A": "BODY", "H": "TITLE", "C": "CAPTION"}
        category = code_map.get(prefix)
        if category:
            return category, number

    # 구형: articles/headlines/captions_SG### 변형 (간단히 search)
    m_old = re.search(r"(ARTICLES|HEADLINES|CAPTIONS)[ _]?SG\s*0*(\d+)", normalized)
    if m_old:
        original_category = m_old.group(1).lower()
        number = int(m_old.group(2))
        db_category = map_json_category_to_db(original_category)
        if db_category:
            return db_category, number

    logger.warning(f"Could not parse rule_id '{rule_id}' with known formats")
    return None

async def get_guides_by_applicable_rules(
    db: AsyncSession, *, rule_ids: List[str]
) -> List[StyleGuide]:
    """
    파싱된 rule_id 리스트를 기반으로 StyleGuide 객체들을 조회합니다.
    
    예: ["articles_SG018", "headlines_SG005"] 입력 시,
    (category='articles', number=18) OR (category='headlines', number=5)
    조건으로 DB를 조회합니다.

    Args:
        db: Async SQLAlchemy 세션.
        rule_ids: AI 서버에서 받은 rule_id 문자열 리스트.

    Returns:
        조회된 StyleGuide 객체 리스트.
    """
    if not rule_ids:
        return []

    conditions = []
    parsed_rules = []

    # 1. 모든 rule_id를 파싱하여 (category, number) 튜플로 변환
    for rule_id in rule_ids:
        parsed = parse_rule_id(rule_id)
        if parsed:
            parsed_rules.append(parsed)
        else:
            # 파싱 실패 시 경고 로깅 (선택사항)
            logger.warning(f"Could not parse rule_id '{rule_id}'")

    if not parsed_rules:
        return []

    # 2. 각 (category, number) 튜플에 대해 SQLAlchemy 복합 조건 생성
    # 예: (StyleGuide.category == 'articles' AND StyleGuide.number == 18)
    for category, number in parsed_rules:
        conditions.append(
            and_(
                StyleGuide.category == category,
                StyleGuide.number == number
            )
        )
    
    # 3. 모든 조건을 OR로 묶어 최종 쿼리 생성
    # or_() 함수는 여러 조건을 OR 관계로 결합합니다.
    query = select(StyleGuide).where(or_(*conditions))
    
    result = await db.execute(query)
    return result.scalars().all()
