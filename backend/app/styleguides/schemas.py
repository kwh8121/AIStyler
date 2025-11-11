from datetime import datetime
from typing import List, Literal, Optional, Dict, Any

from pydantic import Field, field_validator

from ..models import CustomModel
from .models import StyleCategory


class StyleGuideJSONInput(CustomModel):
    """JSON 파일에서 입력되는 스타일 가이드 형식"""
    number: int
    category: str = Field(..., description="articles, headlines, captions")
    content: List[str] = Field(..., description="규칙 내용 배열")
    examples: Dict[str, List[str]] = Field(default_factory=dict, description="correct/incorrect 예시")


class StyleGuideBase(CustomModel):
    """기본 스타일 가이드 필드 (하위 호환성 유지)"""
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    category: str = Field(..., description="카테고리 (TITLE, BODY, CAPTION 또는 articles, headlines, captions)")
    docs: Optional[str] = Field(None, description="기존 형식의 문서")


class StyleGuideCreate(StyleGuideBase):
    """스타일 가이드 생성 (JSON 형식 지원)"""
    # JSON 형식 필드들
    number: Optional[int] = Field(None, description="스타일 가이드 번호")
    content: Optional[List[str]] = Field(None, description="규칙 내용 배열")
    examples_correct: Optional[List[str]] = Field(None, description="올바른 예시 배열")
    examples_incorrect: Optional[List[str]] = Field(None, description="잘못된 예시 배열")
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        """카테고리 검증 - JSON 형식과 DB 형식 모두 허용"""
        valid_json = {"headlines", "articles", "captions"}  # JSON 입력 형식
        valid_db = {"TITLE", "BODY", "CAPTION"}  # DB/프론트엔드 형식
        
        if v not in (valid_json | valid_db):
            raise ValueError(f"Invalid category: {v}. Must be one of: {', '.join(valid_json | valid_db)}")
        return v


class StyleGuideUpdate(CustomModel):
    """스타일 가이드 업데이트"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    category: Optional[str] = None
    docs: Optional[str] = None
    number: Optional[int] = None
    content: Optional[List[str]] = None
    examples_correct: Optional[List[str]] = None
    examples_incorrect: Optional[List[str]] = None


class StyleGuideOut(CustomModel):
    """스타일 가이드 출력 (모든 필드 포함)"""
    id: int
    # 기존 필드들
    name: Optional[str] = None
    category: str
    docs: Optional[str] = None
    # JSON 형식 필드들
    number: Optional[int] = None
    content: Optional[List[str]] = None
    examples_correct: Optional[List[str]] = None
    examples_incorrect: Optional[List[str]] = None
    # 메타데이터
    version: int
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class BulkStyleGuideCreate(CustomModel):
    """대량 스타일 가이드 생성 (JSON/기존 형식 모두 지원)"""
    items: Optional[List[StyleGuideCreate]] = Field(None, description="기존 형식의 스타일 가이드 배열")
    style_guides: Optional[List[StyleGuideJSONInput]] = Field(None, description="JSON 형식의 스타일 가이드 배열")
    mode: Literal["error", "skip"] = Field(default="error", description="중복 시 처리 방식")
    
    @field_validator('*')
    @classmethod
    def validate_input_format(cls, v, info):
        """items 또는 style_guides 중 하나는 반드시 제공되어야 함"""
        if info.field_name == 'style_guides':
            # 전체 모델 검증은 model_validator에서 수행
            return v
        return v


class BulkResult(CustomModel):
    """대량 작업 결과"""
    created: int
    skipped: int
    total: int
    details: Optional[Dict[str, Any]] = Field(None, description="추가 상세 정보")

