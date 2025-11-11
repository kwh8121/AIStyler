# backend/app/articles/schemas.py
from ..models import CustomModel
from .models import ArticleCategory, OperationType
from pydantic import Field, field_validator
from typing import Optional, List
from datetime import datetime

# DeepL API 지원 언어 코드 목록
VALID_SOURCE_LANG_CODES = {
    "AR", "BG", "CS", "DA", "DE", "EL", "EN", "EN-GB", "EN-US", 
    "ES", "ET", "FI", "FR", "HU", "ID", "IT", "JA", "KO", "LT", 
    "LV", "NB", "NL", "PL", "PT", "PT-BR", "PT-PT", "RO", "RU", 
    "SK", "SL", "SV", "TR", "UK", "ZH", "ZH-HANS", "ZH-HANT"
}

VALID_TARGET_LANG_CODES = {
    "AR", "BG", "CS", "DA", "DE", "EL", "EN-GB", "EN-US", 
    "ES", "ET", "FI", "FR", "HU", "ID", "IT", "JA", "KO", "LT", 
    "LV", "NB", "NL", "PL", "PT-BR", "PT-PT", "RO", "RU", 
    "SK", "SL", "SV", "TR", "UK", "ZH", "ZH-HANS", "ZH-HANT"
}

class AppliedStyleGuide(CustomModel):
    """적용된 스타일 가이드 정보"""
    style_id: int = Field(..., description="스타일 가이드 ID")
    number: Optional[int] = Field(None, description="스타일 가이드 번호")
    name: str = Field(..., description="스타일 가이드 이름")
    category: str = Field(..., description="스타일 가이드 카테고리")
    docs: Optional[str] = Field(None, description="스타일 가이드 상세 내용")
    applied_at: datetime = Field(..., description="적용 시간")
    note: Optional[str] = Field(None, description="적용 시 메모")
    
    # 문장별 교정 정보 (있는 경우)
    sentence_index: Optional[int] = Field(None, description="문장 순서 (0부터 시작)")
    before_text: Optional[str] = Field(None, description="원본 문장")
    after_text: Optional[str] = Field(None, description="교정된 문장")
    violations: Optional[list] = Field(None, description="해당 문장의 violations 리스트")

class ArticleCorrectionRequest(CustomModel):
    news_key: str = Field(..., min_length=1, max_length=36)
    category: ArticleCategory
    text: str
    prompt: str | None = None  # 선택: 즉시 덮어쓰기

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v):
        """대소문자 구분 없이 카테고리 문자열을 Enum으로 정규화.

        허용 별칭:
        - seo, seo_title -> SEO
        - title, headlines, headline -> TITLE
        - body, articles, article -> BODY
        - caption, captions -> CAPTION
        - articles_translator, translator -> BODY
        """
        if isinstance(v, ArticleCategory):
            return v
        if isinstance(v, str):
            key = v.strip().upper().replace("-", "_")
            aliases = {
                "SEO": ArticleCategory.SEO,
                "SEO_TITLE": ArticleCategory.SEO,
                "TITLE": ArticleCategory.TITLE,
                "HEADLINES": ArticleCategory.TITLE,
                "HEADLINE": ArticleCategory.TITLE,
                "BODY": ArticleCategory.BODY,
                "ARTICLES": ArticleCategory.BODY,
                "ARTICLE": ArticleCategory.BODY,
                "CAPTION": ArticleCategory.CAPTION,
                "CAPTIONS": ArticleCategory.CAPTION,
                "ARTICLE_TRANSLATOR": ArticleCategory.BODY,
                "ARTICLES_TRANSLATOR": ArticleCategory.BODY,
                "TRANSLATOR": ArticleCategory.BODY,
            }
            if key in aliases:
                return aliases[key]
            # 최후 수단: Enum에 직접 매핑 시도
            return ArticleCategory(key)
        return v

class ArticleCorrectionResponse(CustomModel):
    history_id: int
    news_key: str
    category: ArticleCategory
    version: int
    before_text: str
    after_text: str
    operation_type: Optional[OperationType] = None
    created_at: Optional[datetime] = None
    applied_styles: List[AppliedStyleGuide] = Field(default_factory=list, description="적용된 스타일 가이드 목록")

class NewsHistoryResponse(CustomModel):
    history_id: int
    news_key: str
    category: ArticleCategory
    version: int
    original_text: str
    before_text: str
    after_text: str
    operation_type: Optional[OperationType] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    created_at: Optional[datetime] = None
    applied_styles: List[AppliedStyleGuide] = Field(default_factory=list, description="적용된 스타일 가이드 목록")

class TranslationRequest(CustomModel):
    news_key: str = Field(..., min_length=1, max_length=36)
    category: ArticleCategory
    text: str
    source_lang: Optional[str] = Field(None, max_length=10)  # 자동 감지 시 None
    target_lang: str = Field(default="EN-US", max_length=10)

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v):
        if isinstance(v, ArticleCategory):
            return v
        if isinstance(v, str):
            key = v.strip().upper().replace("-", "_")
            aliases = {
                "SEO": ArticleCategory.SEO,
                "SEO_TITLE": ArticleCategory.SEO,
                "TITLE": ArticleCategory.TITLE,
                "HEADLINES": ArticleCategory.TITLE,
                "HEADLINE": ArticleCategory.TITLE,
                "BODY": ArticleCategory.BODY,
                "ARTICLES": ArticleCategory.BODY,
                "ARTICLE": ArticleCategory.BODY,
                "CAPTION": ArticleCategory.CAPTION,
                "CAPTIONS": ArticleCategory.CAPTION,
                "ARTICLE_TRANSLATOR": ArticleCategory.BODY,
                "ARTICLES_TRANSLATOR": ArticleCategory.BODY,
                "TRANSLATOR": ArticleCategory.BODY,
            }
            if key in aliases:
                return aliases[key]
            return ArticleCategory(key)
        return v
    
    @field_validator('source_lang')
    @classmethod
    def validate_source_lang(cls, v):
        if v and v not in VALID_SOURCE_LANG_CODES:
            raise ValueError(f"Invalid source language code: {v}. Valid codes are: {', '.join(sorted(VALID_SOURCE_LANG_CODES))}")
        return v
    
    @field_validator('target_lang')
    @classmethod
    def validate_target_lang(cls, v):
        if v not in VALID_TARGET_LANG_CODES:
            raise ValueError(f"Invalid target language code: {v}. Valid codes are: {', '.join(sorted(VALID_TARGET_LANG_CODES))}")
        return v

class TranslationResponse(CustomModel):
    translated_text: str
    source_lang: str
    target_lang: str
    history_id: Optional[int] = None
    version: Optional[int] = None

class HistoryFilterRequest(CustomModel):
    operation_type: Optional[OperationType] = None

class TitleGenerationRequest(CustomModel):
    input_text: str = Field(..., min_length=1, description="원본 제목 텍스트")
    selected_type: Optional[str] = Field(None, description="선택된 유형")
    data_type: str = Field(..., description="헤드라인 작성 규칙")
    model: str = Field(default="o4-mini", description="사용할 GPT 모델명")
    guideline_text: Optional[str] = Field(None, description="추가 가이드라인 텍스트")
    news_key: Optional[str] = Field(None, description="뉴스 키 (히스토리 저장용)")

class TitleGenerationResponse(CustomModel):
    edited_title: str = Field(..., description="개선된 제목")
    seo_titles: list[str] = Field(..., description="SEO 최적화된 제목 목록")
    raw_response: str = Field(..., description="GPT의 전체 응답")

# CMS 연동을 위한 스키마들
class CMSSaveRequest(CustomModel):
    article_id: str = Field(..., description="클라이언트가 생성한 고유 Article ID (news_key로 저장됨)")
    category: str = Field(..., description="카테고리 (headlines, articles, captions, articles_translator, seo)")
    content: str = Field(..., description="콘텐츠")
    author_id: str = Field(..., max_length=100, description="CMS 작성자 아이디(최대 100자)")

class CMSSaveResponse(CustomModel):
    article_id: str = Field(..., description="저장된 Article ID (요청시 받은 값과 동일)")
    news_key: str = Field(..., description="뉴스 키 (article_id와 동일)")
    category: str = Field(..., description="카테고리")

class CMSGetResponse(CustomModel):
    article_id: str = Field(..., description="Article UUID")
    news_key: str = Field(..., description="뉴스 키")
    category: str = Field(..., description="카테고리")
    content: str = Field(..., description="콘텐츠")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")

class HistoryRestoreRequest(CustomModel):
    """히스토리 복원 요청"""
    news_key: str = Field(..., min_length=1, max_length=255, description="뉴스 키")
    category: ArticleCategory = Field(..., description="카테고리")
    history_id: int = Field(..., description="복원할 히스토리 ID")

    @field_validator("category", mode="before")
    @classmethod
    def convert_category(cls, v):
        """대소문자 구분 없이 카테고리 문자열을 Enum으로 정규화 (ArticleCorrectionRequest와 동일 로직)"""
        if isinstance(v, ArticleCategory):
            return v
        if isinstance(v, str):
            key = v.strip().upper().replace("-", "_")
            aliases = {
                "SEO": ArticleCategory.SEO,
                "SEO_TITLE": ArticleCategory.SEO,
                "TITLE": ArticleCategory.TITLE,
                "HEADLINES": ArticleCategory.TITLE,
                "HEADLINE": ArticleCategory.TITLE,
                "BODY": ArticleCategory.BODY,
                "ARTICLES": ArticleCategory.BODY,
                "ARTICLE": ArticleCategory.BODY,
                "CAPTION": ArticleCategory.CAPTION,
                "CAPTIONS": ArticleCategory.CAPTION,
                "ARTICLE_TRANSLATOR": ArticleCategory.BODY,
                "ARTICLES_TRANSLATOR": ArticleCategory.BODY,
                "TRANSLATOR": ArticleCategory.BODY,
            }
            if key in aliases:
                return aliases[key]
        # Fallback: Pydantic이 기본 변환 시도
        return v

class HistoryRestoreResponse(CustomModel):
    """히스토리 복원 응답"""
    history_id: int = Field(..., description="새로 생성된 히스토리 ID")
    version: int = Field(..., description="버전 번호")
    before_text: str = Field(..., description="복원된 입력 텍스트")
    after_text: str = Field(..., description="복원된 출력 텍스트")
