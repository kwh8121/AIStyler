# backend/app/articles/models.py
from enum import Enum as PyEnum
import uuid
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Boolean,
    Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base

class ArticleCategory(str, PyEnum):
    SEO = "SEO"
    TITLE = "TITLE"
    BODY = "BODY"
    CAPTION = "CAPTION"

class ArticleStatus(str, PyEnum):
    # Will be ENUM with uppercase values after migration
    DRAFT = "DRAFT"
    TRANSLATING = "TRANSLATING"
    TRANSLATED = "TRANSLATED"
    CORRECTING = "CORRECTING"
    COMPLETED = "COMPLETED"

class OperationType(str, PyEnum):
    # Will be uppercase values after migration
    CORRECTION = "CORRECTION"
    TRANSLATION = "TRANSLATION"
    TRANSLATION_CORRECTION = "TRANSLATION_CORRECTION"
    RESTORATION = "RESTORATION"  # 히스토리 복원

class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        Index("ix_articles_news_key_category", "news_key", "category"),
        UniqueConstraint("news_key", "category", name="uq_articles_news_key_category"),
    )

    id = Column(Integer, primary_key=True, index=True)
    public_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()), index=True)  # 외부 노출용 UUID
    # 하나의 뉴스(기사)를 묶는 식별자
    news_key = Column(String(255), nullable=False, index=True)  # 클라이언트가 제공하는 고유 ID (최대 255자)
    category = Column(SQLEnum(ArticleCategory, name="article_category"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    status = Column(SQLEnum(ArticleStatus, name="articlestatus"), default=ArticleStatus.DRAFT, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    cms_author = Column(String(100), nullable=True)  # CMS 작성자 정보

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="articles")

    def __repr__(self) -> str:
        return f"Article(id={self.id}, news_key={self.news_key!r}, category={self.category!r}, status={self.status!r})"
    def __str__(self) -> str:
        return f"{self.news_key}-{self.category}"

class TextCorrectionHistory(Base):
    __tablename__ = "text_correction_history"
    __table_args__ = (
        Index("ix_tch_news_key_category_version", "news_key", "category", "version"),
        Index("ix_tch_news_key_category_created", "news_key", "category", "created_at"),
        Index("ix_tch_news_key_created", "news_key", "created_at"),
        Index("ix_tch_news_key", "news_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    news_key = Column(String(255), nullable=False)
    category = Column(SQLEnum(ArticleCategory, name="article_category"), nullable=False)
    version = Column(Integer, nullable=False)
    original_text = Column(Text, nullable=True)  # 완전 원본 텍스트 (변환 전)
    before_text = Column(Text, nullable=False)
    after_text = Column(Text, nullable=False)
    prompt = Column(Text, nullable=True)  # 선택 입력 프롬프트
    operation_type = Column(
        SQLEnum(OperationType, name="operationtype"),
        nullable=False,
        server_default="CORRECTION",
        index=True
    )
    source_lang = Column(String(10), nullable=True)  # 번역 시 소스 언어
    target_lang = Column(String(10), nullable=True)  # 번역 시 타겟 언어
    created_by_user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # article relationship 제거
    applied_styles = relationship(
        "TextCorrectionHistoryStyle",
        back_populates="history",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"TextCorrectionHistory(id={self.id}, news_key={self.news_key}, category={self.category}, version={self.version})"
    def __str__(self) -> str:
        return f"History#{self.id} {self.news_key}-{self.category} v{self.version}"


class SEOGenerationHistory(Base):
    """SEO 제목 생성 히스토리 테이블"""
    __tablename__ = "seo_generation_history"
    __table_args__ = (
        Index("ix_seo_history_news_key", "news_key"),
        Index("ix_seo_history_created_at", "created_at"),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    news_key = Column(String(255), nullable=True)  # 기사 연결은 선택적
    input_text = Column(Text, nullable=False)  # 원본 입력 텍스트
    edited_title = Column(Text, nullable=False)  # 생성된 메인 제목
    seo_titles = Column(Text, nullable=False)  # JSON 문자열로 저장 (SQLite 호환성)
    raw_response = Column(Text)  # GPT 원본 응답
    prompt_used = Column(Text)  # 실제 사용된 프롬프트
    model = Column(String(50))  # gpt-4, o4-mini 등
    data_type = Column(Text)  # 헤드라인 작성 규칙
    guideline_text = Column(Text, nullable=True)  # 추가 가이드라인
    created_by_user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self) -> str:
        return f"SEOGenerationHistory(id={self.id}, news_key={self.news_key}, created_at={self.created_at})"
    def __str__(self) -> str:
        return f"SEO#{self.id} {self.news_key or 'no-article'}"
    
class ArticlePrompt(Base):
    __tablename__ = "article_prompts"
    __table_args__ = (
        UniqueConstraint("category", name="uq_article_prompts_category"),
    )

    id = Column(Integer, primary_key=True, index=True)
    category = Column(SQLEnum(ArticleCategory, name="article_prompts_category"), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"ArticlePrompt(id={self.id}, category={self.category!r})"
    def __str__(self) -> str:
        return f"Prompt({self.category})"
