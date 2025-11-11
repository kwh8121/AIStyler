# backend/app/styleguides/models.py
from enum import Enum as PyEnum
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, JSON,
    Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base

class StyleCategory(str, PyEnum):
    """DB에 저장되는 표준 카테고리 (대문자)"""
    TITLE = "TITLE"
    BODY = "BODY"
    CAPTION = "CAPTION"

class StyleGuide(Base):
    __tablename__ = "style_guides"
    __table_args__ = (
        # 기존 unique constraint 유지하되 nullable 고려
        UniqueConstraint("number", "category", name="uq_style_guides_number_category"),
        Index("ix_style_guides_category", "category"),
        Index("ix_style_guides_number", "number"),
    )

    id = Column(Integer, primary_key=True, index=True)
    
    # JSON 형식 필드들 (새로운 구조)
    number = Column(Integer, nullable=True, index=True)        # JSON의 number 필드
    category = Column(String(50), nullable=False, index=True)  # articles, headlines, captions 또는 기존 enum
    content = Column(JSON, nullable=True)                      # JSON 배열: ["규칙1", "규칙2"]
    examples_correct = Column(JSON, nullable=True)             # JSON 배열: ["예시1", "예시2"]
    examples_incorrect = Column(JSON, nullable=True)           # JSON 배열: ["예시1", "예시2"]
    
    # 기존 필드들 (하위 호환성을 위해 nullable로 변경)
    name = Column(String(128), nullable=True)                  # 기존 가이드 식별명
    docs = Column(Text, nullable=True)                         # 기존 가이드 설명/규칙 본문
    
    # 공통 필드들
    version = Column(Integer, default=1, nullable=False)       # 가이드 자체 버전
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True))

    # history와의 조인 테이블을 통해 간접 연결
    applied_histories = relationship(
        "TextCorrectionHistoryStyle",
        back_populates="style_guide",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"StyleGuide(id={self.id}, name={self.name!r}, category={self.category!r}, version={self.version})"
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.category})"


class TextCorrectionHistoryStyle(Base):
    """
    교정 이력(버전)과 스타일가이드 간 다대다 연결.
    한 번의 교정에서 여러 가이드를 참고할 수 있음.
    """
    __tablename__ = "text_correction_history_styles"
    __table_args__ = (
        # Changed: Now allows multiple sentences per (history_id, style_id) combination
        # Each sentence is uniquely identified by sentence_index
        UniqueConstraint("history_id", "style_id", "sentence_index", name="uq_history_style_sentence"),
        Index("ix_hist_style_history", "history_id"),
        Index("ix_hist_style_style", "style_id"),
        Index("ix_hist_style_sentence", "sentence_index"),
    )

    id = Column(Integer, primary_key=True)
    history_id = Column(Integer, ForeignKey("text_correction_history.id", ondelete="CASCADE"), nullable=False)
    style_id = Column(Integer, ForeignKey("style_guides.id", ondelete="CASCADE"), nullable=False)
    applied_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    note = Column(Text, nullable=True)
    
    # 문장별 교정 정보 (새로운 필드들)
    sentence_index = Column(Integer, nullable=True, comment="문장 순서 (0부터 시작)")
    before_text = Column(Text, nullable=True, comment="원본 문장")
    after_text = Column(Text, nullable=True, comment="교정된 문장")
    violations = Column(JSON, nullable=True, comment="해당 문장의 violations 리스트")

    style_guide = relationship("StyleGuide", back_populates="applied_histories")
    history = relationship("TextCorrectionHistory", back_populates="applied_styles")

    def __repr__(self) -> str:
        return f"TCHStyle(id={self.id}, history_id={self.history_id}, style_id={self.style_id})"
    def __str__(self) -> str:
        return f"Hist {self.history_id} / Style {self.style_id}"