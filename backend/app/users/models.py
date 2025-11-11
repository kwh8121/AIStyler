# backend/app/users/models.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), index=True)
    email = Column(String(100), unique=True, index=True)
    role = Column(String(50), default="user")
    last_login = Column(DateTime(timezone=True))
    hashed_password = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True))

    articles = relationship("Article", back_populates="user")

    def __repr__(self) -> str:
        return f"User(id={self.id}, name={self.name!r}, email={self.email!r}, role={self.role!r})"
    def __str__(self) -> str:
        return f"{self.name} ({self.email})"