from pydantic import EmailStr, Field
from typing import Optional
from datetime import datetime

from ..models import CustomModel 

class UserBase(CustomModel):
    email: EmailStr = Field(..., json_schema_extra={"example": "user@example.com"})
    name: str = Field(..., min_length=1, max_length=50, json_schema_extra={"example": "John Doe"})

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, json_schema_extra={"example": "strongpassword123"})

class UserUpdate(CustomModel):
    email: Optional[EmailStr] = Field(None, json_schema_extra={"example": "new_email@example.com"})
    name: Optional[str] = Field(None, min_length=1, max_length=50, json_schema_extra={"example": "Johnathan Doe"})

class UserUpdatePassword(CustomModel):
    current_password: str = Field(..., json_schema_extra={"example": "strongpassword123"})
    new_password: str = Field(..., min_length=8, json_schema_extra={"example": "new_strong_password_456"})

class UserPublic(UserBase):
    id: int = Field(..., json_schema_extra={"example": 1})

class UserMe(UserPublic):
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

class AdminCreate(UserCreate):
    role: str = Field("user", pattern="^(user|admin)$")