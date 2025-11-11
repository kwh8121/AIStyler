from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import random

from passlib.context import CryptContext

from .models import User as UserModel
from .schema import UserCreate, UserUpdate

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_user(user_data: UserCreate, db: AsyncSession, role: str = "user") -> UserModel:
    existing_user = await get_user_by_email(user_data.email, db)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    
    hashed_password = pwd_context.hash(user_data.password)
    db_user = UserModel(
        name=user_data.name,
        email=user_data.email,
        hashed_password=hashed_password,
        role=role,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[UserModel]:
    """모든 사용자 목록을 페이지네이션하여 조회합니다."""
    result = await db.execute(
        select(UserModel)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def update_user(db: AsyncSession, db_user: UserModel, user_in: UserUpdate) -> UserModel:
    """사용자 정보를 수정합니다. None이 아닌 필드만 업데이트합니다."""
    # Pydantic 모델을 딕셔너리로 변환하되, 명시적으로 값이 할당된 필드만 추출합니다.
    update_data = user_in.model_dump(exclude_unset=True)

    if "email" in update_data:
        existing_user = await get_user_by_email(update_data["email"], db)
        if existing_user and existing_user.id != db_user.id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered by another user")

    # 객체 필드 업데이트
    for field, value in update_data.items():
        setattr(db_user, field, value)
    
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def delete_user_by_id(db: AsyncSession, user_id: int) -> None:
    """사용자 ID를 기반으로 사용자를 삭제합니다. (Soft Delete도 고려 가능)"""
    user_to_delete = await get_user_by_id(user_id, db)
    if not user_to_delete:
        # 이미 없거나 잘못된 ID인 경우, 그냥 통과시켜도 무방합니다. (멱등성)
        return
    await db.delete(user_to_delete)
    await db.commit()
    return


async def get_user_by_id(user_id: int, db: AsyncSession) -> Optional[UserModel]:
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    return result.scalar_one_or_none()

async def get_user_by_email(email: str, db: AsyncSession) -> Optional[UserModel]:
    result = await db.execute(select(UserModel).where(UserModel.email == email))
    return result.scalar_one_or_none()

async def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def update_last_login(user: UserModel, db: AsyncSession) -> None:
    # DB 서버 시간 기준으로 기록
    user.last_login = func.now()
    await db.commit()

async def get_or_create_user(
    db: AsyncSession,
    *,
    email: str | None = None,
    user_id: int | None = None,
    name: str | None = None,
    role: str = "user"
) -> UserModel:
    if user_id is not None:
        user = await get_user_by_id(user_id, db)
        if user:
            return user
    if email is not None:
        user = await get_user_by_email(email, db)
        if user:
            return user
    # 생성: 비밀번호 없이 생성
    db_user = UserModel(
        name=name or (email.split("@")[0] if email else "guest"),
        email=email or f"guest_{random.randint(0, 999999999)}@local",
        hashed_password=None,
        role=role,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user