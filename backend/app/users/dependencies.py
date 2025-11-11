# backend/app/users/dependencies.py
from fastapi import Header, Depends
from ..database import SessionDep
from .models import User as UserModel
from .service import get_or_create_user

async def resolve_actor_user(
    db: SessionDep,
    x_user_id: int | None = Header(default=None, convert_underscores=False),
    x_user_email: str | None = Header(default=None, convert_underscores=False),
) -> UserModel:
    return await get_or_create_user(db, email=x_user_email, user_id=x_user_id)