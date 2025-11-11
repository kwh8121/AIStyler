from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..database import SessionDep
from ..users.models import User as UserModel

from .schema import UserCreate, UserUpdate, UserPublic, UserMe, AdminCreate
from . import service as user_service
from ..auth.dependencies import CurrentUser, require_admin

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserMe, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: SessionDep):
    return await user_service.create_user(user_data, db)

@router.get("/", response_model=List[UserPublic], dependencies=[Depends(require_admin)])
async def list_users(db: SessionDep, skip: int = 0, limit: int = 100):
    return await user_service.get_users(db, skip=skip, limit=limit)

@router.get("/me", response_model=UserMe)
async def read_users_me(current_user: UserModel = CurrentUser):
    return current_user

@router.patch("/me", response_model=UserMe)
async def update_users_me(
    db: SessionDep,
    user_update_data: UserUpdate,
    current_user: UserModel = CurrentUser,
):
    return await user_service.update_user(db, db_user=current_user, user_in=user_update_data)

@router.get("/{user_id}", response_model=UserPublic)
async def get_user_by_id_route(
    user_id: int,
    db: SessionDep,
    current_user: UserModel = CurrentUser,
):
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    user = await user_service.get_user_by_id(user_id, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_users_me(
    db: SessionDep,
    current_user: UserModel = CurrentUser,
):
    await user_service.delete_user_by_id(db, user_id=current_user.id)
    return

@router.post("/admin/create", response_model=UserMe, dependencies=[Depends(require_admin)])
async def create_admin_user(
    user_data: AdminCreate, # 관리자용 스키마 사용
    db: SessionDep
):
    """(관리자 전용) 새로운 사용자(관리자 포함)를 생성합니다."""
    return await user_service.create_user(user_data, db)