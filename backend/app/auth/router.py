from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict

from ..database import SessionDep
from .service import authenticate_user, create_access_token, create_refresh_token, get_user_from_refresh_token
from ..users.service import update_last_login

from .schema import TokenRefreshRequest

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
async def login(
    db: SessionDep,
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Dict[str, str]:
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = await create_access_token(user=user)
    refresh_token = await create_refresh_token(user=user)

    # last_login 업데이트 (로그인 성공 시)
    await update_last_login(user=user, db=db)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }

@router.post("/refresh")
async def refresh_access_token(
    request: TokenRefreshRequest,
    db: SessionDep
) -> Dict[str, str]:
    # 1. 전달받은 Refresh Token을 검증하고, payload를 디코딩하는 의존성/서비스 필요
    user = await get_user_from_refresh_token(request.refresh_token, db)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")

    # 2. 새로운 Access Token 발급
    new_access_token = await create_access_token(user=user)
    # 토큰 리프레시 시점에도 마지막 접속 시간 갱신(선택 사항)
    await update_last_login(user=user, db=db)
    return {"access_token": new_access_token, "token_type": "bearer"}