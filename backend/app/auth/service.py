from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict

from ..users import service as user_service
from ..users.models import User

from datetime import datetime, timedelta
from jose import jwt, JWTError
from ..config import settings


async def create_access_token(user: User) -> str:
    """
    사용자 객체를 기반으로 Access Token을 생성합니다.
    토큰에 역할(role)과 타입(type) 정보를 추가합니다.
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": user.email,
        "role": user.role,  # 사용자 역할 정보 추가
        "type": "access",   # 토큰 타입 명시
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

async def create_refresh_token(user: User) -> str:
    """
    사용자 객체를 기반으로 Refresh Token을 생성합니다.
    토큰에 타입(type) 정보를 추가합니다.
    """
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {
        "sub": user.email,
        "type": "refresh",  # 토큰 타입 명시
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

async def _decode_token(token: str) -> Optional[Dict]:
    """
    토큰을 디코딩하고 기본적인 유효성을 검사하는 내부 헬퍼 함수.
    (서명, 만료 시간 등)
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        # 토큰 디코딩 실패 (변조, 만료 등) 시 None 반환
        return None

async def get_user_from_access_token(token: str, db: AsyncSession) -> Optional[User]:
    """
    Access Token을 검증하고 해당 사용자를 반환합니다.
    """
    payload = await _decode_token(token)
    
    # 페이로드가 없거나, 토큰 타입이 'access'가 아니면 유효하지 않음
    if payload is None or payload.get("type") != "access":
        return None
        
    email = payload.get("sub")
    if email is None:
        return None
        
    return await user_service.get_user_by_email(email, db)


async def get_user_from_refresh_token(token: str, db: AsyncSession) -> Optional[User]:
    """
    Refresh Token을 검증하고 해당 사용자를 반환합니다.
    이 함수는 /auth/refresh 엔드포인트에서 사용됩니다.
    """
    # 1. 토큰을 디코딩하고 기본적인 유효성을 검사합니다.
    payload = await _decode_token(token)

    # 2. 페이로드가 없거나, 토큰 타입이 'refresh'가 아니면 유효하지 않은 토큰입니다.
    #    이를 통해 Access Token으로 Refresh를 시도하는 등의 공격을 막습니다.
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type or invalid token",
        )
    
    # 3. 페이로드에서 사용자 이메일(sub)을 추출합니다.
    email = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not find user from token",
        )
        
    # 4. 이메일을 사용하여 데이터베이스에서 사용자 정보를 조회합니다.
    user = await user_service.get_user_by_email(email, db)
    
    # 5. 사용자가 존재하지 않으면(예: 탈퇴한 사용자), 에러를 발생시킵니다.
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token not found",
        )
        
    return user

async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> Optional[User]:
    """
    사용자 이메일과 비밀번호로 인증을 시도합니다.
    성공 시 User 객체를, 실패 시 None을 반환합니다.
    """
    
    user = await user_service.get_user_by_email(email, db)
    
    if not user or not await user_service.verify_password(password, user.hashed_password):
        return None
    return user