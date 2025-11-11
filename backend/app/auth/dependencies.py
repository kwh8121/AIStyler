from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from ..database import SessionDep

from ..users.models import User
from ..auth import service as auth_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user_from_access_token(
    db: SessionDep,
    token: str = Depends(oauth2_scheme), 
) -> User:
    user = await auth_service.get_user_from_access_token(token=token, db=db)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

CurrentUser = Depends(get_current_user_from_access_token)

def require_admin(
    current_user: User = CurrentUser
) -> None:
    """
    현재 사용자가 'admin' 역할을 가지고 있는지 확인하는 의존성.
    관리자가 아닐 경우, 403 Forbidden 에러를 발생시킵니다.
    이 함수는 반환값이 없으며, 오직 권한 체크의 목적으로만 사용됩니다.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Admin privileges required"
        )

