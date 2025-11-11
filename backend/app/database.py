from typing import Annotated, AsyncGenerator

from fastapi import Depends

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

Base = declarative_base()

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,               # 연결 사전 체크
    pool_recycle=1800,                # 30분마다 재연결해 RDS 타임아웃 방지
    connect_args={
        "ssl": True if getattr(settings, "POSTGRES_SSLMODE", "disable") == "require" else False,
        # 필요 시: "ssl_context": ssl.create_default_context()  # 고급 SSL 설정
    },
)

async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as sess:  # async with으로 자동 commit/rollback 처리
        yield sess

# Annotated 별칭: 다른 모듈에서 `db: SessionDep` 만 적으면 세션이 주입됩니다.
SessionDep = Annotated[AsyncSession, Depends(get_db)] 