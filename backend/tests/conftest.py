import os
import sys
from pathlib import Path
import asyncio
import pytest

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient, ASGITransport

# 테스트용 환경 변수 세팅 (app 모듈 임포트 전에 적용)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_SSLMODE", "disable")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("CORS_ORIGINS", "*")

# sys.path에 backend 추가하여 'app' 패키지 검색 가능하게 함
repo_root = Path(__file__).resolve().parents[2]
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.main import app
from app.database import Base
from app.database import get_db as real_get_db


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    # 메모리 SQLite로 빠른 테스트
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def db(test_engine):
    async_session = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.fixture(autouse=True)
async def override_db(db):
    async def _get_db():
        yield db
    app.dependency_overrides[real_get_db] = _get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


