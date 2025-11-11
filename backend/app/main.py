import os
import time
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from .database import engine, async_session_factory
from .db_models import *
from .config import settings
from .auth.router import router as auth_router
from .users.router import router as users_router
from .styleguides.router import router as styleguides_router
from .articles.router import router as articles_router

# 로깅 설정 (Docker 환경 최적화)
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # stdout으로 명시적 출력
    ]
)

# 특정 모듈 로그 레벨 설정
logging.getLogger("app.articles.service").setLevel(log_level)
logging.getLogger("app.articles.router").setLevel(log_level)
logging.getLogger("app.styleguides.service").setLevel(log_level)

# 로그 설정 확인
logger = logging.getLogger(__name__)
logger.info(f"Application starting with log level: {settings.LOG_LEVEL}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional one-time seed of style guides from JSON
    try:
        style_guides_file = os.getenv("STYLE_GUIDES_FILE")
        if style_guides_file:
            from sqlalchemy import select, func
            from .styleguides import service as styleguides_service
            from .styleguides.models import StyleGuide

            file_path = Path(style_guides_file)
            if not file_path.exists():
                logger.warning(f"STYLE_GUIDES_FILE not found: {file_path}")
            else:
                async with async_session_factory() as session:
                    # Seed only if table is empty
                    count = (await session.execute(select(func.count(StyleGuide.id)))).scalar_one()
                    if count == 0:
                        try:
                            payload = json.loads(file_path.read_text(encoding="utf-8"))
                            result = await styleguides_service.bulk_import_json(session, payload, mode="skip")
                            logger.info(
                                f"Seeded style guides from {file_path}: created={result.created}, skipped={result.skipped}, total={result.total}"
                            )
                        except Exception as e:
                            logger.exception(f"Failed to seed style guides from {file_path}: {e}")
                    else:
                        logger.info("Style guides table is not empty; skipping seed on startup.")
    except Exception as e:
        logger.exception(f"Startup seed error: {e}")

    yield
    await engine.dispose()

os.environ["TZ"] = "Asia/Seoul"
time.tzset()

app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware for admin authentication
# app.add_middleware(SessionMiddleware, secret_key="admin-session-secret")

# 라우터 등록
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(styleguides_router)
app.include_router(articles_router)

# 간단한 헬스 체크 엔드포인트 (프로덕션 헬스체크 용도)
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
