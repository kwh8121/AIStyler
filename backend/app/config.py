import json
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")

    # FastAPI 애플리케이션 설정
    ENVIRONMENT: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"  # 로그 레벨 설정

    # (Optional) 외부에서 주입되는 포트 설정 값들 – 실제 사용 여부와 관계없이 Validation 에러 방지
    FRONTEND_PORT: int | None = None
    BACKEND_PORT: int | None = None
    DB_PORT: int | None = None

    # 데이터베이스 설정
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "styler_db"
    POSTGRES_SSLMODE: str = "disable"
    DATABASE_URL: str = "postgresql+asyncpg://user:postgres@db:5432/styler_db"

    # JWT 인증 설정
    JWT_SECRET_KEY: str = "your-secret-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS 설정
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:80",
        "http://localhost",
    ]

    # DeepL API 설정
    DEEPL_API_KEY: str | None = None
    DEEPL_API_FREE: bool = True  # True: 무료 API, False: Pro API
    # 번역 엔진 선택: "openai" | "openai-2stage" | "deepl" | "auto"
    TRANSLATION_PROVIDER: str = "openai"

    # vLLM 서버 설정
    VLLM_URL: str = "http://host.docker.internal:8888/v1/chat/completions"
    VLLM_MODEL_NAME: str = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

    # OpenAI API 설정
    OPENAI_API_KEY: str | None = None
    OPENAI_API_SEO_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-5-mini"
    USE_OPENAI: bool = False  # OpenAI 사용 여부 플래그
    OPENAI_REASONING_EFFORT: str = "low"
    OPENAI_DUMP_PROMPTS: bool = False
    OPENAI_PROMPT_DUMP_DIR: str = "logs/openai_prompts"
    # GPT-5 v2 toggle + logging
    GPT5_V2_ENABLED: bool = False
    GPT5_V2_DUMP_PROMPTS: bool = False
    GPT5_V2_LOG_DIR: str = "logs/gpt5v2"
    # v2 Expert 분리 여부 (True=3-experts, False=단일 호출)
    GPT5_V2_USE_EXPERT_SPLIT: bool = False

    # Gemini API 설정
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    USE_GEMINI: bool = False  # Gemini 사용 여부 플래그

    # AI Server 설정
    AI_SERVER_URL: str = "http://localhost:8080"
    USE_AI_SERVER: bool = False  # AI Server 사용 여부 플래그

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _normalise_cors(cls, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        raise ValueError("CORS_ORIGINS must be a string or list of strings")


settings = Config()
