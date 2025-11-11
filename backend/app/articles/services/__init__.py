# backend/app/articles/services/__init__.py
"""
Articles 서비스 모듈 패키지

각 모듈별 책임:
- translation.py: DeepL 번역 관련 기능
- ai_correction.py: AI 서버 연동 교정 기능
- openai_correction.py: OpenAI API 직접 연동 교정 기능
- prompt_builder.py: 프롬프트 생성 및 관리
- history.py: 교정/번역 이력 DB 관리
- seo.py: SEO 타이틀 생성 기능
"""

# 각 모듈에서 필요한 함수들을 import하여 재export
# 이렇게 하면 기존 코드에서 import 경로를 크게 변경하지 않아도 됨