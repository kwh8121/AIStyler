"""
Gemini API 직접 연동 교정 모듈
OpenAI API 경로와 동일한 SSE 페이로드 형식을 유지
"""

import json
import time
import logging
import pysbd
from typing import AsyncGenerator, List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...styleguides import service as style_guide_service
from ...styleguides.models import StyleCategory
from ..models import ArticleCategory
from .translation import translate_text
from .prompt_builder import (
    generate_openai_style_analysis_prompt,
    generate_openai_correction_prompt,
)

# 기존 덤프 유틸 재사용
from .openai_correction import dump_prompt

logger = logging.getLogger(__name__)


async def analyze_style_violations_gemini(
    text: str,
    category: ArticleCategory,
    style_guides: List,
    db: AsyncSession,
) -> Dict:
    """Gemini API를 사용하여 스타일가이드 위반 분석(JSON 구조화 응답)."""
    try:
        from google import genai

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        analysis_prompt = generate_openai_style_analysis_prompt(style_guides, category.value)
        dump_prompt(
            "gemini_analysis",
            analysis_prompt,
            {
                "category": category.value,
                "style_guide_count": len(style_guides),
                "text_length": len(text),
            },
        )

        logger.info("🔍 Gemini analyzing style violations as JSON")

        # 간단 JSON 스키마(지원되는 OpenAPI subset)
        schema = {
            "type": "object",
            "properties": {
                "violations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_id": {"type": "string"},
                            "sentence_index": {"type": "integer"},
                        },
                        "required": ["rule_id", "sentence_index"],
                    },
                },
                "total_violations": {"type": "integer"},
            },
            "required": ["violations", "total_violations"],
        }

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL or "gemini-2.5-flash",
            contents=[analysis_prompt, text],
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )

        result_text = response.text or "{}"
        try:
            parsed = json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning("Gemini analysis JSON parse failed; using empty violations")
            parsed = {"violations": [], "total_violations": 0}

        violations = parsed.get("violations", [])
        applicable_rules = list(set(v.get("rule_id", "") for v in violations if v.get("rule_id")))

        # DB style_ids 매핑
        style_ids = []
        for guide in style_guides:
            for rule_id in applicable_rules:
                if f"SG{guide.number:03d}" in rule_id:
                    style_ids.append(guide.id)
                    break

        return {
            "violations": violations,
            "total_violations": len(violations),
            "applicable_rules": applicable_rules,
            "style_ids": style_ids,
        }

    except Exception as e:
        logger.error(f"Gemini style analysis failed: {e}")
        return {
            "violations": [],
            "total_violations": 0,
            "applicable_rules": [],
            "style_ids": [],
        }


async def call_gemini_correction_stream(
    prompt: Optional[str],
    text: str,
    category: ArticleCategory,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    Gemini API만을 사용한 교정 스트리밍 파이프라인.
    - DeepL 번역(자동 감지)
    - Gemini JSON 분석
    - Gemini 스트리밍 교정
    - OpenAI SSE 포맷과 동일한 이벤트를 생성
    """
    start_time = time.time()
    try:
        from google import genai

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Step 1. 번역
        yield json.dumps({"status": "translating", "message": "번역중..."})
        t0 = time.time()
        before_en, source_lang, target_lang = await translate_text(text, source_lang=None, target_lang="EN-US")
        t_translation = time.time() - t0
        yield json.dumps({"status": "translation_complete", "message": "번역 완료", "elapsed": round(t_translation, 3)})

        # Step 2. 스타일가이드 조회
        category_map = {
            ArticleCategory.TITLE: StyleCategory.TITLE,
            ArticleCategory.SEO: StyleCategory.TITLE,
            ArticleCategory.BODY: StyleCategory.BODY,
            ArticleCategory.CAPTION: StyleCategory.CAPTION,
        }
        style_category = category_map.get(category, StyleCategory.BODY)
        style_guides = await style_guide_service.list_styleguides(db, category=style_category, limit=100)

        # Step 3. 분석(JSON)
        yield json.dumps({"status": "applying_style", "message": "스타일 가이드 적용중..."})
        t1 = time.time()
        analysis = await analyze_style_violations_gemini(before_en, category, style_guides, db)
        t_analysis = time.time() - t1

        violations = analysis.get("violations", [])
        applicable_rules = analysis.get("applicable_rules", [])
        style_ids = analysis.get("style_ids", [])

        yield json.dumps({
            "type": "analysis",
            "data": {
                "applicable_rules": applicable_rules,
                "style_guide_violations": [
                    {"id": rid, "description": f"Style guide violation: {rid}"}
                    for rid in applicable_rules
                ],
                "style_ids": style_ids,
                "violations_count": len(violations),
            },
        })
        yield json.dumps({"status": "analysis_complete", "message": "스타일 가이드 분석 완료"})

        # Step 4. 교정(스트리밍)
        full_corrected_text = before_en
        if violations or (prompt and prompt.strip()):
            correction_prompt = generate_openai_correction_prompt(before_en, violations, style_guides, prompt)
            dump_prompt(
                "gemini_correction",
                correction_prompt,
                {
                    "category": category.value,
                    "violations": len(violations),
                    "style_ids": style_ids,
                    "text_length": len(before_en),
                },
            )

            t2 = time.time()
            collected: List[str] = []

            stream = client.models.generate_content_stream(
                model=settings.GEMINI_MODEL or "gemini-2.5-flash",
                contents=[correction_prompt],
                config={
                    # 필요 시 온도/토큰 등 세부값 노출 가능
                },
            )

            for chunk in stream:
                delta_text = getattr(chunk, "text", None)
                if not delta_text:
                    continue
                collected.append(delta_text)
                yield json.dumps({
                    "type": "delta",
                    "data": {"choices": [{"delta": {"content": delta_text}}]},
                })

            full_corrected_text = "".join(collected)
            t_correction = time.time() - t2
        else:
            # 위반 없음: 원문을 그대로 스트리밍
            for ch in before_en:
                yield json.dumps({
                    "type": "delta",
                    "data": {"choices": [{"delta": {"content": ch}}]},
                })
            t_correction = 0.0

        # Step 5. 문장별 교정 정보
        seg = pysbd.Segmenter(language="en", clean=False)
        original_sentences = seg.segment(before_en)
        corrected_sentences = seg.segment(full_corrected_text)

        sentence_corrections = {}
        for idx in range(min(len(original_sentences), len(corrected_sentences))):
            sentence_violations = [
                v["rule_id"] for v in violations if v.get("sentence_index") == idx
            ]
            sentence_corrections[idx] = {
                "original": original_sentences[idx] if idx < len(original_sentences) else "",
                "corrected": corrected_sentences[idx] if idx < len(corrected_sentences) else "",
                "violations": sentence_violations,
            }

        yield json.dumps({
            "type": "sentence_corrections",
            "data": {
                "sentence_corrections": sentence_corrections,
                "total_sentences": len(original_sentences),
                "corrected_sentences": len(corrected_sentences),
                "full_text": full_corrected_text,
            },
        })
        yield json.dumps({"status": "sentence_parsing_complete", "message": "문장별 교정 파싱 완료"})

        # Step 6. 최종 요약(번역 정보 포함)
        total_time = time.time() - start_time
        final_analysis = {
            "applicable_rules": applicable_rules,
            "style_ids": style_ids,
            "sentence_corrections": sentence_corrections,
            "full_text": full_corrected_text,
            "translation": {
                "before_text": before_en,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
            "processing_time": {
                "translation": t_translation,
                "analysis": t_analysis,
                "correction": t_correction,
                "total": total_time,
            },
        }
        yield json.dumps({"type": "final_analysis", "data": final_analysis})
        yield json.dumps({"status": "complete", "message": "교정 완료"})

    except Exception as e:
        logger.error(f"Gemini correction stream error: {e}")
        yield json.dumps({"type": "error", "data": {"message": f"처리 중 오류가 발생했습니다: {str(e)}"}})
        raise
