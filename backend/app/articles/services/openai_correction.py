# backend/app/articles/services/openai_correction.py
"""
OpenAI API 직접 연동 교정 모듈
AI 서버 없이 OpenAI API만으로 스타일가이드 분석 및 교정 수행
"""

import json
import time
import logging
import pysbd
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...styleguides import service as style_guide_service
from ...styleguides.models import StyleCategory
from ..models import ArticleCategory, OperationType
from .translation import translate_text
from .prompt_builder import (
    generate_openai_style_analysis_prompt,
    generate_openai_correction_prompt,
)
import re

logger = logging.getLogger(__name__)


def get_date_context() -> str:
    """Rule 4, 5, 10, 23을 위한 현재 날짜 및 날짜 매핑 테이블 생성"""
    now = datetime.now()
    
    # AP Style Month Abbreviations (Rule 10: March-July are full names)
    months_abbr = {
        1: "Jan.", 2: "Feb.", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "Aug.", 9: "Sept.", 10: "Oct.", 11: "Nov.", 12: "Dec."
    }
    
    today_str = f"{months_abbr[now.month]} {now.day}, {now.year} ({now.strftime('%A')})"
    
    lines = []
    lines.append(f"REFERENCE DATES INFORMATION (Today is {today_str})")
    lines.append("="*40)
    lines.append("[DATE REPLACEMENT TABLE]")
    lines.append("Instructions: If you find these dates in the text, replace them EXACTLY as mapped below.")
    lines.append("-" * 30)

    # ±7일 범위 생성 (과거 7일 ~ 미래 3일)
    dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for offset in range(-7, 4):
        d = now + timedelta(days=offset)
        mon_name = months_abbr[d.month]
        full_date_key = f"{mon_name} {d.day}"
        weekday = dows[d.weekday()]
        
        # Rule 4 Logic: 1-6 days ago -> weekday, 7+ days ago -> Month Day
        if -6 <= offset <= -1:
            replacement = weekday # 1-6 days ago
            status = "Within 7 days (Use Weekday)"
        elif offset == 0:
            replacement = f"{weekday}"
            status = "Today"
        elif 1 <= offset <= 6:
             replacement = weekday
             status = "Upcoming (Use Weekday)"
        else:
            replacement = full_date_key
            status = "Exact Date (Use Month Day)"
            
        lines.append(f" * {full_date_key} -> Use: \"{replacement}\" ({status})")

    lines.append("-" * 30)
    lines.append("\n[STRICT FORMATTING RULES]")
    lines.append(f"1. Rule 23 (Year): Omit the year if it is current ({now.year}). (e.g., 'Nov. 15, {now.year}' -> 'Nov. 15')")
    lines.append("2. Rule 10 (Months): NEVER abbreviate March, April, May, June, July. ALWAYS abbreviate others.")
    lines.append("3. Rule 5 (Ranges):")
    lines.append("   - Same month: 'Jan. 15-20'")
    lines.append("   - Different months: 'Jan. 28-Feb. 5' (Include month for both if different)")
    lines.append("4. Chain of Thought: Identify the absolute date first, check the table above, then apply Rule 23/10/5.")
    
    return "\n".join(lines)


def dump_prompt(kind: str, prompt_text: str, metadata: Optional[Dict] = None) -> None:
    """Persist OpenAI prompts to disk when debugging is enabled."""
    if not settings.OPENAI_DUMP_PROMPTS:
        return

    try:
        base_dir = Path(settings.OPENAI_PROMPT_DUMP_DIR or "logs/openai_prompts")
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_{kind}_{uuid.uuid4().hex[:8]}.txt"
        file_path = base_dir / filename

        with file_path.open("w", encoding="utf-8") as handle:
            if metadata:
                handle.write(json.dumps(metadata, ensure_ascii=False))
                handle.write("\n\n")
            handle.write(prompt_text)

        logger.debug("Saved OpenAI prompt dump: %s", file_path)
    except Exception as exc:
        logger.debug("Failed to dump OpenAI prompt (%s): %s", kind, exc)


def map_category_to_json(category: ArticleCategory) -> str:
    """ArticleCategory enum을 JSON 카테고리 문자열로 변환"""
    mapping = {
        ArticleCategory.SEO: "headlines",
        ArticleCategory.TITLE: "headlines",
        ArticleCategory.BODY: "articles",
        ArticleCategory.CAPTION: "captions",
    }
    return mapping.get(category, "articles")


async def analyze_style_violations_openai(
    text: str,
    category: ArticleCategory,
    style_guides: List,
    db: AsyncSession
) -> Dict:
    """
    OpenAI API를 사용하여 스타일가이드 위반 분석

    Returns:
        {
            "violations": [...],
            "total_violations": n,
            "applicable_rules": ["articles_SG001", ...],
            "style_ids": [1, 2, 3, ...]
        }
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # 날짜 컨텍스트 생성
        date_context = get_date_context()

        # 분석용 프롬프트 생성
        analysis_prompt = generate_openai_style_analysis_prompt(style_guides, category.value, date_context=date_context)
        dump_prompt(
            "analysis",
            analysis_prompt,
            {
                "category": category.value,
                "style_guide_count": len(style_guides),
                "text_length": len(text),
            },
        )

        logger.info(f"🔍 Analyzing style violations with OpenAI for {len(text)} chars")

        # OpenAI API 호출 (JSON 모드)
        analysis_model = settings.OPENAI_MODEL or "gpt-4o-mini"

        response = await client.responses.create(
            model=analysis_model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": analysis_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            ],
        )

        # 응답 파싱
        result_text = ""
        for item in response.output or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    result_text += content.text
        logger.debug(f"OpenAI analysis response: {result_text[:500]}...")

        try:
            analysis_result = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            analysis_result = {"violations": [], "total_violations": 0}

        # 적용 가능한 규칙 ID 추출
        violations = analysis_result.get("violations", [])
        applicable_rules = list(set(v.get("rule_id", "") for v in violations if v.get("rule_id")))

        # DB에서 style_ids 매핑
        style_ids = []
        for guide in style_guides:
            for rule_id in applicable_rules:
                if f"SG{guide.number:03d}" in rule_id:
                    style_ids.append(guide.id)
                    break

        logger.info(f"✅ Found {len(violations)} violations across {len(applicable_rules)} rules")

        return {
            "violations": violations,
            "total_violations": len(violations),
            "applicable_rules": applicable_rules,
            "style_ids": style_ids
        }

    except Exception as e:
        logger.error(f"OpenAI style analysis failed: {e}")
        return {
            "violations": [],
            "total_violations": 0,
            "applicable_rules": [],
            "style_ids": []
        }


async def call_openai_correction_stream(
    prompt: str,
    text: str,
    category: ArticleCategory,
    db: AsyncSession
) -> AsyncGenerator[str, None]:
    """
    OpenAI API만을 사용한 교정 스트리밍
    1. OpenAI로 번역
    2. OpenAI로 스타일가이드 분석
    3. OpenAI로 교정 스트리밍
    """

    start_time = time.time()
    logger.info(f"🚀 Starting OpenAI-only correction stream for category={category}")

    try:
        # Step 1: 번역 (환경 설정에 따른 제공자 사용)
        logger.info("Step 1: Translating text...")
        yield json.dumps({"status": "translating", "message": "번역중..."})

        translation_start = time.time()
        # 자동 감지로 소스 언어를 판별하여 불필요한 재번역/오번역을 방지
        before_en, source_lang, target_lang = await translate_text(
            text,
            source_lang=None,
            target_lang="EN-US"
        )
        translation_time = time.time() - translation_start
        logger.info(f"Translation completed in {translation_time:.3f}s")

        # 처리 시간 포함하여 전송
        yield json.dumps({"status": "translation_complete", "message": "번역 완료", "elapsed": round(translation_time, 3)})

        # Step 2: DB에서 카테고리별 모든 스타일가이드 로드
        logger.info(f"Step 2: Loading style guides for category {category.value}")

        # ArticleCategory를 StyleCategory enum으로 변환 (SEO/Translator 포함)
        category_map = {
            ArticleCategory.TITLE: StyleCategory.TITLE,
            ArticleCategory.SEO: StyleCategory.TITLE,
            ArticleCategory.BODY: StyleCategory.BODY,
            ArticleCategory.CAPTION: StyleCategory.CAPTION,
        }
        style_category = category_map.get(category, StyleCategory.BODY)

        # 해당 카테고리의 모든 스타일가이드 조회
        style_guides = await style_guide_service.list_styleguides(
            db,
            category=style_category,
            limit=100
        )

        if not style_guides:
            logger.warning(f"No style guides found for category {style_category}")
            # 스타일가이드가 없으면 원본 텍스트 그대로 반환
            for char in before_en:
                yield json.dumps({"choices": [{"delta": {"content": char}}]})
            return

        logger.info(f"Loaded {len(style_guides)} style guides for {style_category}")

        # Step 3: OpenAI로 스타일가이드 위반 분석
        logger.info("Step 3: Analyzing style violations with OpenAI...")
        yield json.dumps({"status": "applying_style", "message": "스타일 가이드 적용중..."})

        analysis_start = time.time()
        analysis_result = await analyze_style_violations_openai(
            before_en,
            category,
            style_guides,
            db
        )
        analysis_time = time.time() - analysis_start
        logger.info(f"Analysis completed in {analysis_time:.3f}s")

        # 분석 결과 전송
        violations = analysis_result.get("violations", [])
        applicable_rules = analysis_result.get("applicable_rules", [])
        style_ids = analysis_result.get("style_ids", [])

        yield json.dumps({
            "type": "analysis",
            "data": {
                "applicable_rules": applicable_rules,
                "style_guide_violations": [
                    {
                        "id": rule_id,
                        "description": f"Style guide violation: {rule_id}"
                    }
                    for rule_id in applicable_rules
                ],
                "style_ids": style_ids,
                "violations_count": len(violations)
            }
        })

        yield json.dumps({"status": "analysis_complete", "message": "스타일 가이드 분석 완료"})

        # Step 4: OpenAI로 교정 수행 (스트리밍)
        logger.info("Step 4: Correcting text with OpenAI (streaming)...")

        correction_time = 0  # 교정 시간 초기화

        # 위반 사항이 없고 추가 지침도 없으면 원문 반환
        if not violations and not prompt:
            logger.info("No violations and no additional instructions, returning original text")
            for char in before_en:
                yield json.dumps({
                    "type": "delta",
                    "data": {"choices": [{"delta": {"content": char}}]}
                })
            full_corrected_text = before_en
        else:
            # 위반 사항이 있거나 추가 지침이 있으면 교정 수행
            if violations:
                logger.info(f"Found {len(violations)} violations, performing correction...")
            if prompt:
                logger.info(f"Additional instructions provided: {prompt[:100]}...")

            # 날짜 컨텍스트 생성
            date_context = get_date_context()

            correction_prompt = generate_openai_correction_prompt(
                before_en,
                violations,
                style_guides,
                prompt,
                date_context=date_context
            )

            dump_prompt(
                "correction",
                correction_prompt,
                {
                    "category": category.value,
                    "violations": len(violations),
                    "style_ids": style_ids,
                    "text_length": len(before_en),
                },
            )

            correction_start = time.time()
            collected_chunks: list[str] = []

            correction_model = settings.OPENAI_MODEL or "gpt-4o-mini"

            # 추가 지침을 시스템 레벨에도 반영해 우선순위 강화
            system_text = "You are a professional editor.\nFollow the provided rule examples strictly when rewriting; mirror the '✓ Correct' pattern and prefer minimal edits."
            if prompt and str(prompt).strip():
                system_text += f"\nADDITIONAL INSTRUCTION (apply strictly): {str(prompt).strip()}"

            request_params = {
                "model": correction_model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_text}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": correction_prompt}],
                    },
                ],
                "stream": True,
                "temperature": 0.1
            }
            if settings.OPENAI_REASONING_EFFORT:
                request_params["reasoning"] = {
                    "effort": settings.OPENAI_REASONING_EFFORT
                }

            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            stream = await client.responses.create(**request_params)

            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "") or ""
                    if delta_text:
                        collected_chunks.append(delta_text)
                        yield json.dumps({
                            "type": "delta",
                            "data": {"choices": [{"delta": {"content": delta_text}}]}
                        })
                elif event_type in {"response.completed", "response.done"}:
                    break

            correction_time = time.time() - correction_start
            full_corrected_text = "".join(collected_chunks)
            # Courtesy of ... dedupe and ". /" normalization for captions (safety net)
            try:
                if category == ArticleCategory.CAPTION and full_corrected_text:
                    full_corrected_text = re.sub(r'(Courtesy of\s+[^\.\n]+?)(?:\.?\s+\1\.?)+$', r'\1', full_corrected_text, flags=re.IGNORECASE)
                    full_corrected_text = re.sub(r'\.(\s*)/(\s+)(Courtesy of|Yonhap|AP|Reuters|AFP|Getty Images|EPA|Bloomberg|Korea Times)\b', r' \2\3', full_corrected_text, flags=re.IGNORECASE)
            except Exception:
                pass
            logger.info(
                f"Correction completed in {correction_time:.3f}s, output: {len(full_corrected_text)} chars"
            )

        # Step 5: 문장별 교정 정보 생성 (선택적)
        logger.info("Step 5: Generating sentence-level corrections...")

        # 원본과 교정된 텍스트를 문장으로 분리
        seg = pysbd.Segmenter(language="en", clean=False)
        original_sentences = seg.segment(before_en)
        # 교정이 수행되었으면 교정된 텍스트, 아니면 원본
        corrected_sentences = seg.segment(full_corrected_text)

        # 문장별 교정 정보 매핑
        sentence_corrections = {}
        for idx in range(min(len(original_sentences), len(corrected_sentences))):
            # 해당 문장에 대한 위반사항 찾기
            sentence_violations = [
                v["rule_id"] for v in violations
                if v.get("sentence_index") == idx
            ]

            sentence_corrections[idx] = {
                "original": original_sentences[idx] if idx < len(original_sentences) else "",
                "corrected": corrected_sentences[idx] if idx < len(corrected_sentences) else "",
                "violations": sentence_violations
            }

        # 문장별 교정 정보 전송
        yield json.dumps({
            "type": "sentence_corrections",
            "data": {
                "sentence_corrections": sentence_corrections,
                "total_sentences": len(original_sentences),
                "corrected_sentences": len(corrected_sentences),
                "full_text": full_corrected_text
            }
        })

        yield json.dumps({"status": "sentence_parsing_complete", "message": "문장별 교정 파싱 완료"})

        # Step 6: 최종 분석 결과 (DB 저장용)
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.3f}s")

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
                "translation": translation_time,
                "analysis": analysis_time,
                "correction": correction_time if (violations or prompt) else 0,
                "total": total_time
            }
        }

        yield json.dumps({"type": "final_analysis", "data": final_analysis})

        # 완료 메시지
        yield json.dumps({"status": "complete", "message": "교정 완료"})

    except Exception as e:
        logger.error(f"OpenAI correction stream error: {e}")
        yield json.dumps({
            "type": "error",
            "data": {"message": f"처리 중 오류가 발생했습니다: {str(e)}"}
        })
        raise
