# backend/app/articles/services/translation.py
"""
번역 서비스 모듈
OpenAI API를 사용한 텍스트 번역 기능 제공

기존 DeepL 연동 코드는 요청에 따라 주석 처리하고,
동일한 인터페이스(translate_text, translate_to_en)를 OpenAI 기반으로 제공합니다.
"""

from openai import AsyncOpenAI
import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional, Tuple
import os

from ...config import settings

logger = logging.getLogger(__name__)

# --- Prompts ---

PROMPT_HEADLINE = """Role: You are a veteran headline writer. Convert the Korean headline to an English AP-style headline.

Core rules:
- Preserve the original meaning; do not add hype or new facts.
- Headline case: Use sentence case. No period at the end.
- Use present tense, strong verbs, and AP numerals (figures for 10+, percentages, ages, dates).
- Avoid starting with a date. Use AP date forms if a date must appear.
- Keep it concise (~6–12 words or under ~65 characters when possible).
- Use commonly accepted English names for people/places; otherwise, use standard romanization.

Output format:
- One AP-style English headline only (no subtitle, no brackets)."""

PROMPT_BODY = """Role: You are a veteran reporter (20 years). Write an English news article in AP style strictly from the Korean text provided.

Core rules:
- Do not add or infer facts; do not omit material facts present in the source. Preserve meaning and nuance.
- No headline, no subheads, no quotes unless they appear in the source.
- Keep the lead concise (1–2 sentences) and use short paragraphs (1–3 sentences each).
- Use active voice, neutral tone, and plain language.
- Dates & time: Follow AP style. Do not begin sentences with a date. Within 7 days use the day of week; otherwise use “Mon., Tue., …” month abbreviations where applicable (e.g., Jan., Feb., Aug., Sept., Oct., Nov., Dec.) and Arabic numerals.
- Numbers: Follow AP numerals rules (use figures for 10 and above; always use figures for ages, dates, percentages, dimensions, money).
- Titles: Capitalize formal titles before names; lowercase after names. No courtesy titles (Mr., Ms.).
- Names & places: Use widely accepted English names when known; otherwise use official romanization. If a name may be unclear, add the Korean in parentheses on first mention only.
- Units & currency: Use KRW as “won” with figures; include commas (e.g., 12,300 won). Convert only if the source gives a conversion.
- No dateline unless explicitly provided in the source.
- Keep proper nouns as in the source unless an established English style exists.

Output format:
- Pure article body only. No headline, no bullet points, no brackets.
- Maintain factual accuracy and AP style throughout."""

PROMPT_CAPTION = """Role: You are a wire-desk caption editor. Convert the Korean caption to an English photo caption in AP style.

Core rules:
- Present-tense description in the first sentence. Use a second sentence for background or timing if needed (past tense acceptable there).
- Identify people clearly on first reference (full name, role/title if given). No courtesy titles.
- “From left” placement: If listing people left-to-right, write “From left,” before the list. Do not place “from left” after the first name.
- Location and day-of-week/date follow AP style; do not begin sentences with a date. Use month abbreviations where applicable.
- Keep it tight and clear (ideally 1–2 sentences, max ~35 words unless necessary).
- Do not add or infer details beyond the source. Do not quote or invent speech.
- No credit lines, brackets, hashtags, or emojis.

Output format:
- Single AP-style caption only, no headline or bullets.
- If IDs are uncertain, keep only what is supported by the source text."""

# DeepL (선택적)
try:
    import deepl  # type: ignore
except Exception:
    deepl = None

_deepl_client: Optional["deepl.DeepLClient"] = None
async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

def get_deepl_client() -> Optional["deepl.DeepLClient"]:
    """DeepL 클라이언트를 가져오거나 생성 (필요 시)"""
    global _deepl_client
    if deepl is None:
        return None
    if _deepl_client is None and settings.DEEPL_API_KEY:
        try:
            server_url = "https://api-free.deepl.com" if settings.DEEPL_API_FREE else None
            _deepl_client = deepl.DeepLClient(settings.DEEPL_API_KEY, server_url=server_url)  # type: ignore
            logger.info(f"DeepL client initialized (Free: {settings.DEEPL_API_FREE})")
        except Exception as e:
            logger.error(f"Failed to initialize DeepL client: {e}")
            return None
    return _deepl_client

# 선택된(또는 자동 결정된) 번역 엔진을 기록하기 위한 최근 값
_last_provider: str = "unknown"

def get_translation_provider() -> str:
    """최근 사용된 번역 엔진 또는 설정 의도값을 반환 (로그/표시용)"""
    explicit = (settings.TRANSLATION_PROVIDER or "").strip().lower() or "openai"
    # 최근 호출에서 실제 사용된 것이 있으면 그 값을 우선
    return _last_provider or explicit

# Optional translation style guidelines loader
_guidelines_text: Optional[str] = None

def _load_translation_guidelines_text() -> Optional[str]:
    """Load compact translation style rules from translation_style.json if present.

    Returns a compact bullet list string suitable for injecting into the system prompt.
    """
    global _guidelines_text
    if _guidelines_text is not None:
        return _guidelines_text
    try:
        style_path = Path(__file__).parent / "translation_style.json"
        if not style_path.exists():
            logger.debug("translation_style.json not found; skipping style rules injection")
            _guidelines_text = None
            return _guidelines_text
        with style_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.get("translation_guidelines", [])
        lines: list[str] = []
        for it in items:
            instr = (it.get("instruction") or "").strip()
            if not instr:
                continue
            lines.append(f"- {instr}")
        compact = "\n".join(lines[:50])
        _guidelines_text = compact if compact else None
        if _guidelines_text:
            logger.info(f"Loaded translation style guidelines: {len(lines)} rules (using {min(len(lines), 50)})")
        return _guidelines_text
    except Exception as exc:
        logger.debug(f"Failed to load translation_style.json: {exc}")
        _guidelines_text = None
        return _guidelines_text

def _env_float(name: str, default: float | None) -> float | None:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y")

def _dump_translation_prompt(kind: str, prompt_text: str, metadata: Optional[dict] = None) -> None:
    """Persist translation prompts to disk when debugging is enabled.

    Files are written under settings.OPENAI_PROMPT_DUMP_DIR (default logs/openai_prompts).
    """
    if not getattr(settings, "OPENAI_DUMP_PROMPTS", False):
        return
    try:
        base_dir = Path(getattr(settings, "OPENAI_PROMPT_DUMP_DIR", "logs/openai_prompts") or "logs/openai_prompts")
        base_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = f"{ts}_translation_{kind}_{uuid.uuid4().hex[:8]}.txt"
        path = base_dir / fname
        with path.open("w", encoding="utf-8") as f:
            if metadata:
                try:
                    f.write(json.dumps(metadata, ensure_ascii=False))
                    f.write("\n\n")
                except Exception:
                    pass
            f.write(prompt_text or "")
        logger.debug("Saved translation prompt dump: %s", path)
    except Exception as exc:
        logger.debug("Failed to dump translation prompt (%s): %s", kind, exc)

def _normalize_target_lang(target_lang: str | None) -> str:
    """타겟 언어 표준화 (DeepL 호환 표기를 내부적으로 정리)"""
    if not target_lang:
        return "EN-US"
    tl = target_lang.strip().upper()
    if tl in {"EN", "EN-GB", "EN-US"}:
        # 기본은 EN-US로 통일
        return "EN-US"
    return tl

def _map_lang_name_to_code(lang: str) -> str:
    """OpenAI가 반환한 언어명을 간단한 코드로 매핑"""
    if not lang:
        return "UNKNOWN"
    name = lang.strip().lower()
    mapping = {
        "korean": "KO",
        "ko": "KO",
        "english": "EN",
        "en": "EN",
        "japanese": "JA",
        "ja": "JA",
        "chinese": "ZH",
        "zh": "ZH",
        "french": "FR",
        "fr": "FR",
        "german": "DE",
        "de": "DE",
        "spanish": "ES",
        "es": "ES",
    }
    return mapping.get(name, name.upper()[:8]) or "UNKNOWN"

def _contains_hangul(text: str) -> bool:
    if not text:
        return False
    # 한글 음절 + 자모 범위 포함 여부
    for ch in text:
        code = ord(ch)
        if (
            0xAC00 <= code <= 0xD7A3  # 한글 음절
            or 0x1100 <= code <= 0x11FF  # 한글 자모
            or 0x3130 <= code <= 0x318F  # 호환 자모
        ):
            return True
    return False

def _is_probably_english(text: str) -> bool:
    if not text:
        return True
    # 간단 휴리스틱: 라틴/숫자/공백/문장부호 비율이 매우 높고, 한글이 없으면 영어로 간주
    if _contains_hangul(text):
        return False
    total = len(text)
    allowed = 0
    alpha = 0
    for ch in text:
        o = ord(ch)
        if (
            0x20 <= o <= 0x7E  # ASCII 영역
            or ch in "\n\r\t"
        ):
            allowed += 1
            if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                alpha += 1
    # 영문자 최소 1자 이상이고, 전체의 90% 이상이 ASCII면 영어로 판단
    return alpha > 0 and allowed / max(total, 1) >= 0.9

async def _responses_create_async(*, model: str, input_blocks: list, tools: list, tool_choice: str | None,
                                      temperature: float | None, top_p: float | None,
                                      include_reasoning: bool, reasoning_effort: str | None,
                                      text_verbosity: str | None):

        params: dict = {
            "model": model,
            "input": input_blocks,
            "tools": tools,
        }
        if tool_choice:
            params["tool_choice"] = tool_choice
        if temperature:
            params["temperature"] = temperature
        if top_p:
            params["top_p"] = top_p
        if include_reasoning and reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            params["text"] = {"verbosity": text_verbosity}

        try:
            return await async_client.responses.create(**params)
        except Exception as e:
            # Fallback: remove optional fields (reasoning/text) and retry once
            params.pop("reasoning", None)
            params.pop("text", None)
            try:
                return await async_client.responses.create(**params)
            except Exception:
                raise

async def _openai_translate(
    text: str,
    target_lang: str,
    category: Optional[str] = None,
) -> Tuple[str, str]:
    """OpenAI Responses API를 사용한 번역 + 언어 감지.

    Returns: (translated_text, detected_source_lang)
    """
    # 카테고리별 프롬프트 선택
    if category == "TITLE":
        system_text = PROMPT_HEADLINE
    elif category == "BODY":
        system_text = PROMPT_BODY
    elif category == "CAPTION":
        system_text = PROMPT_CAPTION
    else:
        # 기존 기본 프롬프트 (Fallback)
        system_text = """
You are a professional news translator and editor for The Korea Times.

Your mission:

Produce an English article that can be directly published for a global English-speaking audience.

The Korea Times explains news in and about Korea to international readers.

Target readers: educated general public worldwide (not only readers in Korea).

Tone: professional, neutral, AP-style clarity, internationally readable.

Style: concise, factual, journalistic prose.

Task:
You will receive one source article written in Korean or mixed Korean/English. Translate it into final, publication-ready English in one pass.

Follow these requirements:

Accuracy and factual integrity

Preserve all factual content exactly: numbers, dates, positions/titles, legal statements, policy claims, quotations, company names, institution names.

Do not invent or speculate. Do not add new facts.

You may add a minimal appositive clarification (a short explanatory phrase) only if it is required for a global reader to understand a Korea-specific reference (for example, identify a ministry as "the Ministry of XXX, Korea's [brief role]"). Keep such clarifications to one short phrase.

Natural, idiomatic English for global readers

You MAY paraphrase or adapt idioms, honorific language, culturally specific expressions, or indirect politeness so that the result sounds natural to a native English news reader.

You MUST keep the underlying meaning, tone, and stance of the source.

Do not use literal calques of Korean expressions if they would sound awkward in English. Use normal international newsroom English.

Register and voice

Write in clear international news English suitable for publication.

Avoid repetitive honorifics, formalities, and rhetorical flourishes common in Korean official statements.

Names, official titles, and institutions must remain consistent throughout the article.

Keep paragraphs readable and well-structured. This is not a sentence-by-sentence subtitle translation; it is a publishable article.

Self-check before output (do this silently)

Check for:
a) mistranslations or omissions,
b) added meaning that was not in the source,
c) unnatural or overly literal phrasing,
d) inconsistent terminology for the same entity.

Fix these issues in your final output.

This self-check is internal; do not describe it in the output.

Output format

Output ONLY the final English article as normal paragraphs.

Do NOT include the original text, notes, bullet points, explanations, glossaries, or commentary.

Do NOT include any section headers like "Translation:" or "Analysis:".

Deliver only clean publication-ready English prose.

Source article:
[PASTE SOURCE ARTICLE HERE]
"""
    
    # 스타일 가이드 주입 (기본 프롬프트 사용 시에만 적용하거나, 전체 적용 고려)
    # 여기서는 기존 로직 유지: guidelines가 있으면 추가
    guidelines = _load_translation_guidelines_text()
    if guidelines:
        system_text += "\n\nApply these translation style rules strictly:\n" + guidelines
    
    user_instruction = "Task:\nTranslate and rewrite the following Korean article into an English AP-style article, following all rules above."
    if category == "TITLE":
         user_instruction = "Task:\nConvert the Korean headline to an English AP-style headline."
    elif category == "CAPTION":
         user_instruction = "Task:\nTranslate and edit the following Korean caption into a concise, AP-style English caption, following all rules above."

    # Target language instruction (if generic)
    if not category:
         user_instruction = "Target language: EN-US"

    t0 = time.time()
    try:
        _dump_translation_prompt(
            "system",
            system_text,
            {
                "model": "gpt-5",
                "provider": "openai",
                "target_lang": target_lang,
                "guidelines_included": bool(guidelines),
                "guidelines_lines": len(guidelines.splitlines()) if guidelines else 0,
            },
        )
        _dump_translation_prompt(
            "input",
            f"INSTRUCTION:\n{user_instruction}\n\nTEXT:\n{text}",
            {"length_text": len(text or ""), "function": "emit_translation"},
        )
    except Exception:
        pass
    tools = [{
        "type": "function",
        "name": "emit_translation",
        "description": "Return translated English (US) text and detected source language code.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "translated_text": {
                    "type": "string",
                    "description": "Final translated text in US English (EN-US)."
                },
                "detected_source_lang": {
                    "type": "string",
                    "description": "Detected source language code.",
                    "enum": ["KO", "EN"]
                }
            },
            "required": ["translated_text", "detected_source_lang"],
            "additionalProperties": False
        }
    }]

    translate_model = os.getenv('TRANSLATE_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest'))
    tool_choice = os.getenv('TRANSLATE_TOOL_CHOICE', 'required')
    temp = _env_float('TRANSLATE_TEMPERATURE', None)
    top_p = _env_float('TRANSLATE_TOP_P', None)
    include_reasoning = _env_bool('TRANSLATE_INCLUDE_REASONING', False)
    reasoning_effort = _env_float('TRANSLATE_REASONING_EFFORT', None)
    text_verbosity = os.getenv('TRANSLATE_TEXT_VERBOSITY',  None)

    input=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_instruction},
        {"role": "user", "content": text},
    ],

    response = await _responses_create_async(
        model=translate_model,
        input_blocks=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_instruction},
            {"role": "user", "content": text},
        ],
        tools=tools,
        tool_choice=tool_choice,
        temperature=temp,
        top_p=top_p,
        include_reasoning=include_reasoning,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )

    elapsed = time.time() - t0
    logger.info("model and, reasoning: ", settings.OPENAI_MODEL, settings.OPENAI_REASONING_EFFORT)

    # 함수 호출 우선 파싱, 실패 시 텍스트 백업
    translated, detected = text, "UNKNOWN"
    func_found = False
    try:
        for item in (response.output or []):
            if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == "emit_translation":
                args_text = getattr(item, "arguments", "") or ""
                data = json.loads(args_text) if args_text else {}
                translated = data.get("translated_text") or translated
                detected = _map_lang_name_to_code(data.get("detected_source_lang"))
                func_found = True
                break
    except Exception:
        func_found = False

    if not func_found:
        raw_text = ""
        for item in response.output or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    raw_text += content.text
        translated = raw_text.strip() or translated
        detected = detected or "UNKNOWN"
        try:
            logger.warning("OpenAI function_call not returned; falling back to output_text parsing")
        except Exception:
            pass
    else:
        try:
            logger.info(
                f"OpenAI function_call 'emit_translation' used: detected={detected}, translated_len={len(translated)}"
            )
        except Exception:
            pass

    logger.info(
        f"OpenAI(FC) translation took {elapsed:.3f}s, detected={detected}, len={len(translated)}"
    )
    return translated, detected

async def _openai_translate_with_prompts(
    text: str,
    target_lang: str,
    system_text: str,
    user_instruction: str,
    dump_label: str = "A",
) -> Tuple[str, str]:
    """OpenAI 번역 호출 (임의의 시스템/유저 프롬프트 사용).

    Args:
        text: 입력 텍스트
        target_lang: 타겟 언어 코드 (예: EN-US)
        system_text: 시스템 프롬프트 텍스트
        user_instruction: 유저 프롬프트 텍스트
        dump_label: 프롬프트 덤프 파일 라벨 구분자 (A/B 등)

    Returns:
        (translated_text, detected_source_lang)
    """
    from openai import AsyncOpenAI

    # 스타일 가이드가 존재하면 시스템 프롬프트에 주입
    guidelines = _load_translation_guidelines_text()
    sys_text_final = system_text
    if guidelines:
        sys_text_final += "\n\nApply these translation style rules strictly:\n" + guidelines

    t0 = time.time()
    try:
        _dump_translation_prompt(
            f"system_{dump_label}",
            sys_text_final,
            {
                "model": "gpt-5",
                "provider": "openai",
                "target_lang": target_lang,
                "guidelines_included": bool(guidelines),
                "guidelines_lines": len(guidelines.splitlines()) if guidelines else 0,
            },
        )
        _dump_translation_prompt(
            f"input_{dump_label}",
            f"INSTRUCTION:\n{user_instruction}\n\nTEXT:\n{text}",
            {"length_text": len(text or ""), "function": "emit_translation"},
        )
    except Exception:
        pass

    tools = [{
        "type": "function",
        "name": "emit_translation",
        "description": "Return translated English (US) text and detected source language code.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "translated_text": {
                    "type": "string",
                    "description": "Final translated text in US English (EN-US)."
                },
                "detected_source_lang": {
                    "type": "string",
                    "description": "Detected source language code.",
                    "enum": ["KO", "EN"]
                }
            },
            "required": ["translated_text", "detected_source_lang"],
            "additionalProperties": False
        }
    }]

    translate_model = os.getenv('TRANSLATE_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest'))
    tool_choice = os.getenv('TRANSLATE_TOOL_CHOICE', 'required')
    temp = _env_float('TRANSLATE_TEMPERATURE', None)
    top_p = _env_float('TRANSLATE_TOP_P', None)
    include_reasoning = _env_bool('TRANSLATE_INCLUDE_REASONING', False)
    reasoning_effort = _env_float('TRANSLATE_REASONING_EFFORT', None)
    text_verbosity = os.getenv('TRANSLATE_TEXT_VERBOSITY',  None)

    input=[
        {"role": "system", "content": sys_text_final},
        {"role": "user", "content": user_instruction},
        {"role": "user", "content": text},
    ],

    response = await _responses_create_async(
        model=translate_model,
        input_blocks=[
            {"role": "system", "content": sys_text_final},
            {"role": "user", "content": user_instruction},
            {"role": "user", "content": text},
        ],
        tools=tools,
        tool_choice=tool_choice,
        temperature=temp,
        top_p=top_p,
        include_reasoning=include_reasoning,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )

    elapsed = time.time() - t0

    translated, detected = text, "UNKNOWN"
    func_found = False
    try:
        for item in (response.output or []):
            if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == "emit_translation":
                args_text = getattr(item, "arguments", "") or ""
                data = json.loads(args_text) if args_text else {}
                translated = data.get("translated_text") or translated
                detected = _map_lang_name_to_code(data.get("detected_source_lang"))
                func_found = True
                break
    except Exception:
        func_found = False

    if not func_found:
        raw_text = ""
        for item in response.output or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    raw_text += content.text
        translated = raw_text.strip() or translated
        detected = detected or "UNKNOWN"
        try:
            logger.warning("OpenAI function_call not returned; falling back to output_text parsing")
        except Exception:
            pass
    else:
        try:
            logger.info(
                f"OpenAI(FC,{dump_label}) detected={detected}, translated_len={len(translated)}"
            )
        except Exception:
            pass

    logger.info(
        f"OpenAI(FC,{dump_label}) translation took {elapsed:.3f}s, detected={detected}, len={len(translated)}"
    )
    return translated, detected

async def _deepl_translate(
    text: str,
    source_lang: Optional[str],
    target_lang: str,
) -> Tuple[str, str]:
    """DeepL API를 사용한 번역 + 언어 감지.

    Returns: (translated_text, detected_source_lang)
    """
    client = get_deepl_client()
    if not client:
        raise RuntimeError("DeepL client not available")

    t0 = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            formality="prefer_more",
            model_type="quality_optimized",
        )
    )
    elapsed = time.time() - t0
    translated = result.text
    detected = getattr(result, "detected_source_lang", None) or source_lang or "UNKNOWN"
    logger.info(
        f"DeepL translation took {elapsed:.3f}s, detected={detected}, len={len(translated)}"
    )
    return translated, str(detected)

async def translate_text(
    text: str,
    source_lang: Optional[str] = None,
    target_lang: str = "EN-US",
    category: Optional[str] = None,  # "TITLE", "BODY", "CAPTION"
) -> Tuple[str, str, str]:
    """범용 텍스트 번역 함수 (OpenAI)

    Args:
        text: 번역할 텍스트
        source_lang: 소스 언어 코드 (None이면 자동 감지; OpenAI 측에서 감지)
        target_lang: 타겟 언어 코드 (예: EN-US)
        category: 기사 컴포넌트 카테고리 (TITLE, BODY, CAPTION) - 전용 프롬프트 사용 시 필요

    Returns:
        (번역된 텍스트, 감지된 소스 언어, 타겟 언어) 튜플
    """
    # 빈 텍스트 체크
    if not text or not text.strip():
        return text, source_lang or "UNKNOWN", _normalize_target_lang(target_lang)

    normalized_target = _normalize_target_lang(target_lang)

    # 번역 필요 여부 빠른 판단: 타겟이 영어이고, 텍스트가 영어로 보이면 API 호출 생략
    if normalized_target in {"EN", "EN-US", "EN-GB"}:
        # 호출자가 EN으로 명시한 경우 바로 패스
        if (source_lang or "").upper().startswith("EN"):
            logger.info("Skipping translation: source_lang indicates English")
            return text, source_lang or "EN", normalized_target
        # 텍스트 기준 휴리스틱
        if _is_probably_english(text):
            logger.info("Skipping translation: text appears to be English")
            return text, "EN", normalized_target

    # 어떤 프로바이더를 사용할지 결정
    provider = (settings.TRANSLATION_PROVIDER or "").strip().lower() or "openai"
    use_openai = provider == "openai"
    use_deepl = provider == "deepl"
    use_openai_two_stage = provider == "openai-2stage"

    if provider == "auto":
        # 우선순위: OpenAI -> DeepL
        use_openai = bool(settings.OPENAI_API_KEY)
        use_deepl = bool(not use_openai and settings.DEEPL_API_KEY)

    global _last_provider

    # 먼저 2단계 OpenAI 프로바이더 처리
    if use_openai_two_stage:
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured; falling back to DeepL if available")
        else:
            try:
                translated, detected, _target = await translate_text_two_stage(
                    text,
                    source_lang=source_lang,
                    target_lang=normalized_target,
                )
                _last_provider = "openai-2stage"
                return translated, (source_lang or detected or "UNKNOWN"), normalized_target
            except Exception as e:
                logger.error(f"OpenAI(two-stage) translation error: {e}; trying single-stage or DeepL")

    if use_openai:
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured; falling back to DeepL if available")
        else:
            try:
                translated, detected = await _openai_translate(text, normalized_target, category=category)
                _last_provider = "openai"
                return translated, (source_lang or detected or "UNKNOWN"), normalized_target
            except Exception as e:
                logger.error(f"OpenAI translation error: {e}; trying DeepL if configured")

    if use_deepl or (provider in {"openai", "openai-2stage", "auto"}):
        if settings.DEEPL_API_KEY and deepl is not None:
            try:
                translated, detected = await _deepl_translate(text, source_lang, normalized_target)
                _last_provider = "deepl"
                return translated, (source_lang or detected or "UNKNOWN"), normalized_target
            except Exception as e:
                logger.error(f"DeepL translation error: {e}; returning original text")
        else:
            logger.debug("DeepL not configured or library missing")

    # 둘 다 사용 불가한 경우 원문 반환
    _last_provider = "none"
    return text, source_lang or "UNKNOWN", normalized_target

async def translate_to_en(text: str) -> Tuple[str, str, str]:
    """한국어 텍스트를 영어(미국)로 번역 (OpenAI)

    Returns: (translated_en, detected_source_lang, target_lang)
    """
    translated, detected_source, target_lang = await translate_text(
        text, source_lang="KO", target_lang="EN-US"
    )

    if not translated or not translated.strip():
        logger.warning(f"OpenAI returned empty result for: '{text[:50]}'")
        return text, detected_source, target_lang

    return translated, detected_source, target_lang

async def translate_title(
    text: str,
    style: Optional[str] = None,
) -> Tuple[str, str, str]:
    """단일 단계 제목 번역 (헤드라인 전용)

    - 한국어/영어 혼합 제목을 전 세계 독자를 위한 한 줄 영어 헤드라인으로 변환
    - 설명/문장형 출력 금지, 불필요한 관사/군더더기 제거
    - 반환: (headline_en, detected_source_lang, target_lang)
    """

    # system_text = "You are a veteran journalist with 20 years of experience.Based on the provided information, write an AP-style English headline.  However, do not start each sentence with the date. Also, when writing the headline, do not alter the meaning of the provided information or add any additional content.Note: Please output the results in English."

    system_text = PROMPT_HEADLINE

    tools = [{
        "type": "function",
        "name": "emit_headline",
        "description": "Return a single English news headline.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "One English news headline."}
            },
            "required": ["headline"],
            "additionalProperties": False
        }
    }]

    # Log mode and input meta
    try:
        logger.info(
            f"[headline] translate_title start: mode={style}, input_len={len(text or '')}"
        )
    except Exception:
        pass

    try:
        import time as _time
        _t0 = _time.time()

        input=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": text},
        ],

        translate_model = os.getenv('TRANSLATE_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest'))
        tool_choice = os.getenv('TRANSLATE_TOOL_CHOICE', 'required')
        temp = _env_float('TRANSLATE_TEMPERATURE', None)
        top_p = _env_float('TRANSLATE_TOP_P', None)
        include_reasoning = _env_bool('TRANSLATE_INCLUDE_REASONING', False)
        reasoning_effort = _env_float('TRANSLATE_REASONING_EFFORT', None)
        text_verbosity = os.getenv('TRANSLATE_TEXT_VERBOSITY',  None)
        
        resp = await _responses_create_async(
            model=translate_model,
            input_blocks=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": text},
            ],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temp,
            top_p=top_p,
            include_reasoning=include_reasoning,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        
        _elapsed = _time.time() - _t0
        try:
            logger.info(f"[headline] OpenAI responses.create ok in {_elapsed:.2f}s (mode={style})")
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Headline translation API error: {e}")
        # 폴백: 일반 번역으로 처리
        translated, detected = await _openai_translate(text, "EN-US")
        res = _cleanup_headline(
            translated,
            remove_final_period=(style == "headline"),
        )
        try:
            logger.info(f"[headline] fallback=_openai_translate used (mode={style}), out_len={len(res)}")
        except Exception:
            pass
        return res, ("KO" if _contains_hangul(text) else "UNKNOWN"), "EN-US"

    headline = ""
    try:
        for item in (resp.output or []):
            if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == "emit_headline":
                args_text = getattr(item, "arguments", "") or ""
                data = json.loads(args_text) if args_text else {}
                headline = data.get("headline") or ""
                break
        if not headline:
            # fallback to output_text
            for item in resp.output or []:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", "") == "output_text":
                        headline += content.text
    except Exception:
        pass

    headline = _cleanup_headline(
        headline or text,
        remove_final_period=(style == "headline"),
    )
    try:
        logger.info(f"[headline] translate_title done: mode={style}, out_len={len(headline)}")
    except Exception:
        pass
    detected = "KO" if _contains_hangul(text) else "UNKNOWN"
    return headline, detected, "EN-US"

def _cleanup_headline(
    s: str,
    remove_final_period: bool = False,
) -> str:
    """헤드라인 후처리: 한 줄, 길이/공백 정리. 옵션에 따라 마침표/제한 적용."""
    try:
        s = (s or "").replace("\n", " ").replace("\r", " ")
        s = " ".join(s.split())
        # 끝 마침표 제거(헤드라인 모드에서만)
        if remove_final_period and s.endswith("."):
            s = s[:-1]
        return s.strip()
    except Exception:
        return (s or "").strip()

async def translate_text_two_stage(
    text: str,
    source_lang: Optional[str] = None,
    target_lang: str = "EN-US",
    system_text_A: Optional[str] = None,
    user_instruction_A: Optional[str] = None,
    system_text_B: Optional[str] = None,
    user_instruction_B: Optional[str] = None,
) -> Tuple[str, str, str]:
    """OpenAI API를 두 번 호출하여 서로 다른 프롬프트(A, B)로 순차 적용하는 번역 함수.

    단계1(A): 원문을 타겟 언어로 번역
    단계2(B): 단계1 결과를 B 프롬프트로 재가공/정제

    Args:
        text: 번역할 원문 텍스트
        source_lang: 소스 언어 힌트 (없으면 자동 감지)
        target_lang: 타겟 언어 코드 (예: EN-US)
        system_text_A: 1단계 시스템 프롬프트 (기본값 내장)
        user_instruction_A: 1단계 사용자 프롬프트 (기본값 내장)
        system_text_B: 2단계 시스템 프롬프트 (기본값 내장)
        user_instruction_B: 2단계 사용자 프롬프트 (기본값 내장)

    Returns:
        (최종 번역문, 감지된 소스 언어, 타겟 언어)
    """
    if not text or not text.strip():
        return text, source_lang or "UNKNOWN", _normalize_target_lang(target_lang)

    normalized_target = _normalize_target_lang(target_lang)

    # 타겟이 영어이고 입력이 영어로 보이면 스킵
    if normalized_target in {"EN", "EN-US", "EN-GB"}:
        if (source_lang or "").upper().startswith("EN"):
            logger.info("Two-stage: Skipping translation (source_lang indicates English)")
            return text, source_lang or "EN", normalized_target
        if _is_probably_english(text):
            logger.info("Two-stage: Skipping translation (text appears to be English)")
            return text, "EN", normalized_target

    if not settings.OPENAI_API_KEY:
        logger.error("Two-stage: OPENAI_API_KEY not configured; returning original text")
        return text, source_lang or "UNKNOWN", normalized_target

    # 기본 A/B 프롬프트 설정 (필요 시 호출부에서 덮어쓰기)
    sys_A = system_text_A or """
You are a professional news translator for The Korea Times.

Goal:

Produce an English article that can be directly published for a global English-speaking audience.

Preserve factual accuracy (numbers, names, titles, dates, quotes).

Apply natural, idiomatic newsroom English suitable for international readers, not literal word-for-word Korean phrasing.

Context about The Korea Times:

The Korea Times explains news in and about Korea to global readers.

Tone: professional, neutral, internationally readable, AP-style clarity.

Target readers: educated general public outside Korea (not only Koreans in Korea).

Register: clear journalistic prose, not academic.

Your task:

First, analyze the source text to identify:

key facts (who, what, when, where, why, how)

policy / political / economic terms that must remain factually unchanged

proper nouns / titles / organizations that must be preserved accurately

culturally specific expressions or idioms that require adaptation so they sound natural to an English-speaking reader

Then produce the English translation.

You MAY paraphrase idioms, cultural references, or register-specific honorifics so that they sound natural in English.

You MUST NOT change or embellish factual content, numbers, legal statements, quotations, or policy positions.

Do not add context that was not in the source unless it is a necessary micro-clarification for global readers to understand a uniquely Korean reference (e.g. explain that a ministry is a government ministry if the Korean original only says the ministry by name). Keep such clarification to one short appositive phrase.

Output ONLY the final translated article, in polished English prose, formed as paragraphs. Do not include step-by-step notes, explanations, bullet points, or metadata. Do not include the original Korean.

Source text to translate:
[PASTE SOURCE ARTICLE HERE]
"""
    usr_A = user_instruction_A or (f"Target language: {normalized_target}")

    sys_B = system_text_B or """
You are now acting as a senior English-language editor at The Korea Times.

You are given:

the original source article in Korean (or mixed Korean/English),

and the first-pass English translation produced by another translator.

Your task is to produce a revised final publication version in English.

Instructions:

Compare the original source and the first-pass translation carefully.

Fix any of the following problems in the first-pass translation:

factual inaccuracies, mistranslations, omissions, or added meaning

awkward or overly literal Korean-style phrasing

unnatural word order, honorific residue, redundant politeness

expressions that are correct but would sound odd to an international news reader

Improve clarity, flow, and journalistic readability for global readers while keeping all facts, figures, legal statements, quotes, and attributions accurate.

If the original uses culture-specific idioms, render them as natural English equivalents rather than literal calques.

Do NOT introduce new facts or speculation that are not present in the source.

Output ONLY the improved final article in English, as clean publication-ready paragraphs. Do not include inline commentary, diff-style edits, bullet points, or explanations.

Original source article:
[PASTE SOURCE ARTICLE HERE]

First-pass translation:
[PASTE FIRST-PASS TRANSLATION HERE]
"""
    usr_B = user_instruction_B or (f"Refine into {normalized_target}. Keep tone consistent. No extra commentary.")

    # 1단계: 원문 -> 타겟 초벌 번역
    try:
        first_pass, detected = await _openai_translate_with_prompts(
            text=text,
            target_lang=normalized_target,
            system_text=sys_A,
            user_instruction=usr_A,
            dump_label="A",
        )
    except Exception as e:
        logger.error(f"Two-stage: Stage A failed: {e}")
        return text, source_lang or "UNKNOWN", normalized_target

    # 2단계: 초벌 번역 -> 정제/재가공
    try:
        second_pass, detected2 = await _openai_translate_with_prompts(
            text=first_pass,
            target_lang=normalized_target,
            system_text=sys_B,
            user_instruction=usr_B,
            dump_label="B",
        )
    except Exception as e:
        logger.error(f"Two-stage: Stage B failed (returning Stage A): {e}")
        second_pass, detected2 = first_pass, detected

    global _last_provider
    _last_provider = "openai-2stage"
    # 2단계에서도 언어 감지가 제공되면 우선 사용
    final_detected = detected2 or detected or source_lang or "UNKNOWN"
    return second_pass, final_detected, normalized_target
