"""
OpenAI GPT-5 v2 pipeline adapter
Integrates the external v2 GPT-5 (3-experts) styler into current SSE + DB flow.
"""

import json
import time
import logging
import asyncio
from typing import AsyncGenerator, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import nltk
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...styleguides import service as style_guide_service
from ..models import ArticleCategory, OperationType
from .translation import translate_text, translate_title, _is_probably_english
import re

logger = logging.getLogger(__name__)

# Lazy singleton for v2 styler to avoid repeated initialization
_styler_v2 = None


def _dump_v2(kind: str, text: str, metadata: Optional[Dict] = None) -> None:
    """Dump v2 prompts or outputs to file if enabled."""
    try:
        enabled = settings.GPT5_V2_DUMP_PROMPTS or settings.OPENAI_DUMP_PROMPTS
    except Exception:
        enabled = False
    if not enabled:
        return

    try:
        base_dir = settings.GPT5_V2_LOG_DIR or settings.OPENAI_PROMPT_DUMP_DIR or "logs/gpt5v2"
        import os, uuid
        from datetime import datetime
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(base_dir, f"{ts}_{kind}_{uuid.uuid4().hex[:8]}.txt")
        with open(path, "w", encoding="utf-8") as f:
            if metadata:
                f.write(json.dumps(metadata, ensure_ascii=False))
                f.write("\n\n")
            f.write(text or "")
        logger.debug("[gpt5v2] dumped %s to %s", kind, path)
    except Exception as exc:
        logger.debug("[gpt5v2] dump failed for %s: %s", kind, exc)


def _get_styler_v2():
    global _styler_v2
    if _styler_v2 is not None:
        return _styler_v2

    from .gpt5v2.ai_styler_gpt5_3experts import AIStylerGPT5_3Experts

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for GPT-5 v2 pipeline")

    logger.info("Initializing GPT-5 v2 styler (3-Experts)...")
    # Expert 분리 토글 (기본 True, 환경/설정으로 제어 가능)
    try:
        use_expert_split = bool((settings.__dict__.get('GPT5_V2_USE_EXPERT_SPLIT', False)))
    except Exception:
        # settings에 없으면 환경변수 직접 확인
        import os as _os
        raw = str(_os.getenv('GPT5_V2_USE_EXPERT_SPLIT', 'true')).strip().lower()
        use_expert_split = raw in ('1', 'true', 'yes', 'y')

    _styler_v2 = AIStylerGPT5_3Experts(
        api_key=api_key,
        reasoning_effort=settings.OPENAI_REASONING_EFFORT or "low",
        text_verbosity="low",
        use_compressed_prompt=False,
        use_expert_split=use_expert_split,
    )
    logger.info("GPT-5 v2 styler initialized")
    return _styler_v2


def _reapply_a09_name_order(text: str) -> str:
    """
    BODY 최종 교정문에 대해, 안전한 범위에서 A09(한국인 이름 순서)를 재적용합니다.
    - 'Given-name Family' (하이픈 연결) + 한국 성씨 화이트리스트일 때만
    - 쉼표 표기(예: 'Hwang, Jin-seok')나 이미 family-first는 영향 없음(패턴 미적용)
    """
    if not text:
        return text
    surnames = [
        'Kim', 'Kwak', 'Gwak', 'Lee','Rhee','Yi','Park','Bak','Pak','Choi','Jung','Jeong','Kang','Cho','Jo',
        'Yoon','Yun','Jang','Chang','Lim','Im','Han','Shin','Sin','Yoo','Yu','Hwang','Kwon','Gwon',
        'Oh','O','Seo','Suh','Moon','Mun','Ryu','Nam','Song','Hong','Jeon','Chun','Jun','Ko','Koh',
        'Bae','Pae','Baek','Paek','Byun','Byeon','Cha','Ha','Heo','Hur','No','Roh','Noh'
    ]
    surname_regex = r"(?:" + "|".join(sorted(set(surnames), key=len, reverse=True)) + r")"
    # 두 번째 토큰은 대소문자 모두 허용 (예: Jin-seok, Jin-Seok)
    pattern = re.compile(rf"\b([A-Z][a-z]+-[A-Za-z][a-z]+)\s+({surname_regex})\b")

    changes: List[tuple[str, str]] = []

    def repl(m: re.Match) -> str:
        given = m.group(1)
        family = m.group(2)
        original = f"{given} {family}"
        corrected = f"{family} {given}"
        changes.append((original, corrected))
        return corrected

    new_text = pattern.sub(repl, text)

    # 실제 치환이 발생한 경우에만 로그 출력
    if changes:
        # 최대 3건만 프리뷰
        preview = "; ".join([f"{o} -> {c}" for o, c in changes[:3]])
        more = f" (+{len(changes)-3} more)" if len(changes) > 3 else ""
        logger.info(f"[gpt5v2] A09 reapplied on BODY: {len(changes)} change(s): {preview}{more}")

    return new_text


def _wrap_with_tags(text: str, category: ArticleCategory) -> str:
    """Wrap a single-category text into [TITLE]/[BODY]/[CAPTION] containers."""
    title = text if category == ArticleCategory.TITLE or category == ArticleCategory.SEO else ""
    body = text if category == ArticleCategory.BODY else ""
    caption = text if category == ArticleCategory.CAPTION else ""
    return f"[TITLE]{title}[/TITLE][BODY]{body}[/BODY][CAPTION]{caption}[/CAPTION]"


def _extract_component_from_tagged(full_text: str, category: ArticleCategory) -> str:
    import re
    if category in (ArticleCategory.TITLE, ArticleCategory.SEO):
        m = re.search(r"\[TITLE\](.*?)\[/TITLE\]", full_text, re.DOTALL)
    elif category == ArticleCategory.BODY:
        m = re.search(r"\[BODY\](.*?)\[/BODY\]", full_text, re.DOTALL)
    else:
        m = re.search(r"\[CAPTION\](.*?)\[/CAPTION\]", full_text, re.DOTALL)
    return (m.group(1) if m else "").strip()


def _prefix_for_category(category: ArticleCategory) -> str:
    if category in (ArticleCategory.TITLE, ArticleCategory.SEO):
        return "T"
    if category == ArticleCategory.BODY:
        return "B"
    return "C"


def _collect_applicable_rules(violations: List, category: ArticleCategory) -> List[str]:
    """Collect unique A/H/C rule_ids for the selected category only."""
    prefix = {ArticleCategory.TITLE: "H", ArticleCategory.SEO: "H",
              ArticleCategory.BODY: "A",
              ArticleCategory.CAPTION: "C"}[category]

    seen = set()
    rules: List[str] = []
    for v in violations:
        # v may be dataclass StyleViolation or dict
        rid = getattr(v, 'rule_id', None) if hasattr(v, 'rule_id') else v.get('rule_id') if isinstance(v, dict) else None
        sid = getattr(v, 'sentence_id', None) if hasattr(v, 'sentence_id') else v.get('sentence_id') if isinstance(v, dict) else None

        if not rid or not sid or sid == "N/A":
            continue
        if rid.startswith(prefix) and rid not in seen:
            seen.add(rid)
            rules.append(rid)
    return rules


    


def _split_sentences_nltk(text: str) -> List[str]:
    if not text:
        return []
    try:
        return [s.strip() for s in nltk.sent_tokenize(text) if s and s.strip()]
    except Exception:
        import re as _re
        return [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s and s.strip()]


def _build_sentence_corrections(orig_component_text: str, violations: List, category: ArticleCategory, corrected_component_text: Optional[str] = None) -> Dict[int, Dict]:
    """Return mapping of sentence_index -> {original, corrected, violations} using NLTK and per-violation corrections.

    - Bases indices on the original component's NLTK segmentation
    - For indices with a correction in violations, use that corrected sentence; otherwise, keep original
    """
    before_sents = _split_sentences_nltk(orig_component_text or "")
    corrected_sents = _split_sentences_nltk(corrected_component_text or "") if corrected_component_text is not None else []

    # Build violation map by zero-based sentence index in the selected component
    component_prefix = _prefix_for_category(category)
    viol_by_idx: Dict[int, List[str]] = {}
    corr_by_idx: Dict[int, str] = {}
    for v in violations:
        rid = getattr(v, 'rule_id', None) if hasattr(v, 'rule_id') else v.get('rule_id') if isinstance(v, dict) else None
        sid = getattr(v, 'sentence_id', None) if hasattr(v, 'sentence_id') else v.get('sentence_id') if isinstance(v, dict) else None
        corr_sent = getattr(v, 'corrected_sentence', None) if hasattr(v, 'corrected_sentence') else (
            v.get('corrected_sentence') if isinstance(v, dict) else None
        )
        if not rid or not sid or sid == "N/A":
            continue
        if not sid.startswith(component_prefix):
            continue
        # sid like B3 → index 2
        try:
            idx = int(sid[1:]) - 1
        except Exception:
            continue
        if idx < 0:
            continue
        viol_by_idx.setdefault(idx, []).append(rid)
        if corr_sent:
            corr_by_idx[idx] = corr_sent

    # Merge into sentence_corrections
    sent_corr: Dict[int, Dict] = {}
    for i, original in enumerate(before_sents):
        default_corrected = corrected_sents[i] if corrected_sents and i < len(corrected_sents) else original
        sent_corr[i] = {
            "original": original,
            "corrected": corr_by_idx.get(i, default_corrected),
            "violations": viol_by_idx.get(i, []),
        }
    return sent_corr


async def call_gpt5v2_correction_stream(
    prompt: str,
    text: str,
    category: ArticleCategory,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    GPT-5 v2 기반 교정 스트리밍
    - DeepL 번역 (자동 감지)
    - GPT-5 v2 Styler로 정규식+AI Detection+Correction 수행
    - 기존 SSE 이벤트 구조 준수
    """
    start_time = time.time()
    logger.info(f"🚀 Starting GPT-5 v2 correction stream for category={category}")

    try:
        # Step 1: Translation
        yield json.dumps({"status": "translating", "message": "번역중..."}, ensure_ascii=False)

        t0 = time.time()
        # 제목/SEO: 입력이 이미 영어면 번역/재서술을 생략하여 원문을 그대로 사용
        if category in (ArticleCategory.TITLE, ArticleCategory.SEO):
            if _is_probably_english(text or ""):
                logger.info("[gpt5v2] Headline appears to be English; skipping translate_title")
                before_en, source_lang, target_lang = text, "EN", "EN-US"
            else:
                logger.info("[gpt5v2] Using translate_title (single-pass) for headline/SEO")
                before_en, source_lang, target_lang = await translate_title(text)
        else:
            print("nononono hi headline  ?? ? ? ?")
            before_en, source_lang, target_lang = await translate_text(
                text,
                source_lang=None,
                target_lang="EN-US",
            )
        t_translation = time.time() - t0
        yield json.dumps({"status": "translation_complete", "message": "번역 완료"}, ensure_ascii=False)

        # Step 2: Prepare input for v2 styler
        wrapped = _wrap_with_tags(before_en, category)
        # Generate article_date (KST) for style rules depending on recency
        try:
            article_date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
        except Exception:
            article_date = datetime.utcnow().date().isoformat()
        _dump_v2(
            "wrapped_input",
            wrapped,
            {
                "category": category.value if hasattr(category, "value") else str(category),
                "article_date": article_date,
            },
        )

        # Step 3: Analysis + Correction via v2 styler
        yield json.dumps({"status": "applying_style", "message": "스타일 가이드 적용중..."}, ensure_ascii=False)
        s0 = time.time()
        styler = _get_styler_v2()
        # 전달된 사용자 추가 지침(prompt)을 Styler에 넘겨 Detection/Correction 지시문에 반영
        result = await asyncio.to_thread(styler.correct_article, wrapped, article_date, prompt)
        t_ai = time.time() - s0

        corrected_tagged = result.get("corrected_text", "")
        violations = result.get("violations", [])

        # Applicable rules and style_ids
        logger.info(
            "[gpt5v2] Violations detected: total=%s",
            len(violations) if isinstance(violations, list) else "unknown",
        )

        applicable_rules = _collect_applicable_rules(violations, category)
        _dump_v2(
            "violations_summary",
            json.dumps(
                {
                    "applicable_rules": applicable_rules,
                    "violations_count": len(applicable_rules),
                    "article_date": article_date,
                },
                ensure_ascii=False,
            ),
        )
        try:
            guides = await style_guide_service.get_guides_by_applicable_rules(db, rule_ids=applicable_rules)
            style_ids = [g.id for g in guides]
        except Exception as e:
            logger.debug(f"Style id mapping failed: {e}")
            style_ids = []

        logger.info(
            "[gpt5v2] Applicable rules=%s, resolved style_ids=%s",
            applicable_rules,
            style_ids,
        )

        yield json.dumps({
            "type": "analysis",
            "data": {
                "applicable_rules": applicable_rules,
                "style_guide_violations": [
                    {"id": rid, "description": f"Style guide violation: {rid}"} for rid in applicable_rules
                ],
                "style_ids": style_ids,
                "violations_count": len(applicable_rules),
            },
        })

        yield json.dumps({"status": "analysis_complete", "message": "스타일 가이드 분석 완료"}, ensure_ascii=False)

        # Step 4: Stream corrected text (component only)
        full_corrected = _extract_component_from_tagged(corrected_tagged, category)
        if not full_corrected:
            logger.info("[gpt5v2] Corrected text empty for category %s. Falling back to translated text.", category)
            full_corrected = before_en

        # Re-apply safe A09 rule (BODY only): ensure family-first order persists after AI replacement
        if category == ArticleCategory.BODY and full_corrected:
            before_a09 = full_corrected
            full_corrected = _reapply_a09_name_order(full_corrected)
            if before_a09 != full_corrected:
                logger.info("[gpt5v2] Reapplied A09 (family-first) on BODY corrected text")

        # Courtesy of ... dedupe and ". /" normalization for captions (safety net)
        courtesy_dedupe_applied = False
        if category == ArticleCategory.CAPTION and full_corrected:
            try:
                # Collapse repeated "Courtesy of X" at the end (allowing optional trailing periods/spaces)
                deduped = re.sub(r'(Courtesy of\s+[^\.\n]+?)(?:\.?\s+\1\.?)+$', r'\1', full_corrected, flags=re.IGNORECASE)
                courtesy_dedupe_applied = (deduped != full_corrected)
                full_corrected = deduped
                # Normalize ". / Agency" -> " / Agency"
                full_corrected = re.sub(r'\.\s*/\s*(Courtesy of|Yonhap|AP|Reuters|AFP|Getty Images|EPA|Bloomberg|Korea Times)\b', r' / \1', full_corrected, flags=re.IGNORECASE)
                # Normalize Korea Times file credit: avoid "Korea Times / File"; prefer " / Korea Times file"
                # Step 1) collapse variations to token
                full_corrected = re.sub(r'Korea\s+Times\s*/\s*File\b', 'Korea Times file', full_corrected, flags=re.IGNORECASE)
                # Step 2) lowercase 'File' in credit
                full_corrected = re.sub(r'Korea\s+Times\s+File\b', 'Korea Times file', full_corrected)
                # Step 3) ensure " / " before the token (and not a period immediately before)
                full_corrected = re.sub(r'\.\s*(?=Korea\s+Times\s+file\b)', ' ', full_corrected)
                full_corrected = re.sub(r'(?<!/)\s+(Korea\s+Times\s+file\b)', r' / \1', full_corrected)

                # Dedupe trailing repeated Korea Times file credits (e.g., "... / Korea Times file / Korea Times file.")
                full_corrected = re.sub(r'(\s*/\s*Korea\s+Times\s+file\.?)(\s*/\s*Korea\s+Times\s+file\.?)+\s*$', r' / Korea Times file', full_corrected, flags=re.IGNORECASE)
                # Also collapse trailing token duplicates without leading '/'
                full_corrected = re.sub(r'(\s+Korea\s+Times\s+file\.?)(\s+Korea\s+Times\s+file\.?)+\s*$', r' / Korea Times file', full_corrected, flags=re.IGNORECASE)
            except Exception:
                courtesy_dedupe_applied = False

        # Enforce currency conversion policy: only the first KRW amount keeps dollar parentheses; later ones drop them
        try:
            import re as _re
            def _enforce_single_won_conversion(_text: str) -> str:
                pat = _re.compile(
                    r'('  # group 1: KRW amount token
                    r'(?:₩\s?\d[\d,]*(?:\.\d+)?'  # ₩120,000 or ₩120,000.50
                    r'|'  # or
                    r'\b\d[\d,]*(?:\.\d+)?\s*(?:won|krw)\b'  # 120,000 won / 120,000won / 120,000 KRW
                    r')'
                    r')\s*\(\s*(?:US\$|\$)[^\)]+\)',
                    _re.IGNORECASE
                )
                seen = False
                def repl(m: _re.Match) -> str:
                    nonlocal seen
                    if seen:
                        return m.group(1)
                    seen = True
                    return m.group(0)
                return pat.sub(repl, _text)
            full_corrected = _enforce_single_won_conversion(full_corrected or "")
        except Exception as _exc:
            logger.debug("[gpt5v2] won conversion enforcement skipped: %s", _exc)

        _dump_v2(
            "corrected_text",
            full_corrected,
            {"category": category.value if hasattr(category, 'value') else str(category)},
        )
        collected: List[str] = []
        # Chunked streaming to match SSE pattern
        CHUNK = 300
        for i in range(0, len(full_corrected), CHUNK):
            chunk = full_corrected[i:i+CHUNK]
            collected.append(chunk)
            # Router가 payload_data만 전달하므로 data.choices에 담아 전송합니다
            yield json.dumps({
                "type": "delta",
                "data": {
                    "choices": [
                        {"delta": {"content": chunk}}
                    ]
                }
            })

        # Step 5: Sentence-level corrections (use original component text for baseline)
        orig_component_text = _extract_component_from_tagged(_wrap_with_tags(before_en, category), category)
        # If dedupe was applied for caption, synthesize a C16 violation for history (maps to the sentence containing Courtesy)
        if courtesy_dedupe_applied and category == ArticleCategory.CAPTION:
            try:
                before_sents = _split_sentences_nltk(orig_component_text or "")
                pat_dup = re.compile(r'(Courtesy of\s+[^\.\n]+?)(?:\.?\s+\1\.?)+$', re.IGNORECASE)
                target_idx = None
                for i, s in enumerate(before_sents):
                    if pat_dup.search(s):
                        target_idx = i
                        break
                if target_idx is None and before_sents:
                    # If duplication spanned across joining, fall back to last sentence
                    target_idx = len(before_sents) - 1
                if target_idx is not None:
                    sid = f"C{target_idx+1}"
                    # Append synthetic violation for history
                    violations = list(violations) if not isinstance(violations, list) else violations
                    violations.append({
                        'rule_id': 'C16',
                        'sentence_id': sid,
                        'component': 'caption',
                        'rule_description': 'Courtesy credit deduplication',
                        'violation_type': 'Source Attribution',
                        'corrected_sentence': None  # will be filled from corrected_sents in builder
                    })
            except Exception:
                pass

        sent_map = _build_sentence_corrections(orig_component_text, violations, category, corrected_component_text=full_corrected)
        yield json.dumps({
            "type": "sentence_corrections",
            "data": {
                "sentence_corrections": sent_map,
                "total_sentences": len(sent_map),
                "corrected_sentences": len(sent_map),
                "full_text": full_corrected,
            },
        }, ensure_ascii=False)
        yield json.dumps({"status": "sentence_parsing_complete", "message": "문장별 교정 파싱 완료"}, ensure_ascii=False)

        # Step 6: Final analysis payload (for DB save)
        total_time = time.time() - start_time
        final_analysis = {
            "applicable_rules": applicable_rules,
            "style_ids": style_ids,
            "sentence_corrections": sent_map,
            "full_text": full_corrected,
            "article_date": article_date,
            "translation": {
                "before_text": before_en,
                "source_lang": source_lang or "UNKNOWN",
                "target_lang": target_lang or "EN-US",
            },
            "processing_time": {
                "translation": t_translation,
                "analysis": t_ai,
                "correction": 0,  # included in t_ai
                "total": total_time,
            },
        }
        _dump_v2("final_analysis", json.dumps(final_analysis, ensure_ascii=False))
        yield json.dumps({"type": "final_analysis", "data": final_analysis}, ensure_ascii=False)
        yield json.dumps({"status": "complete", "message": "교정 완료"}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"GPT-5 v2 correction stream error: {e}")
        yield json.dumps({"type": "error", "data": {"message": f"처리 중 오류가 발생했습니다: {str(e)}"}})
        raise
