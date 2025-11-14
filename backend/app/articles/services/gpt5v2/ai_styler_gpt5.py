#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korea Times AI Styler - GPT-5 Sentence-Based Matching
======================================================
문장 번호 기반 정확한 매칭으로 텍스트 교체 문제 해결
"""

import os
import json
import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI
import nltk
from .rule_based_corrector import RuleBasedCorrector

# NLTK 데이터 다운로드 확인
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StyleViolation:
    """스타일 위반 정보"""
    rule_id: str
    component: str
    rule_description: str
    sentence_id: str  # T1, B3, C2 등
    original_sentence: str
    corrected_sentence: str
    violation_type: str


class AIStylerGPT5Sentence:
    """GPT-5 문장 번호 기반 AI 스타일 교정기"""

    def __init__(self, api_key: Optional[str] = None, reasoning_effort: str = "low", text_verbosity: str = "low", use_compressed_prompt: bool = False):
        """
        Args:
            reasoning_effort: "minimal" | "low" | "medium" | "high"
            text_verbosity: "low" | "medium" | "high"
            use_compressed_prompt: True to use hard prompt compression (default: False)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)  # 병렬 호출용

        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.use_compressed_prompt = use_compressed_prompt

        self.regex_corrector = RuleBasedCorrector()
        self.style_guides = self._load_style_guides()
        self.ai_rules = self._filter_ai_rules()

        logger.info(f"GPT-5 Sentence-Based AI Styler 초기화 완료 - AI 처리 규칙: {len(self.ai_rules)}개")
        logger.info(f"  Reasoning effort: {reasoning_effort}, Text verbosity: {text_verbosity}, Compressed prompt: {use_compressed_prompt}")

    # --- OpenAI call helpers (env‑driven, with graceful fallback) ---
    def _env_float(self, name: str, default: float | None) -> float | None:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "y")

    async def _responses_create_async(self, *, model: str, input_blocks: list, tools: list, tool_choice: str | None,
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
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if include_reasoning and reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            params["text"] = {"verbosity": text_verbosity}

        try:
            return await self.async_client.responses.create(**params)
        except Exception as e:
            # Fallback: remove optional fields (reasoning/text) and retry once
            params.pop("reasoning", None)
            params.pop("text", None)
            try:
                return await self.async_client.responses.create(**params)
            except Exception:
                raise

    def _responses_create_sync(self, *, model: str, input_blocks: list, tools: list, tool_choice: str | None,
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
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if include_reasoning and reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            params["text"] = {"verbosity": text_verbosity}

        try:
            return self.client.responses.create(**params)
        except Exception:
            params.pop("reasoning", None)
            params.pop("text", None)
            return self.client.responses.create(**params)

    def _load_style_guides(self) -> Dict:
        """스타일 가이드 로드 (모듈 상대 경로)"""
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, 'gpt_style_guide.json')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _filter_ai_rules(self) -> Dict:
        """AI가 처리할 규칙만 필터링

        - 기본: JSON metadata의 excluded_regex_rules를 제외
        - 토글: GPT5_V2_INCLUDE_ALL_HEADLINE_RULES=true이면 헤드라인(H*)는 제외 목록 무시
        """
        import os as _os
        meta = self.style_guides.get('metadata', {}) if isinstance(self.style_guides, dict) else {}
        excluded = set(meta.get('excluded_regex_rules', []) or [])

        # 운영/QA 토글: 헤드라인 전수 검출을 강제하고 싶을 때 사용
        include_all_headline = str(_os.getenv('GPT5_V2_INCLUDE_ALL_HEADLINE_RULES', 'false')).strip().lower() in ('1', 'true', 'yes', 'y')

        ai_rules: Dict = {}
        for rule_id, rule_data in (self.style_guides.get('rules', {}) or {}).items():
            if include_all_headline and isinstance(rule_id, str) and rule_id.startswith('H'):
                ai_rules[rule_id] = rule_data
                continue
            if rule_id not in excluded:
                ai_rules[rule_id] = rule_data

        return ai_rules

    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분할 (NLTK 사용)"""
        if not text or not text.strip():
            return []
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # 폴백: 간단한 구분
            import re as _re
            sentences = _re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s and s.strip()]

    # Prompt dump helper (env-based)
    def _dump_prompt(self, kind: str, text: str, metadata: Optional[Dict] = None) -> None:
        try:
            dump = str(os.getenv('GPT5_V2_DUMP_PROMPTS', os.getenv('OPENAI_DUMP_PROMPTS', 'false'))).lower() == 'true'
            if not dump:
                return
            base_dir = os.getenv('GPT5_V2_LOG_DIR', os.getenv('OPENAI_PROMPT_DUMP_DIR', 'logs/gpt5v2'))
            os.makedirs(base_dir, exist_ok=True)
            from datetime import datetime
            import uuid
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            path = os.path.join(base_dir, f"{ts}_{kind}_{uuid.uuid4().hex[:8]}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                if metadata:
                    f.write(json.dumps(metadata, ensure_ascii=False))
                    f.write('\n\n')
                f.write(text or '')
            logger.debug("[v2] dumped %s to %s", kind, path)
        except Exception as exc:
            logger.debug("[v2] dump failed for %s: %s", kind, exc)

    def _format_kst_date_context(self, article_date: Optional[str]) -> str:
        """Return a compact KST date context block.

        - Input: article_date in ISO (YYYY-MM-DD) or None
        - Output lines:
          Article Date (KST): YYYY-MM-DD (DOW)
          Date context (±7, KST): MM-DD (DOW), ...

        If parsing fails or article_date is falsy, returns empty string.
        """
        if not article_date:
            return ""
        try:
            base = datetime.strptime(article_date, "%Y-%m-%d").date()
        except Exception:
            return f"Article Date (KST): {article_date}"

        dows = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = dows[base.weekday()]

        # Build ±7 day compact table
        days: List[str] = []
        for offset in range(-6, 7):  # inclusive
            d = base + timedelta(days=offset)
            tag = f"{d.month:02d}-{d.day:02d} ({dows[d.weekday()]})"
            if offset == 0:
                tag += " *"  # mark article date
            days.append(tag)

        return "\n".join([
            f"Article Date (KST): {base.isoformat()} ({dow})",
            "Date context (within 7 days, KST): " + ", ".join(days)
        ])

    def _infer_local_reference_date(self, sentences: List[str], article_date: Optional[str] = None) -> Optional[str]:
        """문맥에서 로컬 기준일(today)을 추론하여 ISO 날짜로 반환.

        - 절대 날짜(영문 Month Day 또는 한국어 M월 D일)와 상대표현(내일/이튿날/다음 날/어제/전날,
          tomorrow/the next day/yesterday/the day before)이 같은 문장 또는 인접 문장에 함께 존재하면
          절대 날짜를 기준으로 로컬 today를 계산해 반환.
        - 연도가 없으면 article_date의 연도를 가정.
        - 실패 시 None 반환.
        """
        try:
            if not sentences:
                return None

            # Prepare patterns
            import re
            month_names = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t\.|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
            en_anchor = re.compile(rf"\b({month_names})\s+(\d{{1,2}})\b", re.IGNORECASE)
            ko_anchor = re.compile(r"\b(1[0-2]|[1-9])월\s+(\d{1,2})일\b")

            rel_tokens = {
                'en': {
                    'tomorrow': -1,
                    'the next day': -1,
                    'following day': -1,
                    'yesterday': 1,
                    'the day before': 1,
                },
                'ko': {
                    '내일': -1,
                    '이튿날': -1,
                    '다음 날': -1,
                    '어제': 1,
                    '전날': 1,
                },
            }

            # Year fallback from article_date
            year = None
            if article_date:
                try:
                    year = int(str(article_date).split('T')[0].split('-')[0])
                except Exception:
                    year = None

            def parse_anchor(m: re.Match, is_en: bool) -> Optional[tuple]:
                try:
                    if is_en:
                        mon = m.group(1)
                        day = int(m.group(2))
                        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
                        # normalize month name
                        mon_l = mon.lower().rstrip('.')
                        # map common abbreviations
                        mapping = {
                            'jan':'January','january':'January','feb':'February','february':'February','mar':'March','march':'March',
                            'apr':'April','april':'April','may':'May','jun':'June','june':'June','jul':'July','july':'July','aug':'August','august':'August',
                            'sep':'September','sept':'September','september':'September','oct':'October','october':'October','nov':'November','november':'November',
                            'dec':'December','december':'December'
                        }
                        mon_full = mapping.get(mon_l, None)
                        if not mon_full:
                            return None
                        mon_idx = months.index(mon_full) + 1
                    else:
                        mon_idx = int(m.group(1))
                        day = int(m.group(2))
                    y = year or datetime.utcnow().year
                    from datetime import date
                    return date(y, mon_idx, day)
                except Exception:
                    return None

            # Search anchors and relatives by sentence index
            for i, s in enumerate(sentences):
                en_m = en_anchor.search(s)
                ko_m = ko_anchor.search(s)
                has_anchor = en_m or ko_m

                # look for relative cues in same or adjacent sentence
                window_idxs = [j for j in (i-1, i, i+1) if 0 <= j < len(sentences)]
                rel_dir = 0
                for j in window_idxs:
                    sj = sentences[j]
                    # english
                    for token, delta in rel_tokens['en'].items():
                        if token in sj.lower():
                            rel_dir = delta
                            break
                    if rel_dir == 0:
                        # korean
                        for token, delta in rel_tokens['ko'].items():
                            if token in sj:
                                rel_dir = delta
                                break
                    if rel_dir != 0:
                        break

                if has_anchor and rel_dir != 0:
                    anchor_date = parse_anchor(en_m, True) if en_m else parse_anchor(ko_m, False)
                    if not anchor_date:
                        continue
                    from datetime import timedelta
                    local_today = anchor_date + timedelta(days=rel_dir)
                    return local_today.isoformat()

            return None
        except Exception:
            return None

    def _extract_components(self, article_text: str) -> Dict[str, str]:
        """기사에서 컴포넌트 추출 (표준 태그: [TITLE], [BODY], [CAPTION])"""
        components = {}

        # Title
        title_match = re.search(r'\[TITLE\](.*?)\[/TITLE\]', article_text, re.DOTALL)
        components['title'] = title_match.group(1).strip() if title_match else ""

        # Body
        body_match = re.search(r'\[BODY\](.*?)\[/BODY\]', article_text, re.DOTALL)
        components['body'] = body_match.group(1).strip() if body_match else ""

        # Caption
        caption_match = re.search(r'\[CAPTION\](.*?)\[/CAPTION\]', article_text, re.DOTALL)
        components['caption'] = caption_match.group(1).strip() if caption_match else ""

        return components

    def _create_numbered_sentences(self, components: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """문장 분할 및 번호 부여"""
        numbered_sentences = {}
        sentence_map = {}  # sentence_id -> sentence_text

        # Title sentences
        title_sentences = self._split_sentences(components['title'])
        numbered_sentences['title'] = title_sentences
        for i, sent in enumerate(title_sentences):
            sid = f"T{i+1}"
            sentence_map[sid] = sent

        # Body sentences
        body_sentences = self._split_sentences(components['body'])
        numbered_sentences['body'] = body_sentences
        for i, sent in enumerate(body_sentences):
            sid = f"B{i+1}"
            sentence_map[sid] = sent

        # Caption sentences
        caption_sentences = self._split_sentences(components['caption'])
        numbered_sentences['caption'] = caption_sentences
        for i, sent in enumerate(caption_sentences):
            sid = f"C{i+1}"
            sentence_map[sid] = sent

        return numbered_sentences, sentence_map

    def _map_correction_to_sentence_id(
        self,
        correction,
        sentence_map: Dict[str, str],
        numbered_sentences: Dict[str, List[str]]
    ) -> str:
        """정규식 교정을 sentence_id로 매핑

        Args:
            correction: Correction 객체 (rule_id, component, original, corrected, position)
            sentence_map: sentence_id -> sentence_text 매핑
            numbered_sentences: component -> [sentences] 매핑

        Returns:
            sentence_id (예: "T1", "B3", "C2") 또는 "N/A"
        """
        component = correction.component.lower()  # 'title', 'body', 'caption'

        # Component prefix 매핑
        prefix_map = {
            'title': 'T',
            'body': 'B',
            'caption': 'C'
        }

        if component not in prefix_map:
            return "N/A"

        prefix = prefix_map[component]

        # 해당 component의 문장들 가져오기
        sentences = numbered_sentences.get(component, [])
        if not sentences:
            return "N/A"

        # corrected 텍스트가 포함된 문장 찾기
        # (정규식 교정은 문장 내 일부만 수정하므로, corrected 텍스트를 포함하는 문장을 찾음)
        corrected_text = correction.corrected.strip()
        original_text = correction.original.strip()

        # 1차 시도: corrected 텍스트를 포함하는 문장 찾기
        for i, sentence in enumerate(sentences):
            sentence_id = f"{prefix}{i+1}"
            # corrected 또는 original 텍스트가 문장에 포함되어 있는지 확인
            if corrected_text in sentence or original_text in sentence:
                return sentence_id

        # 2차 시도: 부분 매칭 (단어 기준)
        for i, sentence in enumerate(sentences):
            sentence_id = f"{prefix}{i+1}"
            # 원본 또는 교정된 텍스트의 주요 단어들이 문장에 있는지 확인
            corrected_words = set(corrected_text.split())
            original_words = set(original_text.split())
            sentence_words = set(sentence.split())

            # 교정된 텍스트의 단어 중 50% 이상이 문장에 포함되면 매칭
            if corrected_words and len(corrected_words & sentence_words) >= len(corrected_words) * 0.5:
                return sentence_id
            if original_words and len(original_words & sentence_words) >= len(original_words) * 0.5:
                return sentence_id

        # 매칭 실패 시 첫 번째 문장으로 할당 (component에 문장이 하나만 있는 경우)
        if len(sentences) == 1:
            return f"{prefix}1"

        return "N/A"

    def _map_correction_to_sentence_id_by_position(
        self,
        correction,
        components: Dict[str, str]
    ) -> str:
        """Correction.position 기반 문장 매핑(정규식 보강용)
        - component 텍스트에서 문장 경계를 계산하고 position이 포함된 문장으로 매핑
        - 실패 시 "N/A"
        """
        try:
            component = (getattr(correction, 'component', '') or '').lower()
            if component not in ('title', 'body', 'caption'):
                return "N/A"

            comp_text = components.get(component, '') or ''
            if not comp_text:
                return "N/A"

            sentences = self._split_sentences(comp_text)
            if not sentences:
                return "N/A"

            # 각 문장의 시작/끝 오프셋 계산 (좌->우 검색, 중복 문장 대비 커서 이동)
            offsets = []
            cursor = 0
            for s in sentences:
                idx = comp_text.find(s, cursor)
                if idx < 0:
                    idx = comp_text.find(s)
                    if idx < 0:
                        continue
                start = idx
                end = idx + len(s)
                offsets.append((start, end))
                cursor = end

            pos = getattr(correction, 'position', None)
            if pos is None or not isinstance(pos, int):
                return "N/A"

            for i, (start, end) in enumerate(offsets, start=1):
                if start <= pos < end:
                    prefix = {'title': 'T', 'body': 'B', 'caption': 'C'}[component]
                    return f"{prefix}{i}"
            return "N/A"
        except Exception:
            return "N/A"

    def correct_article(
        self,
        article_text: str,
        article_date: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """기사 전체를 교정"""
        logger.info("=== GPT-5 Sentence-Based 기사 스타일 교정 시작 ===")

        # 호출 단위로 사용자 추가 지침을 보관 (하위 빌더들이 참조)
        try:
            self._extra_instructions = (extra_instructions or "").strip() or None
        except Exception:
            self._extra_instructions = None

        # 사전 스냅샷: 정규식 적용 전 문장 맵 (history용 원문 문장 보존)
        try:
            components_before = self._extract_components(article_text)
            numbered_sentences_before, sentence_map_before = self._create_numbered_sentences(components_before)
        except Exception:
            numbered_sentences_before, sentence_map_before = ({'title': [], 'body': [], 'caption': []}, {})

        # 1단계: 정규식 기반 교정
        logger.info("1단계: 정규식 기반 교정 처리 중...")
        regex_result = self.regex_corrector.correct_article(article_text, article_date)
        regex_corrected_text = regex_result['corrected_text']
        regex_corrections = regex_result['corrections']
        logger.info(f"정규식 교정 완료 - {len(regex_corrections)}개 수정")

        # 2단계: 컴포넌트 추출 및 문장 분할
        logger.info("2단계: 문장 분할 및 번호 부여...")
        components = self._extract_components(regex_corrected_text)
        numbered_sentences, sentence_map = self._create_numbered_sentences(components)

        total_sentences = sum(len(sents) for sents in numbered_sentences.values())
        logger.info(f"  Title: {len(numbered_sentences['title'])}문장")
        logger.info(f"  Body: {len(numbered_sentences['body'])}문장")
        logger.info(f"  Caption: {len(numbered_sentences['caption'])}문장")
        logger.info(f"  Total: {total_sentences}문장")

        logger.info(f"  [중간산출물 0] 번호가 부여된 문장들:")
        for comp_type, sentences in numbered_sentences.items():
            if sentences:
                logger.info(f"    {comp_type.upper()}:")
                for i, sent in enumerate(sentences[:3]):  # 처음 3개만
                    prefix = {'title': 'T', 'body': 'B', 'caption': 'C'}[comp_type]
                    logger.info(f"      [{prefix}{i+1}] {sent[:60]}...")
                if len(sentences) > 3:
                    logger.info(f"      ... (총 {len(sentences)}개 문장)")

        # 3단계: GPT-5 AI 기반 교정
        logger.info("3단계: GPT-5 AI 기반 교정 처리 중...")
        ai_result = self._ai_correct_sentences(numbered_sentences, sentence_map, article_date)

        ai_violations = ai_result['violations']
        logger.info(f"AI 교정 완료 - {len(ai_violations)}개 위반 발견 및 수정")

        # 4단계: 문장 교체 및 기사 재구성
        logger.info("4단계: 문장 교체 및 기사 재구성...")
        corrected_components = self._apply_corrections(components, numbered_sentences, ai_violations)

        # 기사 재구성
        final_text = f"[TITLE]{corrected_components['title']}[/TITLE]"
        final_text += f"[BODY]{corrected_components['body']}[/BODY]"
        final_text += f"[CAPTION]{corrected_components['caption']}[/CAPTION]"

        # 결과 통합
        all_violations = []
        for corr in regex_corrections:
            # 정규식 교정을 sentence_id로 매핑
            sentence_id = self._map_correction_to_sentence_id_by_position(corr, components)
            if sentence_id == "N/A":
                sentence_id = self._map_correction_to_sentence_id(
                    corr,
                    sentence_map,
                    numbered_sentences
                )
            # 원문/교정 문장 텍스트 구성 (가능하면 전체 문장 사용)
            original_sentence_text = sentence_map_before.get(sentence_id, None)
            corrected_sentence_text = sentence_map.get(sentence_id, None)
            if not original_sentence_text:
                original_sentence_text = corr.original
            if not corrected_sentence_text:
                corrected_sentence_text = corr.corrected
            all_violations.append(StyleViolation(
                rule_id=corr.rule_id,
                component=corr.component,
                rule_description=f"Regex-based correction: {corr.rule_id}",
                sentence_id=sentence_id,
                original_sentence=original_sentence_text,
                corrected_sentence=corrected_sentence_text,
                violation_type="regex_pattern"
            ))

        all_violations.extend(ai_violations)

        stats = {
            'total_violations': len(all_violations),
            'regex_corrections': len(regex_corrections),
            'ai_corrections': len(ai_violations),
            'total_sentences': total_sentences,
            'by_component': self._calculate_component_stats(all_violations),
            'by_rule': self._calculate_rule_stats(all_violations)
        }

        logger.info(f"=== 교정 완료 - 총 {stats['total_violations']}개 위반 처리 ===")

        result_obj = {
            'original_text': article_text,
            'corrected_text': final_text,
            'violations': all_violations,
            'stats': stats
        }

        # 사용 후 정리 (메모리/상태 누수 방지)
        try:
            self._extra_instructions = None
        except Exception:
            pass

        return result_obj

    def _ai_correct_sentences(
        self,
        numbered_sentences: Dict[str, List[str]],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """2단계 AI 처리: Detection (GPT-5) -> Correction (GPT-5-mini)"""

        # 1단계: Detection - 위반 감지만
        logger.info("  Step 1: Detection (GPT-5) - 위반 감지 중...")
        detections = self._detect_violations(numbered_sentences, sentence_map, article_date)

        if not detections:
            logger.info("  감지된 위반 없음")
            return {'violations': []}

        logger.info(f"  감지된 위반: {len(detections)}개")

        # 2단계: Correction - 감지된 문장만 교정
        # 감지와 교정 사이에 소폭 지연(기본 2초)으로 레이트/연결 안정화
        try:
            _gap = float(os.getenv('OPENAI_DETECT_TO_CORRECT_DELAY_SEC', '2'))
        except Exception:
            _gap = 2.0
        if _gap > 0:
            logger.info(f"  Detection→Correction 대기: {_gap:.2f}s")
            try:
                import time as _time
                _time.sleep(_gap)
            except Exception:
                pass

        logger.info("  Step 2: Correction - 감지된 문장 교정 중...")
        corrections = self._correct_detected_sentences(detections, sentence_map, article_date)

        logger.info(f"  교정 완료: {len(corrections)}개 문장")

        # Detection과 Correction 결합
        violations = self._merge_detection_and_correction(detections, corrections, sentence_map)

        return {'violations': violations}

    def _detect_violations(
        self,
        numbered_sentences: Dict[str, List[str]],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """컴포넌트별로 병렬 Detection 수행 (Title/Body/Caption 동시 호출)"""

        # asyncio event loop에서 병렬 호출 실행
        return asyncio.run(self._detect_violations_async(numbered_sentences, sentence_map, article_date))

    async def _detect_violations_async(
        self,
        numbered_sentences: Dict[str, List[str]],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """비동기 병렬 Detection"""

        tasks = []
        component_names = []
        delays: List[float] = []

        # Title Detection (H 규칙만)
        if numbered_sentences['title']:
            logger.info(f"  [Title Detection] {len(numbered_sentences['title'])}개 문장 검사 중... (병렬)")
            tasks.append(self._detect_single_component_async('title', numbered_sentences['title'], article_date))
            component_names.append('title')
            delays.append(0.0)

        # Body Detection (A 규칙만)
        if numbered_sentences['body']:
            logger.info(f"  [Body Detection] {len(numbered_sentences['body'])}개 문장 검사 중... (병렬)")
            tasks.append(self._detect_single_component_async('body', numbered_sentences['body'], article_date))
            component_names.append('body')
            delays.append(2.0)

        # Caption Detection (C 규칙만)
        if numbered_sentences['caption']:
            logger.info(f"  [Caption Detection] {len(numbered_sentences['caption'])}개 문장 검사 중... (병렬)")
            tasks.append(self._detect_single_component_async('caption', numbered_sentences['caption'], article_date))
            component_names.append('caption')
            delays.append(2.0)

        # 모든 컴포넌트 병렬 처리 (stagger 적용)
        logger.info(f"  ⚡ {len(tasks)}개 컴포넌트 병렬 Detection 시작 (stagger base: {1.0:.2f}s)...")
        async def delayed_task(coro, delay):
            if delay > 0:
                await asyncio.sleep(delay)
            return await coro
        results = await asyncio.gather(*[delayed_task(t, d) for t, d in zip(tasks, delays)])

        # 결과 통합
        all_detections = []
        for i, (component_name, detections) in enumerate(zip(component_names, results)):
            all_detections.extend(detections)
            logger.info(f"  [{component_name.upper()}] {len(detections)}개 위반 검출")

        logger.info(f"  [병렬 Detection 완료] 총 {len(all_detections)}개 위반 검출")

        # Post-processing: Component 불일치 필터링
        validated_detections = self._validate_component_match(all_detections)

        return validated_detections

    async def _detect_single_component_async(
        self,
        component_type: str,
        sentences: List[str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """단일 컴포넌트에 대해 비동기 Detection 수행 (병렬용)"""
        await asyncio.sleep(5)

        # 해당 컴포넌트의 규칙만 필터링
        if component_type == 'title':
            prefix = 'H'
        elif component_type == 'body':
            prefix = 'A'
        else:  # caption
            prefix = 'C'

        component_rules = {k: v for k, v in self.ai_rules.items() if k.startswith(prefix)}

        # 프롬프트 생성 (해당 컴포넌트 규칙만) - 압축 여부에 따라 선택
        tools = self._build_detection_tools()
        if self.use_compressed_prompt:
            instructions = self._build_detection_instructions_for_component_compressed(component_type, component_rules)
        else:
            # instructions = self.build_optimized_detection_prompt(component_type, component_rules)
            instructions = self._build_detection_instructions_for_component(component_type, component_rules)
        input_content = self._build_input_content_for_component(component_type, sentences, article_date)

        # 프롬프트 로깅
        prompt_type = "COMPRESSED" if self.use_compressed_prompt else "STANDARD"
        logger.info(f" [{component_type.upper()}] {prompt_type} Instructions 크기: {len(instructions)} chars, 규칙 수: {len(component_rules)}개")

        try:
            # Async API 호출 (환경변수 기반 유연 파라미터)
            detect_model = os.getenv('OPENAI_DETECT_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest'))
            tool_choice = os.getenv('OPENAI_TOOL_CHOICE', 'required')
            temp = self._env_float('OPENAI_DETECT_TEMPERATURE', 0.1)
            top_p = self._env_float('OPENAI_DETECT_TOP_P', 0.02)
            include_reasoning = self._env_bool('OPENAI_INCLUDE_REASONING', False)
            text_verbosity = os.getenv('OPENAI_TEXT_VERBOSITY',  None)

            response = await self._responses_create_async(
                model=detect_model,
                input_blocks=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_content},
                ],
                tools=tools,
                tool_choice=tool_choice,
                temperature=temp,
                top_p=top_p,
                include_reasoning=include_reasoning,
                reasoning_effort=self.reasoning_effort,
                text_verbosity=text_verbosity,
            )
            # Dump prompts
            self._dump_prompt(
                f"detect_instructions_{component_type}",
                instructions,
                {"component": component_type, "rules": len(component_rules)}
            )
            self._dump_prompt(
                f"detect_input_{component_type}",
                input_content,
                {"component": component_type}
            )

            detections = []
            for item in response.output:
                if item.type == "function_call" and item.name == "detect_style_violations":
                    function_args = json.loads(item.arguments)
                    detections = function_args.get('violations', [])
                    break

            return detections

        except Exception as e:
            logger.error(f"[{component_type.upper()}] Detection 실패: {str(e)}")
            return []

    def _detect_single_component(
        self,
        component_type: str,
        sentences: List[str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """단일 컴포넌트에 대해 Detection 수행"""

        # 해당 컴포넌트의 규칙만 필터링
        if component_type == 'title':
            prefix = 'H'
            component_name = 'title'
        elif component_type == 'body':
            prefix = 'A'
            component_name = 'body'
        else:  # caption
            prefix = 'C'
            component_name = 'caption'

        component_rules = {k: v for k, v in self.ai_rules.items() if k.startswith(prefix)}

        # 프롬프트 생성 (해당 컴포넌트 규칙만)
        tools = self._build_detection_tools()
        instructions = self._build_detection_instructions_for_component(component_type, component_rules)
        input_content = self._build_input_content_for_component(component_type, sentences, article_date)

        # 프롬프트 로깅
        logger.info(f"    [{component_type.upper()}] Instructions 크기: {len(instructions)} chars, 규칙 수: {len(component_rules)}개")

        try:
            detect_model = os.getenv('OPENAI_DETECT_MODEL', os.getenv('GPT5_V2_DETECT_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest')))
            tool_choice = os.getenv('OPENAI_TOOL_CHOICE', os.getenv('GPT5_V2_TOOL_CHOICE', 'required'))
            temp = self._env_float('OPENAI_DETECT_TEMPERATURE', self._env_float('GPT5_V2_DETECT_TEMPERATURE', 0.1))
            top_p = self._env_float('OPENAI_DETECT_TOP_P', self._env_float('GPT5_V2_DETECT_TOP_P', None))
            include_reasoning = self._env_bool('OPENAI_INCLUDE_REASONING', self._env_bool('GPT5_V2_INCLUDE_REASONING', False))
            text_verbosity = os.getenv('OPENAI_TEXT_VERBOSITY', os.getenv('GPT5_V2_TEXT_VERBOSITY', None))

            response = self._responses_create_sync(
                model=detect_model,
                input_blocks=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_content},
                ],
                tools=tools,
                tool_choice=tool_choice,
                temperature=temp,
                top_p=top_p,
                include_reasoning=include_reasoning,
                reasoning_effort=self.reasoning_effort,
                text_verbosity=text_verbosity,
            )

            detections = []
            for item in response.output:
                if item.type == "function_call" and item.name == "detect_style_violations":
                    function_args = json.loads(item.arguments)
                    detections = function_args.get('violations', [])
                    break

            return detections

        except Exception as e:
            logger.error(f"[{component_type.upper()}] Detection 실패: {str(e)}")
            return []

    def _build_detection_instructions_for_component(
        self,
        component_type: str,
        component_rules: Dict
    ) -> str:
        """단일 컴포넌트에 대한 Detection 지시사항 생성 (JSON 구조 사용)"""

        # 컴포넌트별 이름 매핑 및 특별 강조사항
        component_display_names = {
            'title': ('HEADLINE', 'H', 'Title'),
            'body': ('BODY', 'A', 'Body'),
            'caption': ('CAPTION', 'C', 'Caption')
        }

        display_name, rule_prefix, sentence_type = component_display_names[component_type]

        # 동적으로 규칙 범위(H01-H13 등) 계산
        try:
            nums = sorted(int(k[1:]) for k in component_rules.keys() if isinstance(k, str) and k.startswith(rule_prefix))
            if nums:
                rule_range = f"{rule_prefix}{nums[0]:02d}-{rule_prefix}{nums[-1]:02d}"
            else:
                rule_range = f"{rule_prefix}**"
        except Exception:
            rule_range = f"{rule_prefix}**"

        # 컴포넌트별 특별 강조
        special_focus = ""
        if component_type == 'title':
            special_focus = "\n⚡ **SPECIAL FOCUS FOR HEADLINES**: Pay close attention to H06 (omitting articles a/an/the), H01 (first letter capitalization), and H02 (avoiding ALL CAPS)."
        elif component_type == 'body':
            # BODY 전용 특별 강조: A09 한국인 이름 순서
            special_focus = ""

        # JSON 규칙 포맷팅 (그룹 정보 포함)
        rules_text = []
        for rule_id, rule_data in component_rules.items():
            # 그룹 정보 추가
            group = rule_data.get('group', 'General')
            rule_text = f"**{rule_id} ({group})**: {rule_data['description']}"

            # Few-shot 예제 추가 (첫 번째 예제만)
            if rule_data.get('examples'):
                example = rule_data['examples'][0]
                rule_text += f"\n  ✗ Incorrect: '{example.get('incorrect', '')}'"
                rule_text += f"\n  ✓ Correct: '{example.get('correct', '')}'"

            rules_text.append(rule_text)

        rules_section = '\n\n'.join(rules_text)
        extra = getattr(self, '_extra_instructions', None)
        instructions = f"""
You are an expert copy editor for The Korea Times, specializing in photo captions.
Your task is to meticulously review the provided caption sentence(s) and identify all violations based on the official Caption Style Guide below. You must be thorough, accurate, and use the provided examples to guide your judgment.        
{special_focus}

TASK: DETECT {display_name} style guide violations - BE THOROUGH AND ACCURATE.

{display_name} STYLE GUIDE ({rule_range}) for {sentence_type} sentences:

{rules_section}

DETECTION PRINCIPLES:

BE THOROUGH - Review each sentence carefully against all applicable rules
BE ACCURATE - Report violations you are confident about
USE EXAMPLES - Compare sentences to the examples provided
BALANCE PRECISION AND RECALL - Find violations but avoid over-detection

DETECTION INSTRUCTIONS:

1. Review EVERY sentence systematically against ALL the above style guide rules
2. For violations you are confident about, report:
   - sentence_id (e.g., "T1" for title, "B3" for body, "C2" for caption)
   - rule_id (from the rules above, e.g., "H01", "A05", "C12")
   - component ("{component_type}")
   - rule_description (brief explanation of the violation)
   - violation_type: MUST be the exact group name shown in parentheses after the rule_id (e.g., for "H01 (Capitalization)", use "Capitalization")

3. If one sentence has multiple violations, return multiple entries

4. Check examples carefully - if the sentence matches a violation pattern, report it
5. When reasonably uncertain about borderline cases**, err on the side of not reporting
6. For special focus rules (if mentioned above), be extra thorough

7. If no violations are found, return an empty array []
{('8. Additionally, follow this instruction strictly: ' + extra) if extra else ''}

Note: Only check against {display_name} rules ({rule_range}). Do not apply rules from other sections."""
        return instructions
    
    def build_optimized_detection_prompt(
        self,
        component_type: str,
        component_rules: Dict[str, Any]
    ) -> str:
        """
        구조화, 명료화, 간결함을 통해 최적화된 Detection 프롬프트를 생성합니다.
        """
        component_map = {
            'title': ('HEADLINE', 'H', 'Title'),
            'body': ('BODY', 'A', 'Body'),
            'caption': ('CAPTION', 'C', 'Caption')
        }
        display_name, rule_prefix, _ = component_map[component_type]

        # --- 1. 규칙을 의미론적 그룹으로 묶어 AI의 컨텍스트 이해를 돕습니다 ---
        grouped_rules: Dict[str, list] = {}
        for rule_id, rule_data in component_rules.items():
            # JSON 데이터에 'group'이 없다면 'General Principles'로 기본값 설정
            group = rule_data.get('group', 'General Principles')
            if group not in grouped_rules:
                grouped_rules[group] = []
            # 나중에 사용하기 편하도록 튜플 형태로 저장
            grouped_rules[group].append((rule_id, rule_data))

        # --- 2. 그룹화된 규칙을 바탕으로 가독성 높은 텍스트 섹션을 만듭니다 ---
        rule_sections = []
        # 그룹 이름을 기준으로 정렬하여 항상 일관된 순서의 프롬프트 생성
        for i, (group_name, rules_in_group) in enumerate(sorted(grouped_rules.items()), 1):
            rule_texts = []
            for rule_id, rule_data in sorted(rules_in_group): # 규칙 ID 순으로 정렬
                description = rule_data.get('description', 'N/A')
                
                # 명시적으로 Category를 제공하여 AI가 violation_type을 쉽게 찾도록 함
                rule_text = f"  *   **{rule_id}: {description}**\n      *   **Category:** `{group_name}`"

                # 예시는 항상 제공하는 것이 안정적임 (특히 복잡한 규칙의 경우)
                if rule_data.get('examples'):
                    example = rule_data['examples'][0]
                    incorrect_ex = example.get('incorrect', '')
                    correct_ex = example.get('correct', '')
                    rule_text += f"\n      *   **Incorrect:** `{incorrect_ex}`\n      *   **Correct:** `{correct_ex}`"
                rule_texts.append(rule_text)
            
            section_content = f"### {i}. {group_name}\n\n" + "\n\n".join(rule_texts)
            rule_sections.append(section_content)

        rules_section_str = "\n\n---\n\n".join(rule_sections)
        
        # --- 3. 중복을 제거하고 핵심만 담은 지시사항을 생성합니다 ---
        # `TASK`, `PRINCIPLES`, `INSTRUCTIONS`를 하나로 통합
        instructions = f"""You are an expert copy editor for The Korea Times, specializing in {display_name.lower()}s.
    Your mission is to meticulously review the provided text and identify all violations based on the official Style Guide below.

    ---
    ### **{display_name} Style Guide**
    {rules_section_str}
    ---

    ### **Instructions & Output Format**

    1.  **Systematic Review:** Carefully check each sentence against all relevant rules in the guide.
    2.  **Confident Detection:** Only report violations you are certain about, using the examples as a reference.
    3.  **JSON Output:** For each violation, provide a JSON object with the following structure. If a sentence has multiple violations, create a separate object for each.

        ```json
        {{
        "sentence_id": "C1",
        "rule_id": "C05",
        "component": "{component_type}",
        "rule_description": "Do not use parentheses for positioning.",
        "violation_type": "Person Description"
        }}
        ```
    4.  **`violation_type` Field:** For the `violation_type`, you MUST use the exact string from the `Category` field of the violated rule.
    5.  **No Violations:** If you find no violations, return an empty array `[]`.
    """
        return instructions


    def _build_detection_instructions_for_component_compressed(
        self,
        component_type: str,
        component_rules: Dict
    ) -> str:
        """압축된 Detection 지시사항 생성 (하드 프롬프트 압축 기법 적용)"""

        # 컴포넌트별 이름 매핑
        component_display_names = {
            'title': ('HEADLINE', 'H', 'Title'),
            'body': ('BODY', 'A', 'Body'),
            'caption': ('CAPTION', 'C', 'Caption')
        }

        display_name, rule_prefix, sentence_type = component_display_names[component_type]

        # 동적으로 규칙 범위 계산
        try:
            nums = sorted(int(k[1:]) for k in component_rules.keys() if isinstance(k, str) and k.startswith(rule_prefix))
            if nums:
                rule_range = f"{rule_prefix}{nums[0]:02d}-{rule_prefix}{nums[-1]:02d}"
            else:
                rule_range = f"{rule_prefix}**"
        except Exception:
            rule_range = f"{rule_prefix}**"

        # 규칙을 복잡도별로 분류 (예제가 필요한 것 vs 필요없는 것)
        # 간단한 규칙: 명확한 패턴 (capitalization, number format 등)
        simple_rule_keywords = ['capitalize', 'uppercase', 'lowercase', 'number', 'spell out',
                               'percent', 'dollar', 'comma', 'hyphen', 'colon']

        complex_rules = []  # 예제 포함
        simple_rules = []   # 예제 제외

        for rule_id, rule_data in component_rules.items():
            desc_lower = rule_data['description'].lower()
            is_simple = any(keyword in desc_lower for keyword in simple_rule_keywords)

            if is_simple:
                # 간단한 규칙: ID와 설명만
                simple_rules.append(f"{rule_id}: {rule_data['description']}")
            else:
                # 복잡한 규칙: 예제 포함 (1개만)
                rule_text = f"{rule_id}: {rule_data['description']}"
                if rule_data.get('examples'):
                    example = rule_data['examples'][0]
                    rule_text += f" | X: '{example.get('incorrect', '')}' → O: '{example.get('correct', '')}'"
                complex_rules.append(rule_text)

        # 압축된 포맷 (스키마 기반)
        extra = getattr(self, '_extra_instructions', None)
        instructions = f"""Korea Times copy editor. Detect {display_name} violations ({rule_range}).

RULES ({rule_range}):{chr(10)}"""

        # 복잡한 규칙 먼저 (예제 포함, position bias 활용)
        if complex_rules:
            instructions += '\n'.join(complex_rules) + '\n'

        # 간단한 규칙 (예제 없음)
        if simple_rules:
            instructions += '\n'.join(simple_rules) + '\n'

        # 압축된 지시사항 (9개 → 4개)
        instructions += f"""
TASK:
1. Check all sentences vs rules above
2. Report: sentence_id (e.g. T1/B1/C1), rule_id, component ("{component_type}"), rule_description, violation_type (exact group name)
3. Multiple violations per sentence = multiple entries
4. Empty array [] if none found
{('5. Additionally, follow this instruction strictly: ' + extra) if extra else ''}"""

        return instructions

    def _build_input_content_for_component(
        self,
        component_type: str,
        sentences: List[str],
        article_date: Optional[str] = None
    ) -> str:
        """단일 컴포넌트에 대한 입력 생성"""

        # 컴포넌트별 sentence ID prefix
        prefix_map = {
            'title': 'T',
            'body': 'B',
            'caption': 'C'
        }

        prefix = prefix_map[component_type]
        component_display = component_type.upper()

        date_context = ""
        # 캡션은 로컬 기준일 + 기사 기준일(요일 포함)을 간단히 주입
        if component_type == 'caption':
            local_ref = self._infer_local_reference_date(sentences, article_date)
            ctx_block = None
            if article_date:
                try:
                    ctx_block = self._format_kst_date_context(article_date)
                except Exception:
                    ctx_block = f"Article Date (KST): {article_date}"
            if local_ref and ctx_block:
                date_context = f"\n\nLocal Reference Day (KST): {local_ref}\n{ctx_block}"
            elif local_ref:
                date_context = f"\n\nLocal Reference Day (KST): {local_ref}"
            elif ctx_block:
                date_context = f"\n\n{ctx_block}"
        elif component_type == 'body':
            # 본문에도 동일한 날짜 컨텍스트 제공 (필요 시 상대표현 정합성 개선)
            local_ref = self._infer_local_reference_date(sentences, article_date)
            ctx_block = None
            if article_date:
                try:
                    ctx_block = self._format_kst_date_context(article_date)
                except Exception:
                    ctx_block = f"Article Date (KST): {article_date}"
            if local_ref and ctx_block:
                date_context = f"\n\nLocal Reference Day (KST): {local_ref}\n{ctx_block}"
            elif local_ref:
                date_context = f"\n\nLocal Reference Day (KST): {local_ref}"
            elif ctx_block:
                date_context = f"\n\n{ctx_block}"
        else:
            if article_date:
                ctx = self._format_kst_date_context(article_date)
                if ctx:
                    date_context = "\n\n" + ctx

        # 사용자 추가 지침이 있으면 상단에 명시
        extra = getattr(self, '_extra_instructions', None)

        content = f"""Please review ALL {component_type} sentences and identify style guide violations.

{date_context}

**{component_display} SENTENCES:**
"""

        for i, sent in enumerate(sentences):
            content += f"[{prefix}{i+1}] {sent}\n"

        # TASK 블록 구성 (추가 지시사항이 있으면 5번 항목으로 포함)
        task_lines = [
            "**TASK:**",
            f"1. Check EVERY sentence against ALL applicable {component_type} rules",
            "2. For violations, return:",
            f"   - sentence_id (e.g., \"{prefix}1\", \"{prefix}2\", etc.)",
            "   - rule_id",
            f"   - component (\"{component_type}\")",
            "   - rule_description",
            "   - violation_type",
            "3. Focus on finding ALL violations",
            "4. If no violations, return empty array []",
        ]
        if extra:
            task_lines.append(f"5. Additionally, follow this instruction strictly: {extra}")

        content += "\n".join(task_lines) + "\n"

        return content

    def _validate_component_match(self, detections: List[Dict]) -> List[Dict]:
        """Component와 Rule ID의 일치 여부 검증 (Post-processing filter)"""
        valid_detections = []
        filtered_count = 0

        # Component mapping: sentence_id prefix -> expected rule_id prefix
        component_map = {
            'T': 'H',  # Title -> Headline rules
            'B': 'A',  # Body -> Article/Body rules
            'C': 'C'   # Caption -> Caption rules
        }

        for d in detections:
            sentence_id = d['sentence_id']
            rule_id = d['rule_id']

            # Extract prefixes
            sentence_component = sentence_id[0]  # T, B, or C
            rule_component = rule_id[0]  # H, A, or C

            # Expected rule prefix for this sentence type
            expected_rule_prefix = component_map.get(sentence_component)

            # Validate match
            if rule_component != expected_rule_prefix:
                logger.warning(
                    f"Component 불일치 제거: sentence={sentence_id} (type={sentence_component}) "
                    f"에 rule={rule_id} (type={rule_component}) 적용 시도 - 예상 rule type: {expected_rule_prefix}"
                )
                filtered_count += 1
                continue

            valid_detections.append(d)

        if filtered_count > 0:
            logger.info(f"  Component 불일치 {filtered_count}개 제거됨 (Total: {len(detections)} → Valid: {len(valid_detections)})")

        return valid_detections

    def _correct_detected_sentences(
        self,
        detections: List[Dict],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> Dict[str, str]:
        """컴포넌트별로 병렬 Correction 수행 (Title/Body/Caption 동시 호출)"""

        # asyncio event loop에서 병렬 호출 실행
        return asyncio.run(self._correct_detected_sentences_async(detections, sentence_map, article_date))

    async def _correct_detected_sentences_async(
        self,
        detections: List[Dict],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> Dict[str, str]:
        """비동기 병렬 Correction (rate limit 회피를 위한 stagger 추가)"""

        # 컴포넌트별로 detections 그룹화
        detections_by_component = {'title': [], 'body': [], 'caption': []}
        for d in detections:
            sentence_id = d['sentence_id']
            if sentence_id.startswith('T'):
                detections_by_component['title'].append(d)
            elif sentence_id.startswith('B'):
                detections_by_component['body'].append(d)
            elif sentence_id.startswith('C'):
                detections_by_component['caption'].append(d)

        # 추가 지시사항이 있는 경우, 탐지 결과가 비어도 모든 문장을 교정 대상으로 폴백
        extra = getattr(self, '_extra_instructions', None)
        if extra and not any(detections_by_component.values()) and sentence_map:
            try:
                for sid in sentence_map.keys():
                    if sid.startswith('T'):
                        detections_by_component['title'].append({
                            'sentence_id': sid,
                            'rule_id': 'HX',  # placeholder
                            'component': 'title',
                            'rule_description': 'extra_instruction',
                            'violation_type': 'extra'
                        })
                    elif sid.startswith('B'):
                        detections_by_component['body'].append({
                            'sentence_id': sid,
                            'rule_id': 'AX',
                            'component': 'body',
                            'rule_description': 'extra_instruction',
                            'violation_type': 'extra'
                        })
                    elif sid.startswith('C'):
                        detections_by_component['caption'].append({
                            'sentence_id': sid,
                            'rule_id': 'CX',
                            'component': 'caption',
                            'rule_description': 'extra_instruction',
                            'violation_type': 'extra'
                        })
                logger.info("No violations detected; applying fallback correction for all sentences due to extra instructions.")
            except Exception:
                pass

        tasks = []
        component_names = []
        stagger_delays = []  # 각 task의 시작 지연 (초)

        # Title Correction
        if detections_by_component['title']:
            logger.info(f"  [Title Correction] {len(detections_by_component['title'])}개 위반 교정 중... (stagger 적용)")
            tasks.append(self._correct_single_component_async('title', detections_by_component['title'], sentence_map, article_date))
            component_names.append('title')
            stagger_delays.append(0.0)

        # Body Correction
        if detections_by_component['body']:
            logger.info(f"  [Body Correction] {len(detections_by_component['body'])}개 위반 교정 중... (stagger 적용)")
            tasks.append(self._correct_single_component_async('body', detections_by_component['body'], sentence_map, article_date))
            component_names.append('body')
            stagger_delays.append(2.0)

        # Caption Correction
        if detections_by_component['caption']:
            logger.info(f"  [Caption Correction] {len(detections_by_component['caption'])}개 위반 교정 중... (stagger 적용)")
            tasks.append(self._correct_single_component_async('caption', detections_by_component['caption'], sentence_map, article_date))
            component_names.append('caption')
            stagger_delays.append(2.0)

        # Stagger를 적용한 병렬 처리
        logger.info(f"  ⚡ {len(tasks)}개 컴포넌트 staggered 병렬 Correction 시작 (base: {1.0:.2f}s)...")

        # 각 task에 지연을 적용하여 시작
        async def delayed_task(task, delay):
            if delay > 0:
                await asyncio.sleep(delay)
            return await task

        staggered_tasks = [delayed_task(task, delay) for task, delay in zip(tasks, stagger_delays)]
        results = await asyncio.gather(*staggered_tasks)

        # 결과 통합
        all_corrections = {}
        for i, (component_name, corrections) in enumerate(zip(component_names, results)):
            all_corrections.update(corrections)
            logger.info(f"  [{component_name.upper()}] {len(corrections)}개 문장 교정 완료")

        logger.info(f"  [병렬 Correction 완료] 총 {len(all_corrections)}개 문장 교정")

        return all_corrections

    async def _correct_single_component_async(
        self,
        component_type: str,
        component_detections: List[Dict],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> Dict[str, str]:
        """단일 컴포넌트에 대해 비동기 Correction 수행 (병렬용)"""

        # 감지된 문장 ID들만 추출 (중복 제거)
        detected_sentence_ids = list(set(d['sentence_id'] for d in component_detections))

        # 교정할 문장들
        sentences_to_correct = {
            sid: sentence_map[sid] for sid in detected_sentence_ids if sid in sentence_map
        }

        if not sentences_to_correct:
            return {}

        tools = self._build_correction_tools()
        input_content = self._build_correction_input(sentences_to_correct, component_detections, article_date)
        instructions = self._build_correction_instructions(component_type)

        try:
            # Async API 호출 (환경변수 기반 유연 파라미터)
            correct_model = os.getenv('OPENAI_CORRECT_MODEL', os.getenv('GPT5_V2_CORRECT_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest')))
            tool_choice = os.getenv('OPENAI_TOOL_CHOICE', os.getenv('GPT5_V2_TOOL_CHOICE', 'required'))
            temp = self._env_float('OPENAI_CORRECT_TEMPERATURE', self._env_float('GPT5_V2_CORRECT_TEMPERATURE', None))
            top_p = self._env_float('OPENAI_CORRECT_TOP_P', self._env_float('GPT5_V2_CORRECT_TOP_P', 0.2))
            include_reasoning = self._env_bool('OPENAI_INCLUDE_REASONING', self._env_bool('GPT5_V2_INCLUDE_REASONING', False))
            text_verbosity = os.getenv('OPENAI_TEXT_VERBOSITY', os.getenv('GPT5_V2_TEXT_VERBOSITY', None))

            response = await self._responses_create_async(
                model=correct_model,
                input_blocks=[
                    {"role": "system","content": [{"type": "input_text", "text": instructions}]},
                    {"role": "user", "content": input_content}],
                tools=tools,
                tool_choice=tool_choice,
                temperature=temp,
                top_p=top_p,
                include_reasoning=include_reasoning,
                reasoning_effort=self.reasoning_effort,
                text_verbosity=text_verbosity,
            )
            # Dump prompts
            self._dump_prompt(
                f"correct_instructions_{component_type}",
                instructions,
                {"component": component_type, "count": len(sentences_to_correct)}
            )
            self._dump_prompt(
                f"correct_input_{component_type}",
                input_content,
                {"component": component_type}
            )

            corrections = {}
            for item in response.output:
                if item.type == "function_call" and item.name == "correct_sentences":
                    function_args = json.loads(item.arguments)
                    correction_list = function_args.get('corrections', [])

                    for c in correction_list:
                        corrections[c['sentence_id']] = c['corrected_sentence']
                    break

            return corrections

        except Exception as e:
            logger.error(f"[{component_type.upper()}] Correction 실패: {str(e)}")
            return {}

    def _merge_detection_and_correction(
        self,
        detections: List[Dict],
        corrections: Dict[str, str],
        sentence_map: Dict[str, str]
    ) -> List[StyleViolation]:
        """Detection과 Correction 결과를 결합하여 StyleViolation 생성"""

        violations = []

        logger.info(f"  [중간산출물 4] Merge 단계:")
        logger.info(f"    - Detection 개수: {len(detections)}개")
        logger.info(f"    - Correction 개수: {len(corrections)}개 sentence")

        for i, detection in enumerate(detections):
            sentence_id = detection['sentence_id']
            original_sentence = sentence_map.get(sentence_id, "")
            corrected_sentence = corrections.get(sentence_id, original_sentence)

            has_correction = sentence_id in corrections
            logger.info(f"    #{i+1}: {detection['rule_id']} @ {sentence_id} - Correction: {'✓' if has_correction else '✗'}")

            violations.append(StyleViolation(
                rule_id=detection['rule_id'],
                component=detection['component'],
                rule_description=detection['rule_description'],
                sentence_id=sentence_id,
                original_sentence=original_sentence,
                corrected_sentence=corrected_sentence,
                violation_type=detection['violation_type']
            ))

        logger.info(f"  [중간산출물 5] 최종 Violations: {len(violations)}개")

        return violations

    def _build_detection_tools(self) -> List[Dict]:
        """Detection용 Function calling tools (교정문 없음)"""
        return [{
            "type": "function",
            "name": "detect_style_violations",
            "description": "Detect Korea Times style guide violations and report sentence ID and rule ID only.",
            "strict": True,  # Strict mode 활성화
            "parameters": {
                "type": "object",
                "properties": {
                    "violations": {
                        "type": "array",
                        "description": "List of detected violations with sentence ID and rule ID.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sentence_id": {
                                    "type": "string",
                                    "description": "Sentence identifier (e.g., T1, B3, C2)"
                                },
                                "rule_id": {
                                    "type": "string",
                                    "description": "Style rule ID (H01-H08, A01-A39, C01-C33)"
                                },
                                "component": {
                                    "type": "string",
                                    "description": "Component: title, body, or caption"
                                },
                                "rule_description": {
                                    "type": "string",
                                    "description": "Brief description of the violation"
                                },
                                "violation_type": {
                                    "type": "string",
                                    "description": "Type of violation"
                                }
                            },
                            "required": ["sentence_id", "rule_id", "component", "rule_description", "violation_type"],
                            "additionalProperties": False  # Strict mode 필수
                        }
                    }
                },
                "required": ["violations"],
                "additionalProperties": False  # Strict mode 필수
            }
        }]

    def _build_correction_tools(self) -> List[Dict]:
        """Correction용 Function calling tools (교정문만)"""
        extra = getattr(self, '_extra_instructions', None)
        desc = "Return corrected sentences for detected violations."
        if extra:
            desc += f" Additionally, follow this instruction strictly: {extra}"
        return [{
            "type": "function",
            "name": "correct_sentences",
            "description": desc,
            "strict": True,  # Strict mode 활성화
            "parameters": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "description": "List of corrected sentences.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sentence_id": {
                                    "type": "string",
                                    "description": "Sentence identifier (e.g., T1, B3, C2)"
                                },
                                "corrected_sentence": {
                                    "type": "string",
                                    "description": "The fully corrected sentence with ALL violations fixed"
                                }
                            },
                            "required": ["sentence_id", "corrected_sentence"],
                            "additionalProperties": False  # Strict mode 필수
                        }
                    }
                },
                "required": ["corrections"],
                "additionalProperties": False  # Strict mode 필수
            }
        }]

    def _build_detection_instructions(self) -> str:
        """Detection용 지시사항 (GPT-5) - 간소화 버전"""

        rules_by_component = {
            'headline': [],
            'body': [],
            'caption': []
        }

        for rule_id, rule_data in self.ai_rules.items():
            component = rule_data['component']
            if component == 'title':
                component = 'headline'

            rule_text = f"**{rule_id}**: {rule_data['description']}"

            # Few-shot 예제 추가
            if rule_data.get('examples'):
                example = rule_data['examples'][0]
                rule_text += f"\n  ✗ Incorrect: '{example.get('incorrect', '')}'"
                rule_text += f"\n  ✓ Correct: '{example.get('correct', '')}'"

            rules_by_component[component].append(rule_text)

        instructions = f"""You are an expert copy editor for the Korea Times.

TASK: DETECT style guide violations in the sentences below.**

HEADLINE RULES (H01-H08) for Title sentences:
{chr(10).join(rules_by_component['headline'])}

BODY RULES (A01-A39) for Body sentences:
{chr(10).join(rules_by_component['body'])}

CAPTION RULES (C01-C33) for Caption sentences:
{chr(10).join(rules_by_component['caption'])}

INSTRUCTIONS:

1. Review EVERY sentence carefully
2. Check each sentence against ALL applicable style rules
3. For each violation found, report:
   - sentence_id (e.g., "B3")
   - rule_id (e.g., "A08")
   - component ("title", "body", or "caption")
   - rule_description (brief explanation)
   - violation_type (type of violation)

4. If one sentence has multiple violations, return multiple entries
5. Focus on finding ALL violations (maximize recall)
6. If no violations are found, return an empty array

Note: Title sentences use H rules, Body sentences use A rules, Caption sentences use C rules."""

        return instructions

    def _build_correction_instructions(self, component_type: str) -> str:
        """Correction용 지시사항 (GPT-5-mini)"""
        extra = getattr(self, '_extra_instructions', None)

        instructions = """You are a copy editor for the Korea Times.

TASK: CORRECT the sentences that have violations

You will receive:
1. Sentences that need correction (with their IDs)
2. List of violations detected in each sentence

CORRECTION PROCESS:

1. For each sentence, apply ALL the violations fixes mentioned
2. Return the FULLY CORRECTED sentence
3. Make sure ALL violations in that sentence are fixed

RETURN FORMAT:
- sentence_id: The sentence identifier (e.g., "B3")
- corrected_sentence: The COMPLETE sentence with ALL violations fixed

IMPORTANT:
- Apply ALL fixes to each sentence
- Return the complete corrected sentence, not fragments
- Make minimal changes (only fix the violations)
- Use the provided examples: when a rule includes '✗ Incorrect'/'✓ Correct', rewrite to match the '✓ Correct' pattern
- If multiple rewrites are possible, choose the minimal edit that matches the example"""

        if extra:
            instructions += f"\n\nADDITIONAL INSTRUCTION (apply strictly): {extra}"

        # 캡션 전용 지침만 캡션에 포함
        if component_type == 'caption':
            instructions += """

DATE NORMALIZATION (KST):
- Use the "Date context (within 7 days, KST)" block below as the only basis for conversion.
- If the referenced date is listed on the past side of that context, rewrite as 'last <weekday>'.
- If it is listed on the future side, rewrite as '<weekday>'.
- If it is not listed (older than 7 days or beyond 7 days), keep 'Month Day[, Year]' and omit the weekday.
- If the input date has no year, assume the article year only for deciding recency; do not add a year in the output unless it was present in the input.
- When ambiguous, do not convert; keep the absolute date.
"""

            instructions += """

CREDIT NORMALIZATION (CAPTIONS):
- Exactly one final credit. Do not duplicate or combine credits.
- Sentential captions (has a finite verb): end the sentence with a period, then use either 'Courtesy of [Company].' or an in‑house/agency token (e.g., 'Korea Times file.'); do NOT use '/'.
- Non‑sentential captions (noun phrases): do not introduce a verb. Use ' / ' before the single credit (e.g., ' / Courtesy of [Company]' or ' / Korea Times file').
- Do not place a period immediately before '/': replace '. /' with ' / '.
- Canonicalize 'Korea Times photo' → 'Korea Times file'; never split as 'Korea Times / File'.
- Do not combine 'Courtesy of ...' with any agency/in‑house credit (including 'Korea Times file')."""

        # 본문 전용 날짜 규칙 지침 추가
#         if component_type == 'body':
#             instructions += """

# DATE NORMALIZATION (KST - BODY):
# - Use the "Date context (within 7 days, KST)" block below as the only basis for conversion.
# - If the referenced date is listed on the past side of that context, rewrite as 'last <weekday>'.
# - If it is listed on the future side, rewrite as '<weekday>'.
# - If it is not listed (older than 7 days or beyond 7 days), keep 'Month Day[, Year]' and omit the weekday.
# - If the input date has no year, assume the article year only for deciding recency; do not add a year in the output unless it was present in the input.
# - When ambiguous, do not convert; keep the absolute date.
# """

        # 한국 고유명 표기 교정은 본문/캡션에서만 사용 (헤드라인 제외)
        if component_type in ('body', 'caption'):
            instructions += """

KOREAN-RELATED NOTATION (PALACE NAMES):
- Rewrite 'Gyeongbokgung/Changdeokgung/Deoksugung/Changgyeonggung/Gyeonghuigung' to 'Gyeongbok/Changdeok/Deoksu/Changgyeong/Gyeonghui Palace'.
- If it already appears as '<…>gung Palace', rewrite to '<Base> Palace' (e.g., 'Gyeongbokgung Palace' → 'Gyeongbok Palace').
"""

        return instructions

    def _build_correction_input(
        self,
        sentences_to_correct: Dict[str, str],
        detections: List[Dict],
        article_date: Optional[str] = None
    ) -> str:
        """Correction용 입력 생성 - 규칙 상세 정보와 예제 포함"""

        # 문장별로 감지된 위반들을 그룹화
        violations_by_sentence = {}
        for d in detections:
            sid = d['sentence_id']
            if sid not in violations_by_sentence:
                violations_by_sentence[sid] = []
            violations_by_sentence[sid].append(d['rule_id'])

        extra = getattr(self, '_extra_instructions', None)
        date_context = ""
        # 캡션/본문 교정 입력에도 로컬 기준일 + 기사 기준일(요일 포함) 컨텍스트 블록 주입
        try:
            sids = list(sentences_to_correct.keys())
            is_caption = any(str(sid).startswith('C') for sid in sids)
        except Exception:
            is_caption = False
        if is_caption or any(str(sid).startswith('B') for sid in sids):
            try:
                sents = [sentences_to_correct[sid] for sid in sids]
                local_ref = self._infer_local_reference_date(sents, article_date)
                ctx_block = None
                if article_date:
                    try:
                        ctx_block = self._format_kst_date_context(article_date)
                    except Exception:
                        ctx_block = f"Article Date (KST): {article_date}"
                if local_ref and ctx_block:
                    date_context = f"Local Reference Day (KST): {local_ref}\n{ctx_block}\n\n"
                elif local_ref:
                    date_context = f"Local Reference Day (KST): {local_ref}\n\n"
                elif ctx_block:
                    date_context = f"{ctx_block}\n\n"
            except Exception:
                if article_date:
                    try:
                        date_context = self._format_kst_date_context(article_date) + "\n\n"
                    except Exception:
                        date_context = f"Article Date (KST): {article_date}\n\n"
        else:
            if article_date:
                ctx = self._format_kst_date_context(article_date)
                if ctx:
                    date_context = ctx + "\n\n"

        content = f"{date_context}**SENTENCES TO CORRECT:**\n\n"

        for sid in sorted(sentences_to_correct.keys()):
            original = sentences_to_correct[sid]
            rule_ids = violations_by_sentence.get(sid, [])

            content += f"[{sid}] {original}\n"
            content += f"  Violations to fix:\n"

            # 각 규칙의 상세 정보와 예제 추가
            for rule_id in rule_ids:
                if rule_id in self.ai_rules:
                    rule_data = self.ai_rules[rule_id]
                    content += f"    - **{rule_id}**: {rule_data['description']}\n"

                    # 예제 추가 (첫 번째 예제만)
                    if rule_data.get('examples'):
                        ex = rule_data['examples'][0]
                        content += f"      ✗ Incorrect: '{ex.get('incorrect', '')}'\n"
                        content += f"      ✓ Correct: '{ex.get('correct', '')}'\n"
                else:
                    content += f"    - {rule_id}: (rule details not found)\n"

            content += "\n"

        # TASK 블록 구성 (예시 준수 강조 및 추가 지시사항 포함)
        # 캡션 여부 판단
        try:
            sids = list(sentences_to_correct.keys())
            is_caption = any(str(s).startswith('C') for s in sids)
        except Exception:
            is_caption = False

        task_lines = [
            "**TASK:**",
            "1. For each sentence, apply ALL the violation fixes according to the rule descriptions and examples above.",
            "2. Return the FULLY CORRECTED sentence with all violations fixed.",
            "3. Follow the '✓ Correct' examples exactly when available; choose the minimal edit that matches the example",
        ]
        if is_caption:
            task_lines.append("4. CAPTION-SPECIFIC: Do not introduce a subject or verb; noun-phrase captions are acceptable. Preserve the original structure (phrase vs full sentence).")
            task_lines.append("5. CAPTION-SPECIFIC: Ensure exactly one 'Courtesy of [Company]' credit at the end. If it already exists, do not add or duplicate it.")
            task_lines.append("6. CAPTION-SPECIFIC: Do NOT add any weekday or date unless it appears in the original text; ") # only normalize existing date references per C12.
        if extra:
            step_no = len(task_lines) + 1
            task_lines.append(f"{step_no}. Additionally, follow this instruction strictly: {extra}")

        content += "\n".join(task_lines) + "\n"

        return content

    def _build_input_content(
        self,
        numbered_sentences: Dict[str, List[str]],
        article_date: Optional[str] = None
    ) -> str:
        """입력 컨텐츠 생성 - 번호가 매겨진 문장 리스트"""

        date_context = ""
        if article_date:
            date_context = f"\n\nArticle Date: {article_date}"

        content = f"""Please review ALL sentences and identify style guide violations.

{date_context}

NUMBERED SENTENCES:

TITLE:
"""
        for i, sent in enumerate(numbered_sentences['title']):
            content += f"[T{i+1}] {sent}\n"

        content += "\n**BODY:**\n"
        for i, sent in enumerate(numbered_sentences['body']):
            content += f"[B{i+1}] {sent}\n"

        content += "\n**CAPTION:**\n"
        for i, sent in enumerate(numbered_sentences['caption']):
            content += f"[C{i+1}] {sent}\n"

        content += """
**TASK:**
1. Check EVERY sentence against ALL applicable rules
2. For violations, return:
   - sentence_id (e.g., "B3")
   - rule_id
   - corrected_sentence (the FULL corrected sentence)
3. Focus on finding ALL violations
4. If no violations, return empty array []
"""

        return content

    def _parse_response(
        self,
        response,
        sentence_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """응답 파싱 - 문장 번호 기반"""
        try:
            violations_data = []

            for item in response.output:
                if item.type == "function_call" and item.name == "correct_style_violations":
                    function_args = json.loads(item.arguments)
                    violations_data = function_args.get('violations', [])
                    break

            if not violations_data:
                logger.warning("No function call in response")
                return {'violations': []}

            violations = []
            for v in violations_data:
                sentence_id = v['sentence_id']

                # 원본 문장 가져오기
                original_sentence = sentence_map.get(sentence_id, "")

                if not original_sentence:
                    logger.warning(f"문장 ID를 찾을 수 없음: {sentence_id}")
                    continue

                violations.append(StyleViolation(
                    rule_id=v['rule_id'],
                    component=v['component'],
                    rule_description=v['rule_description'],
                    sentence_id=sentence_id,
                    original_sentence=original_sentence,
                    corrected_sentence=v['corrected_sentence'],
                    violation_type=v['violation_type']
                ))

            logger.info(f"문장 번호 기반 교정 완료: {len(violations)}개 적용")

            return {'violations': violations}

        except Exception as e:
            logger.error(f"Response 파싱 실패: {str(e)}")
            return {'violations': [], 'error': str(e)}

    def _apply_corrections(
        self,
        components: Dict[str, str],
        numbered_sentences: Dict[str, List[str]],
        violations: List[StyleViolation]
    ) -> Dict[str, str]:
        """문장 교체 및 컴포넌트 재구성 (원문 줄바꿈/공백 유지)

        원문 컴포넌트 텍스트에서 NLTK로 문장을 탐지하되, 재조립 시
        문장 사이의 모든 공백(개행 포함)을 원문 그대로 보존한다.
        """

        # sentence_id -> corrected_sentence 매핑
        corrected_by_sid: Dict[str, str] = {}
        for v in violations:
            try:
                if v.sentence_id and v.corrected_sentence:
                    corrected_by_sid[v.sentence_id] = v.corrected_sentence
            except Exception:
                continue

        def rebuild_component(comp_text: str, prefix: str) -> str:
            if not comp_text:
                return comp_text
            # NLTK로 문장 분할 (텍스트만 가져옴)
            sents = self._split_sentences(comp_text)
            if not sents:
                return comp_text
            result_parts: List[str] = []
            pos = 0
            for idx, sent in enumerate(sents, start=1):
                # 현재 문장의 원문 위치를 찾아 앞의 공백을 먼저 추가
                found = comp_text.find(sent, pos)
                if found == -1:
                    # 예상치 못한 경우: 남은 모든 텍스트를 붙이고 종료
                    result_parts.append(comp_text[pos:])
                    logger.debug(f"Sentence not found while rebuilding {prefix}{idx}; preserving tail as-is")
                    return ''.join(result_parts)
                # 앞의 구분자(공백/개행 등) 보존
                if found > pos:
                    result_parts.append(comp_text[pos:found])
                # 교정문 적용 여부
                sid = f"{prefix}{idx}"
                corrected = corrected_by_sid.get(sid, sent)
                result_parts.append(corrected)
                pos = found + len(sent)
            # 남은 꼬리 공백/개행 보존
            if pos < len(comp_text):
                result_parts.append(comp_text[pos:])
            return ''.join(result_parts)

        title_out = rebuild_component(components.get('title', ''), 'T')
        body_out = rebuild_component(components.get('body', ''), 'B')
        caption_out = rebuild_component(components.get('caption', ''), 'C')

        return {
            'title': title_out,
            'body': body_out,
            'caption': caption_out,
        }

    def _calculate_component_stats(self, violations: List[StyleViolation]) -> Dict:
        """컴포넌트별 통계"""
        stats = {}
        for v in violations:
            stats[v.component] = stats.get(v.component, 0) + 1
        return stats

    def _calculate_rule_stats(self, violations: List[StyleViolation]) -> Dict:
        """규칙별 통계"""
        stats = {}
        for v in violations:
            stats[v.rule_id] = stats.get(v.rule_id, 0) + 1
        return stats


def main():
    """테스트"""
    styler = AIStylerGPT5Sentence(reasoning_effort="low", text_verbosity="low")

    test_article = """[TITLE]korea expands childcare support - 50 percent increase[/TITLE][BODY]President Yoon Suk-yeol stated the government will invest 50,000 won ($38) per child. Samsung Electronics chairman Lee Jae-yong attended the ceremony. There are 5 participants.[/BODY][CAPTION]President Yoon Suk-yeol poses for a photo. Yonhap.[/CAPTION]"""

    print("="*80)
    print("GPT-5 Sentence-Based AI Styler Test")
    print("="*80)

    result = styler.correct_article(test_article, "2025-10-18")

    print("\n교정 결과:")
    print(result['corrected_text'])

    print(f"\n\n발견된 위반: {result['stats']['total_violations']}개")
    print(f"  - 정규식: {result['stats']['regex_corrections']}")
    print(f"  - AI: {result['stats']['ai_corrections']}")
    print(f"  - 총 문장 수: {result['stats']['total_sentences']}")

    for v in result['violations']:
        if v.sentence_id != "N/A":
            print(f"\n[{v.sentence_id}] {v.rule_id} ({v.component}):")
            print(f"  Original:  {v.original_sentence[:80]}...")
            print(f"  Corrected: {v.corrected_sentence[:80]}...")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
