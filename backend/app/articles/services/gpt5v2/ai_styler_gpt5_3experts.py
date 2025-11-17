#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korea Times AI Styler - GPT-5 with 3-Expert Body Detection
===========================================================
Body 규칙을 3개 전문가 그룹으로 나눠서 병렬 처리
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
from .ai_styler_gpt5 import AIStylerGPT5Sentence

logger = logging.getLogger(__name__)


class AIStylerGPT5_3Experts(AIStylerGPT5Sentence):
    """3-Expert Body Detection을 사용하는 AI Styler"""

    def __init__(self, api_key: Optional[str] = None, reasoning_effort: str = "low", text_verbosity: str = "low", use_compressed_prompt: bool = False, use_expert_split: Optional[bool] = None):
        super().__init__(api_key, reasoning_effort, text_verbosity, use_compressed_prompt)

        # Expert 분리 사용 여부 (기본 True). None이면 환경변수에서 결정.
        if use_expert_split is None:
            raw = str(os.getenv('GPT5_V2_USE_EXPERT_SPLIT', 'false')).strip().lower()
            use_expert_split = raw in ('1', 'true', 'yes', 'y')
        self.use_expert_split = bool(use_expert_split)

        # Load expert groups configuration (module-relative)
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, 'body_expert_groups.json')
        with open(path, 'r', encoding='utf-8') as f:
            self.expert_config = json.load(f)

        if self.use_expert_split:
            logger.info(f"3-Expert Body Detection 모드 활성화")
        else:
            logger.info(f"Single-pass Detection 모드 활성화 (Expert 분리 비활성)")
        logger.info(f"  Expert 1 (Formatting): {len(self.expert_config['experts']['expert_1_formatting']['rule_ids'])}개 규칙")
        logger.info(f"  Expert 2 (Quotation & Naming): {len(self.expert_config['experts']['expert_2_quotation_naming']['rule_ids'])}개 규칙")
        logger.info(f"  Expert 3 (Grammar): {len(self.expert_config['experts']['expert_3_grammar']['rule_ids'])}개 규칙")

        # Optional: Load caption expert groups (2-experts) if present
        self.caption_expert_config = None
        cap_path = os.path.join(base_dir, 'caption_expert_groups.json')
        try:
            if os.path.exists(cap_path):
                with open(cap_path, 'r', encoding='utf-8') as f:
                    self.caption_expert_config = json.load(f)
                logger.info("2-Expert Caption Detection 모드 활성화")
                try:
                    cap_experts = self.caption_expert_config.get('experts', {})
                    for key, info in cap_experts.items():
                        logger.info(f"  Caption {info.get('name', key)}: {len(info.get('rule_ids', []))}개 규칙")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"Caption expert groups 로드 실패 (단일 호출 사용): {exc}")

    async def _detect_violations_async(
        self,
        numbered_sentences: Dict[str, List[str]],
        sentence_map: Dict[str, str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """비동기 병렬 Detection - Expert 분리 설정에 따라 단일/다중 호출"""

        # Expert 분리 비활성 시, 부모 클래스의 단일 컴포넌트 Detection 사용
        if not self.use_expert_split:
            return await super()._detect_violations_async(numbered_sentences, sentence_map, article_date)

        tasks = []
        component_names = []

        # Title Detection (기존 방식)
        if numbered_sentences['title']:
            logger.info(f"  [Title Detection] {len(numbered_sentences['title'])}개 문장 검사 중... (병렬)")
            tasks.append(self._detect_single_component_async('title', numbered_sentences['title'], article_date))
            component_names.append('TITLE')

        # Body Detection (3-expert 방식)
        if numbered_sentences['body']:
            logger.info(f"  [Body Detection - 3 Experts] {len(numbered_sentences['body'])}개 문장 검사 중... (병렬)")
            tasks.append(self._detect_body_3experts_async(numbered_sentences['body'], article_date))
            component_names.append('BODY-3EXPERTS')

        # Caption Detection (2-expert 방식이 있으면 사용, 없으면 기존 방식)
        if numbered_sentences['caption']:
            if self.caption_expert_config and self.caption_expert_config.get('experts'):
                logger.info(f"  [Caption Detection - 2 Experts] {len(numbered_sentences['caption'])}개 문장 검사 중... (병렬)")
                tasks.append(self._detect_caption_2experts_async(numbered_sentences['caption'], article_date))
                component_names.append('CAPTION-2EXPERTS')
            else:
                logger.info(f"  [Caption Detection] {len(numbered_sentences['caption'])}개 문장 검사 중... (단일)")
                tasks.append(self._detect_single_component_async('caption', numbered_sentences['caption'], article_date))
                component_names.append('CAPTION')

        # 모든 컴포넌트 병렬 처리
        logger.info(f"  ⚡ {len(tasks)}개 컴포넌트 병렬 Detection 시작...")
        results = await asyncio.gather(*tasks)

        # 결과 통합
        all_detections = []
        for i, (component_name, detections) in enumerate(zip(component_names, results)):
            all_detections.extend(detections)
            logger.info(f"  [{component_name}] {len(detections)}개 위반 검출")

        logger.info(f"  [병렬 Detection 완료] 총 {len(all_detections)}개 위반 검출")

        # Post-processing: Component 불일치 필터링
        validated_detections = self._validate_component_match(all_detections)

        return validated_detections

    async def _detect_body_3experts_async(
        self,
        body_sentences: List[str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """Body를 3개 전문가로 나눠서 병렬 검사"""

        # 3개 전문가 API 동시 호출
        expert_tasks = []
        expert_names = []

        for expert_key, expert_data in self.expert_config['experts'].items():
            expert_id = expert_data['expert_id']
            expert_name = expert_data['name']
            rule_ids = expert_data['rule_ids']

            logger.info(f"    [Body {expert_name}] {len(rule_ids)}개 규칙으로 검사 중...")

            # 해당 전문가의 규칙만 추출
            expert_rules = {rid: self.ai_rules[rid] for rid in rule_ids if rid in self.ai_rules}

            # 전문가별 Detection 실행
            task = self._detect_with_specific_rules_async(
                'body',
                body_sentences,
                expert_rules,
                expert_name,
                article_date
            )
            expert_tasks.append(task)
            expert_names.append(expert_name)

        # 3개 전문가 병렬 실행
        logger.info(f"    ⚡ 3개 Body Expert 병렬 실행...")
        expert_results = await asyncio.gather(*expert_tasks)

        # 결과 합치기
        all_body_detections = []
        for expert_name, detections in zip(expert_names, expert_results):
            all_body_detections.extend(detections)
            logger.info(f"    [{expert_name}] {len(detections)}개 위반 검출")

        logger.info(f"    [3-Expert Body 완료] 총 {len(all_body_detections)}개 위반 검출")

        return all_body_detections

    async def _detect_caption_2experts_async(
        self,
        caption_sentences: List[str],
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """Caption을 2개 전문가로 나눠서 병렬 검사"""

        if not self.caption_expert_config:
            # Fallback to single
            return await self._detect_single_component_async('caption', caption_sentences, article_date)

        expert_tasks: List = []
        expert_names: List[str] = []

        for expert_key, expert_data in self.caption_expert_config.get('experts', {}).items():
            expert_name = expert_data.get('name', expert_key)
            rule_ids = expert_data.get('rule_ids', [])

            if not rule_ids:
                continue

            logger.info(f"    [Caption {expert_name}] {len(rule_ids)}개 규칙으로 검사 중...")

            # 해당 전문가의 규칙만 추출 (ai_rules에 존재하는 것만)
            expert_rules = {rid: self.ai_rules[rid] for rid in rule_ids if rid in self.ai_rules}

            task = self._detect_with_specific_rules_async(
                'caption',
                caption_sentences,
                expert_rules,
                expert_name,
                article_date
            )
            expert_tasks.append(task)
            expert_names.append(expert_name)

        if not expert_tasks:
            return []

        logger.info(f"    ⚡ 2개 Caption Expert 병렬 실행...")
        expert_results = await asyncio.gather(*expert_tasks)

        all_caption_detections: List[Dict] = []
        for expert_name, detections in zip(expert_names, expert_results):
            all_caption_detections.extend(detections)
            logger.info(f"    [{expert_name}] {len(detections)}개 위반 검출")

        logger.info(f"    [2-Expert Caption 완료] 총 {len(all_caption_detections)}개 위반 검출")
        return all_caption_detections

    async def _detect_with_specific_rules_async(
        self,
        component_type: str,
        sentences: List[str],
        specific_rules: Dict,
        expert_name: str,
        article_date: Optional[str] = None
    ) -> List[Dict]:
        """특정 규칙들만 사용하여 Detection 수행"""

        # 프롬프트 생성 (특정 규칙들만 포함)
        tools = self._build_detection_tools()
        if self.use_compressed_prompt:
            instructions = self._build_detection_instructions_for_component_compressed(component_type, specific_rules)
        else:
            instructions = self._build_detection_instructions_for_expert(component_type, specific_rules, expert_name)
        input_content = self._build_input_content_for_component(component_type, sentences, article_date)

        # 프롬프트 로깅
        prompt_type = "COMPRESSED" if self.use_compressed_prompt else "STANDARD"
        logger.info(f"      [{expert_name}] {prompt_type} Instructions: {len(instructions)} chars, {len(specific_rules)}개 규칙")

        try:
            # Async API 호출 (재시도 래퍼 적용)
            input_blocks = self._pack_prompt_blocks(instructions, input_content, for_correction=False)
            response = await self.async_client.responses.create(
                model="gpt-5-chat-latest",
                # reasoning={"effort": self.reasoning_effort},
                # text={"verbosity": self.text_verbosity},
                input=input_blocks,
                tools=tools,
                tool_choice="required",
                temperature=0.1
            )
            # Dump prompts (reuse base dumper from parent via logger only)
            try:
                self._dump_prompt(
                    f"detect_expert_instructions_{expert_name}",
                    instructions,
                    {"component": component_type, "rules": len(specific_rules)}
                )
                self._dump_prompt(
                    f"detect_expert_input_{expert_name}",
                    input_content,
                    {"component": component_type}
                )
            except Exception:
                pass

            detections: List[Dict] = []
            for item in response.output:
                if item.type == "function_call" and item.name == "detect_style_violations":
                    function_args = json.loads(item.arguments)
                    detections = function_args.get('violations', [])
                    break

            return detections

        except Exception as e:
            logger.error(f"[{expert_name}] Detection 실패: {str(e)}")
            return []

    def _build_detection_instructions_for_expert(
        self,
        component_type: str,
        expert_rules: Dict,
        expert_name: str
    ) -> str:
        """전문가용 Detection 지시사항 생성 - Balanced precision and recall"""

        # 그룹 정보 및 특별 강조사항
        group_description = ""
        special_focus = ""

        if "Formatting" in expert_name:
            group_description = "You are responsible for checking dates, abbreviations, and number formatting rules."
            special_focus = (
                "\n⚡ **SPECIAL FOCUS**: A04 — Local rebase if absolute date + relative cue; then apply ±7‑day style (past ≤7 days → 'last <weekday>', next ≤7 days → '<weekday>', else 'Month Day'). Assume reference year; avoid ordinals. Report ALL lines in mixed A04 blocks (including the absolute‑date line)."
            )
        elif "Quotation" in expert_name:
            group_description = "You are responsible for checking quotation marks, title capitalization, and Korean-specific notation rules."
            special_focus = "\n⚡ **SPECIAL FOCUS**: Carefully check Title Capitalization (A02) - capitalize titles BEFORE names, lowercase AFTER names. Also check quotation mark placement (A15, A16, A19)."
        elif "Grammar" in expert_name:
            group_description = "You are responsible for checking verb agreement, word choice, and general grammar rules."
            special_focus = "\n⚡ **SPECIAL FOCUS**: Thoroughly check Subject-Verb Agreement (A33) - singular subjects need singular verbs, plural subjects need plural verbs. Review each sentence's subject carefully."

        # 캡션 전용 보조 지시
        if component_type == 'caption' and ('Structure' in expert_name or 'Dates' in expert_name):
            special_focus += "\n- For mixed date blocks (C12), report ALL lines in the block (including the absolute-date line)."
            special_focus += "\n- C19: If 'Courtesy of …' appears and the caption segment has no finite verb, prefer ' / ' before the source (not a period). Do NOT add a verb."

        # 규칙 포맷팅
        rules_text = []
        for rule_id, rule_data in expert_rules.items():
            group = rule_data.get('group', 'General')
            rule_text = f"**{rule_id} ({group})**: {rule_data['description']}"

            # Few-shot 예제 추가
            if rule_data.get('examples'):
                example = rule_data['examples'][0]
                rule_text += f"\n  ✗ Incorrect: '{example.get('incorrect', '')}'"
                rule_text += f"\n  ✓ Correct: '{example.get('correct', '')}'"

            rules_text.append(rule_text)

        rules_section = '\n\n'.join(rules_text)
        article_date = article_date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
        extra = getattr(self, '_extra_instructions', None)
        instructions = f"""You are an expert copy editor for the Korea Times.

{group_description}{special_focus}

TODAY: {article_date}

**TASK: DETECT BODY style guide violations - BE THOROUGH AND ACCURATE.**

**YOUR ASSIGNED RULES ({len(expert_rules)} rules):**

{rules_section}

**DETECTION PRINCIPLES:**

✓ **BE THOROUGH** - Review each sentence carefully against all your assigned rules
✓ **BE ACCURATE** - Report violations you are confident about
✓ **USE EXAMPLES** - Compare sentences to the examples provided in each rule
✓ **BALANCE PRECISION AND RECALL** - Don't over-detect, but don't miss clear violations either

**DETECTION INSTRUCTIONS:**

1. Review EACH sentence against ALL your assigned rules systematically
2. For violations, you should be confident (not necessarily 100%, but reasonably certain)
3. Compare the sentence to the examples - if it matches the violation pattern, report it
4. When reasonably uncertain about a borderline case, err on the side of not reporting

5. For each violation found, report:
   - sentence_id (e.g., "B1", "B2", etc.)
   - rule_id (from your assigned rules above)
   - component ("body")
   - rule_description (brief explanation of the violation)
   - violation_type: **MUST be the exact group name** shown in parentheses after the rule_id

6. If one sentence has multiple violations (within your rules), return multiple entries
{('7. Additionally, follow this instruction strictly: ' + extra) if extra else ''}

**WHAT TO PRIORITIZE:**
✅ Clear pattern matches with the examples
✅ Obvious grammar/style errors
✅ Violations that clearly violate the rule description
✅ Special focus rules mentioned above

**WHAT NOT TO REPORT:**
❌ Violations in rules outside your assigned {len(expert_rules)} rules
❌ Highly ambiguous cases where context might justify the usage
❌ Cases that don't match any example pattern and seem acceptable

**DEFAULT BEHAVIOR:** If no violations are found, return an empty array []

**Remember:** Find ALL clear violations in your assigned rules. Don't be overly cautious - report what violates the rules."""

        return instructions


def main():
    """테스트"""
    styler = AIStylerGPT5_3Experts(reasoning_effort="low", text_verbosity="low")

    test_article = """[TITLE]korea expands childcare support - 50 percent increase[/TITLE][BODY]President Yoon Suk-yeol stated the government will invest 50,000 won ($38) per child. Samsung Electronics chairman Lee Jae-yong attended the ceremony. There are 5 participants. "I'll be there at 9 a.m", she said.[/BODY][CAPTION]President Yoon Suk-yeol poses for a photo. Yonhap.[/CAPTION]"""

    print("="*80)
    print("GPT-5 3-Expert Body Detection Test")
    print("="*80)

    result = styler.correct_article(test_article, "2025-10-18")

    print("\n교정 결과:")
    print(result['corrected_text'])

    print(f"\n\n발견된 위반: {result['stats']['total_violations']}개")
    print(f"  - 정규식: {result['stats']['regex_corrections']}")
    print(f"  - AI: {result['stats']['ai_corrections']}")

    for v in result['violations']:
        if v.sentence_id != "N/A":
            print(f"\n[{v.sentence_id}] {v.rule_id} ({v.component}):")
            print(f"  Original:  {v.original_sentence[:80]}...")
            print(f"  Corrected: {v.corrected_sentence[:80]}...")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
