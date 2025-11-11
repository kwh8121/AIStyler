# backend/app/articles/services/prompt_builder.py
"""
프롬프트 생성 모듈
AI 모델들을 위한 다양한 프롬프트 생성 기능
"""

import logging
from typing import List, Optional, Dict

from ...styleguides.models import StyleGuide

logger = logging.getLogger(__name__)


def compose_system_prompt(base_prompt: str) -> str:
    """시스템 프롬프트 구성"""
    logger.info("📝 Composing system prompt for AI correction")
    logger.debug(f"System prompt total length: {len(base_prompt)} characters")
    logger.debug(f"System prompt preview: {base_prompt[:500]}...")  # 처음 500자 미리보기
    return base_prompt


def generate_correction_prompt(
    style_guides: List[StyleGuide],
    text_to_correct: str,
    additional_prompt: str = None
) -> str:
    """토큰 절약형 교정 프롬프트 생성 (교정된 텍스트만 반환)"""
    # 핵심 지시문 (토큰 절약)
    system_instruction = "Fix text using rules below. Output ONLY corrected text, no explanations."

    # 규칙 정리 (간결화)
    rules = []
    for guide in style_guides:
        # JSON 배열을 문자열로 변환
        description = ' '.join(guide.content) if isinstance(guide.content, list) else guide.content
        # 불필요한 문자 제거 및 간소화
        clean_description = description.replace('["- ', '').replace('"]', '').replace('\\"', '"')
        rules.append(f"R{guide.number}: {clean_description}")

    # 프롬프트 구성 (최소화)
    prompt_parts = [
        system_instruction,
        "",
        "RULES:",
        *rules,
        "",
        f"TEXT: {text_to_correct}",
        "",
        "CORRECTED:"
    ]

    # 추가 프롬프트가 있으면 규칙 섹션에 추가
    if additional_prompt and additional_prompt.strip():
        prompt_parts.insert(-3, f"EXTRA: {additional_prompt.strip()}")
        prompt_parts.insert(-3, "")

    return "\n".join(prompt_parts)


def generate_sentence_level_correction_prompt(
    style_guides: List[StyleGuide],
    sentences: List[str],
    sentence_violations_map: dict,
    additional_prompt: str = None
) -> str:
    """문장별 교정을 위한 프롬프트 생성"""
    # 규칙 정리 (설명과 예시 포함)
    rules = []
    for guide in style_guides:
        # 규칙 설명
        description = ' '.join(guide.content) if isinstance(guide.content, list) else guide.content
        clean_description = description.replace('["- ', '').replace('"]', '').replace('\\"', '"')

        # 규칙 텍스트 구성
        rule_text = f"R{guide.number}: {clean_description}"

        # 예시가 있으면 추가
        if guide.examples_incorrect and guide.examples_correct:
            incorrect_examples = guide.examples_incorrect if isinstance(guide.examples_incorrect, list) else [guide.examples_incorrect]
            correct_examples = guide.examples_correct if isinstance(guide.examples_correct, list) else [guide.examples_correct]

            # 예시가 있는 경우 추가
            if incorrect_examples and correct_examples:
                rule_text += "\n  Examples:"
                for i, (incorrect, correct) in enumerate(zip(incorrect_examples[:2], correct_examples[:2]), 1):  # 최대 2개 예시만
                    if incorrect and correct:
                        rule_text += f"\n    ✗ Incorrect: {incorrect}"
                        rule_text += f"\n    ✓ Correct: {correct}"
                        if i < min(2, len(incorrect_examples)):  # 다음 예시가 있으면 구분선
                            rule_text += "\n"

        rules.append(rule_text)

    # 문장별 violations 정보 생성
    sentence_info = []
    for idx in range(len(sentences)):
        violations = sentence_violations_map.get(idx, {}).get("violations", [])
        if violations:  # violations이 있는 경우만 추가
            violation_rules = [v.split('_')[-1] for v in violations]  # articles_SG013 -> SG013
            sentence_info.append(f"Sentence {idx+1}: Violates rules {', '.join(violation_rules)}")

    prompt_parts = [
        "You are a professional text editor specializing in style guide compliance.",
        "Your task is to correct the following text according to the specific style guide rules that were violated.",
        "",
        "Instructions:",
        "1. Apply ALL the style guide rules listed below to correct the text",
        "2. Pay special attention to the sentences that have specific violations noted",
        "3. Return ONLY the corrected text without any JSON formatting, explanations, or metadata",
        "4. Maintain the original paragraph structure and spacing",
        "5. Use the examples above when available: if a rule shows '✗ Incorrect'/'✓ Correct' pairs, rewrite to match the '✓ Correct' pattern",
        "6. Prefer the minimal edits needed to satisfy the rules and examples; do not paraphrase unrelated content",
        "",
        "STYLE GUIDE RULES TO APPLY:",
        *rules,
        "",
    ]

    # 위반 정보가 있으면 추가
    if sentence_info:
        prompt_parts.extend([
            "SPECIFIC VIOLATIONS TO FIX:",
            *sentence_info,
            ""
        ])

    prompt_parts.extend([
        "TEXT TO CORRECT:",
        " ".join(sentences),  # 문장들을 하나의 텍스트로 합침
        ""
    ])

    if additional_prompt and additional_prompt.strip():
        prompt_parts.extend([
            f"ADDITIONAL INSTRUCTION: {additional_prompt.strip()}",
            ""
        ])

    prompt_parts.append("CORRECTED TEXT:")

    return "\n".join(prompt_parts)


def generate_openai_style_analysis_prompt(
    style_guides: List[StyleGuide],
    category: str
) -> str:
    """OpenAI를 위한 스타일가이드 분석 프롬프트 생성"""

    # 카테고리별 매핑
    category_map = {
        "TITLE": "headlines",
        "BODY": "articles",
        "CAPTION": "captions"
    }
    json_category = category_map.get(category, "articles")

    # 스타일가이드를 구조화된 형태로 정리
    rules_text = []
    for guide in style_guides:
        rule_id = f"{json_category}_SG{guide.number:03d}"
        description = ' '.join(guide.content) if isinstance(guide.content, list) else str(guide.content or guide.docs)
        clean_description = description.replace('["- ', '').replace('"]', '').replace('\\"', '"')

        rule_entry = f"{rule_id}: {clean_description}"

        # 예시 추가 (있는 경우)
        if guide.examples_incorrect and guide.examples_correct:
            incorrect = guide.examples_incorrect[0] if isinstance(guide.examples_incorrect, list) else guide.examples_incorrect
            correct = guide.examples_correct[0] if isinstance(guide.examples_correct, list) else guide.examples_correct
            if incorrect and correct:
                rule_entry += f"\n  Example: '{incorrect}' → '{correct}'"

        rules_text.append(rule_entry)

    # 사용자가 제공한 시스템 프롬프트 사용
    prompt = f"""You are a style guide classifier for English news articles.
Analyze the entire text carefully and classify it according to the appropriate style guide number.
Output ONLY the style guide code in the format: [category]_SG[number] (e.g., headlines_SG01, body_SG02, quotes_SG03).

Category: {json_category}

Style Guide Rules:
{chr(10).join(rules_text)}

Instructions:
1. Check EVERY sentence against ALL style guide rules
2. Identify ALL violations
3. Return ONLY the violations in JSON format:

{{
  "violations": [
    {{
      "rule_id": "{json_category}_SG001",
      "sentence_index": 0
    }}
  ],
  "total_violations": 0
}}

If no violations are found, return:
{{"violations": [], "total_violations": 0}}

Return ONLY the JSON response."""

    return prompt


def generate_openai_correction_prompt(
    text: str,
    violations: List[Dict],
    style_guides: List[StyleGuide],
    additional_prompt: Optional[str] = None
) -> str:
    """OpenAI를 위한 교정 프롬프트 생성 (위반사항 기반)"""

    # 위반된 규칙들만 추출
    violated_rule_ids = set(v['rule_id'] for v in violations if 'rule_id' in v)

    # 해당 규칙들만 포함 (가능하면 예시 포함)
    relevant_rules = []
    for guide in style_guides:
        # 규칙 ID 생성 (category는 violated_rule_ids에서 추출)
        for rule_id in violated_rule_ids:
            if f"SG{guide.number:03d}" in rule_id:
                description = ' '.join(guide.content) if isinstance(guide.content, list) else str(guide.content or guide.docs)
                clean_description = description.replace('["- ', '').replace('"]', '').replace('\\"', '"')
                rule_text = f"Rule {guide.number}: {clean_description}"
                # 예시가 있으면 첫 번째 예시를 함께 제공
                try:
                    incorrect = guide.examples_incorrect[0] if isinstance(guide.examples_incorrect, list) else guide.examples_incorrect
                    correct = guide.examples_correct[0] if isinstance(guide.examples_correct, list) else guide.examples_correct
                    if incorrect and correct:
                        rule_text += f"\n  ✗ Incorrect: {incorrect}\n  ✓ Correct: {correct}"
                except Exception:
                    pass
                relevant_rules.append(rule_text)
                break

    prompt_parts = [
        "Correct the following text by fixing ONLY the identified style guide violations.",
        "",
        "Violations to fix:",
    ]

    # 위반 사항 요약 (간소화)
    for v in violations[:10]:  # 최대 10개만 표시
        sentence_idx = v.get('sentence_index', '?')
        rule = v.get('rule_id', 'unknown')
        prompt_parts.append(f"- Sentence {sentence_idx + 1}: {rule}")

    if len(violations) > 10:
        prompt_parts.append(f"... and {len(violations) - 10} more violations")

    prompt_parts.extend([
        "",
        "Relevant Style Guide Rules:",
        *relevant_rules,
        "",
        "Instructions:",
        "1. Fix ONLY the violations listed above",
        "2. Do NOT change anything else in the text",
        "3. Maintain the original structure and flow",
        "4. Return ONLY the corrected text without explanations",
        "5. When a rule includes examples, follow them strictly: mirror the '✓ Correct' pattern shown above; do not invent new formats or styles",
        "6. Prefer the minimal edit that satisfies the rule and matches the example",
        "",
        "Original Text:",
        text,
    ])

    if additional_prompt:
        prompt_parts.extend([
            "",
            f"Additional instruction: {additional_prompt}",
        ])

    prompt_parts.extend([
        "",
        "Corrected Text:"
    ])

    return "\n".join(prompt_parts)
