# backend/app/articles/service.py
import time
import asyncio
import httpx
import pysbd
import json
import re
import logging
import uuid
from typing import Optional, AsyncGenerator, List
from sqlalchemy import select, func, delete, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from ..config import settings
from ..users.models import User
from .models import Article, TextCorrectionHistory, ArticlePrompt, ArticleCategory, ArticleStatus, OperationType
from ..styleguides.models import StyleGuide, TextCorrectionHistoryStyle
from ..styleguides import service as style_guide_service
from .services.openai_correction import call_openai_correction_stream, dump_prompt

logger = logging.getLogger(__name__)

def map_category_to_ai_server(category: ArticleCategory) -> str:
    """ArticleCategory enum을 AI 서버 category 문자열로 변환"""
    mapping = {
        ArticleCategory.SEO: "headlines",
        ArticleCategory.TITLE: "headlines",
        ArticleCategory.BODY: "articles", 
        ArticleCategory.CAPTION: "captions",
    }
    return mapping.get(category, "articles")

async def get_prompt(db: AsyncSession, *, category: ArticleCategory, override: str | None) -> str:
    # 추가 프롬프트가 실제 내용을 포함하는지 확인
    if override and override.strip():
        logger.info(f"Using override prompt for category {category}")
        logger.debug(f"Override prompt content: {override[:200]}...")  # 처음 200자만 로그
        return override.strip()
    
    # Enum을 문자열로 변환하여 비교
    category_str = category.value if isinstance(category, ArticleCategory) else category
    res = await db.execute(
        select(ArticlePrompt.prompt).where(ArticlePrompt.category == category_str)
    )
    p = res.scalar_one_or_none()
    
    if p:
        logger.info(f"✅ Loaded prompt from DB for category: {category_str}")
        logger.debug(f"DB prompt length: {len(p)} characters")
        logger.debug(f"DB prompt preview: {p[:300]}...")  # 처음 300자 미리보기
    else:
        logger.warning(f"⚠️ No prompt found in DB for category: {category_str}, using default")
    
    # 비어있다면 아래 기본 프롬프트
    default_prompt = f"Correct the following {category_str.lower()} text in English with journalistic style."
    return p or default_prompt

async def translate_to_en(text: str) -> str:
    """한국어 텍스트를 영어로 번역 (프로바이더는 env로 선택)"""
    translated, detected_source, target_lang = await translate_text(
        text, source_lang="KO", target_lang="EN-US"
    )
    return translated, detected_source, target_lang

async def translate_text(
    text: str,
    source_lang: Optional[str] = None,
    target_lang: str = "EN-US",
) -> tuple[str, str, str]:
    """범용 텍스트 번역 함수 (translation 모듈 위임)"""
    from .services.translation import translate_text as _translate
    return await _translate(text=text, source_lang=source_lang, target_lang=target_lang)

def _compose_system_prompt(base_prompt: str) -> str:
    logger.info("📝 Composing system prompt for AI correction")
    logger.debug(f"System prompt total length: {len(base_prompt)} characters")
    logger.debug(f"System prompt preview: {base_prompt[:500]}...")  # 처음 500자 미리보기
    return base_prompt

async def call_ai_server(text: str, category: ArticleCategory) -> dict:
    """AI 서버의 analyze API 호출 (단일 텍스트)
    
    Args:
        text: 분석할 텍스트
        category: 문서 카테고리 (TITLE, BODY, CAPTION)
        
    Returns:
        표준화된 응답 딕셔너리 (기존 코드 호환성 유지)
        
    Raises:
        Exception: AI 서버 호출 실패 시
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.AI_SERVER_URL}/analyze",
                json={
                    "text": text,  # 새 API는 "text" 필드 사용
                    "category": map_category_to_ai_server(category)
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            raw_response = response.json()
            
            # 새로운 응답 형식을 기존 코드가 기대하는 형식으로 변환
            return _convert_single_response(raw_response)
            
    except httpx.TimeoutException:
        logger.error(f"AI Server timeout after 60 seconds")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"AI Server HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"AI Server error: {e}")
        raise

def _convert_single_response(raw_response: dict) -> dict:
    """새로운 AI 서버 응답을 기존 코드 호환 형식으로 변환
    
    Args:
        raw_response: AI 서버의 새로운 형식 응답
        
    Returns:
        기존 코드가 기대하는 형식의 응답
    """
    # 응답 검증
    if not isinstance(raw_response, dict):
        logger.error(f"Invalid response type: {type(raw_response)}")
        return {"success": False, "message": "Invalid response format"}
    
    violations = raw_response.get("violations", [])
    confidence = raw_response.get("confidence", 0.0)
    adapter_version = raw_response.get("adapter_version", "unknown")
    
    # violations 검증
    if not isinstance(violations, list):
        logger.warning(f"Invalid violations format: {type(violations)}, converting to list")
        violations = []
    
    # 빈 violations 처리
    if not violations:
        logger.info("No violations found in AI server response")
        return {
            "success": True,
            "result": {
                "applicable_rules": [],
                "original_text": raw_response.get("text", ""),
                "category": raw_response.get("category", ""),
                "confidence": confidence,
                "adapter_version": adapter_version
            }
        }
    
    # 신뢰도 검증 및 경고
    try:
        confidence = float(confidence)
        if confidence < 0.7:
            logger.warning(f"Low confidence analysis: {confidence:.2f}")
        elif confidence > 1.0:
            logger.warning(f"Suspicious confidence value: {confidence}")
    except (ValueError, TypeError):
        logger.warning(f"Invalid confidence value: {confidence}, defaulting to 0.0")
        confidence = 0.0
    
    # 모델 버전 로깅
    logger.debug(f"AI Server adapter version: {adapter_version}")
    
    # violations 형식 검증
    valid_violations = []
    for violation in violations:
        if isinstance(violation, str) and violation.strip():
            valid_violations.append(violation.strip())
        else:
            logger.warning(f"Invalid violation format: {violation}")
    
    # 기존 형식으로 변환
    converted = {
        "success": True,
        "result": {
            "applicable_rules": valid_violations,
            "original_text": raw_response.get("text", ""),
            "category": raw_response.get("category", ""),
            "confidence": confidence,
            "adapter_version": adapter_version
        }
    }
    
    logger.info(f"Converted response: {len(valid_violations)} violations found (confidence: {confidence:.2f})")
    
    return converted

async def call_ai_server_batch(sentences: List[str], category: ArticleCategory) -> dict:
    """AI 서버의 analyze-batch API 호출 (여러 문장 배치 처리)
    
    Args:
        sentences: 분석할 문장들의 리스트
        category: 문서 카테고리 (모든 문장에 동일하게 적용)
        
    Returns:
        AI 서버 응답 딕셔너리 (병합된 분석 결과)
        
    Raises:
        Exception: AI 서버 호출 실패 시
    """
    if not sentences:
        logger.warning("Empty sentences list provided to batch API")
        return {"success": False, "message": "No sentences to analyze"}
    
    # 빈 문장 제거
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    if not valid_sentences:
        logger.warning("No valid sentences after filtering")
        return {"success": False, "message": "No valid sentences to analyze"}
    
    try:
        # Batch API 형식으로 데이터 구성 (올바른 키 사용)
        batch_data = {
            "items": [
                {"text": sentence, "category": map_category_to_ai_server(category)}
                for sentence in valid_sentences
            ]
        }
        
        logger.info(f"Sending {len(valid_sentences)} sentences to batch API")
        
        async with httpx.AsyncClient(timeout=120.0) as client:  # 배치 처리는 더 긴 timeout
            response = await client.post(
                f"{settings.AI_SERVER_URL}/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            batch_response = response.json()
            
            # 배치 응답을 단일 응답 형태로 병합
            return _merge_batch_response(batch_response, valid_sentences)
            
    except httpx.TimeoutException:
        logger.error(f"AI Server batch timeout after 120 seconds")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"AI Server batch HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"AI Server batch error: {e}")
        raise

def _merge_batch_response(batch_response: dict, sentences: List[str]) -> dict:
    """배치 API 응답을 단일 응답 형태로 병합 (새로운 형식 지원)
    
    Args:
        batch_response: AI 서버의 배치 응답 (새로운 형식)
        sentences: 원본 문장 리스트
        
    Returns:
        병합된 응답 딕셔너리 (기존 코드 호환 형식)
    """
    # 새로운 형식에서는 results 배열에 직접 응답이 들어있음
    results = batch_response.get("results", [])
    
    if not results:
        return {"success": False, "message": "No results in batch response"}
    
    # 모든 문장의 violations 수집
    all_violations = set()
    confidence_scores = []
    adapter_versions = set()
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            # 새로운 형식에서 violations 추출
            violations = result.get("violations", [])
            
            # violations 검증
            if isinstance(violations, list):
                # 유효한 violation만 필터링
                valid_violations = [v for v in violations if isinstance(v, str) and v.strip()]
                all_violations.update(valid_violations)
                
                if len(valid_violations) != len(violations):
                    logger.warning(f"Sentence {i+1}: filtered out {len(violations) - len(valid_violations)} invalid violations")
            else:
                logger.warning(f"Sentence {i+1}: invalid violations format: {type(violations)}")
            
            # 통계 정보 수집
            confidence = result.get("confidence", 0.0)
            try:
                confidence = float(confidence)
                confidence_scores.append(confidence)
                
                # 낮은 신뢰도 경고
                if confidence < 0.7:
                    logger.warning(f"Sentence {i+1} has low confidence: {confidence:.2f}")
                elif confidence > 1.0:
                    logger.warning(f"Sentence {i+1} has suspicious confidence: {confidence:.2f}")
            except (ValueError, TypeError):
                logger.warning(f"Sentence {i+1}: invalid confidence value: {confidence}")
                confidence_scores.append(0.0)
            
            adapter_version = result.get("adapter_version", "unknown")
            if adapter_version and isinstance(adapter_version, str):
                adapter_versions.add(adapter_version)
        else:
            logger.warning(f"Sentence {i+1}: invalid result format: {type(result)}")
    
    # 평균 신뢰도 계산
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    # 어댑터 버전 로깅
    if adapter_versions:
        logger.debug(f"Batch processing adapter versions: {adapter_versions}")
    
    # 기존 형식으로 변환
    merged_result = {
        "success": True,
        "result": {
            "applicable_rules": sorted(list(all_violations)),
            "total_sentences": len(sentences),
            "processed_sentences": len(results),
            "avg_confidence": avg_confidence,
            "adapter_versions": list(adapter_versions)
        }
    }
    
    logger.info(f"Merged batch response: {len(all_violations)} unique violations from {len(results)} sentences (avg confidence: {avg_confidence:.2f})")
    
    return merged_result

async def call_openai_stream_native(prompt: str, text: str) -> AsyncGenerator[str, None]:
    """OpenAI Async API를 사용한 네이티브 스트리밍 구현"""
    try:
        from openai import AsyncOpenAI
        import time
        
        start_time = time.time()
        collected_text = []
        
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        sys_prompt = _compose_system_prompt(prompt)
        
        # OpenAI 요청 파라미터 로깅
        openai_input = f"{sys_prompt}\n\nText to correct:\n{text}"
        logger.info(f"📡 OpenAI API: Model={settings.OPENAI_MODEL}, Input={len(openai_input)} chars")
        
        # 상세 정보는 디버그 레벨로
        logger.debug("="*80)
        logger.debug("🚀 OpenAI API Request Details:")
        if hasattr(settings, 'OPENAI_REASONING_EFFORT') and settings.OPENAI_REASONING_EFFORT:
            logger.debug(f"Reasoning Effort: {settings.OPENAI_REASONING_EFFORT}")
        logger.debug(f"System Prompt Length: {len(sys_prompt)} characters")
        logger.debug(f"Text Length: {len(text)} characters")
        logger.debug("-"*40)
        logger.debug(f"Full Input (first 2000 chars):\n{openai_input[:2000]}...")
        if len(openai_input) > 2000:
            logger.debug(f"... [truncated {len(openai_input) - 2000} more characters]")
        logger.debug("="*80)
        
        # API 요청 파라미터 구성
        request_params = {
            "model": settings.OPENAI_MODEL,
            "input": openai_input,
            "stream": True,
            "temperature": 0.1
        }
        
        # reasoning 파라미터가 환경변수에 있고 비어있지 않은 경우에만 추가
        if hasattr(settings, 'OPENAI_REASONING_EFFORT') and settings.OPENAI_REASONING_EFFORT:
            request_params["reasoning"] = {
                "effort": settings.OPENAI_REASONING_EFFORT
            }
            logger.info(f"Using reasoning effort: {settings.OPENAI_REASONING_EFFORT}")
        else:
            logger.info("Reasoning effort not configured, skipping reasoning parameter")
        
        stream = await client.responses.create(**request_params)
        
        event_count = 0
        async for event in stream:
            event_count += 1
            current_time = time.time() - start_time
            
            # Delta 이벤트 처리
            if hasattr(event, 'type') and event.type == 'response.output_text.delta':
                delta_text = getattr(event, 'delta', '') or ""
                
                if delta_text:
                    collected_text.append(delta_text)
                    yield json.dumps({
                        "choices": [{
                            "delta": {"content": delta_text}
                        }]
                    })
                    logger.debug(f"Streamed chunk {event_count} at {current_time:.3f}s")
                    
            elif hasattr(event, 'type') and event.type in ['response.done', 'response.completed']:
                full_text = "".join(collected_text)
                logger.info(f"✅ OpenAI Response: {current_time:.3f}s, {event_count} events, {len(full_text)} chars")
                
                # 상세 응답은 디버그 레벨로
                logger.debug("-"*40)
                logger.debug(f"Full Response (first 3000 chars):\n{full_text[:3000]}...")
                if len(full_text) > 3000:
                    logger.debug(f"... [truncated {len(full_text) - 3000} more characters]")
                logger.debug("="*80)
                    
    except ImportError:
        logger.error("OpenAI package not installed")
        raise
    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        yield json.dumps({
            "choices": [{
                "delta": {"content": f"Error: {str(e)}"}
            }]
        })

def _generate_correction_prompt(style_guides: List[StyleGuide], text_to_correct: str, additional_prompt: str = None) -> str:
    """
    토큰 절약형 교정 프롬프트 생성 (교정된 텍스트만 반환)
    """
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

def _generate_sentence_level_correction_prompt(
    style_guides: List[StyleGuide], 
    sentences: List[str], 
    sentence_violations_map: dict, 
    additional_prompt: str = None
) -> str:
    """
    문장별 교정을 위한 프롬프트 생성 (JSON 형식 응답 요청)
    """
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
    
    # 문장별 violations 정보 생성 (메모리 효율성을 위해 문장 길이 제한)
    sentence_info = []
    for idx, sentence in enumerate(sentences):
        # 매우 긴 문장은 truncate하여 메모리 사용량 제한
        truncated_sentence = sentence[:500] + "..." if len(sentence) > 500 else sentence
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

def _build_full_text_from_corrections(sentence_corrections: dict, original_sentences: List[str]) -> str:
    """
    문장별 교정 결과를 하나의 전체 텍스트로 합치기
    
    Args:
        sentence_corrections: {index: {"original": str, "corrected": str, "violations": list}}
        original_sentences: 원본 문장 리스트 (spacing/paragraph 정보 유지용)
    
    Returns:
        str: 합쳐진 전체 교정 텍스트
    """
    try:
        # 원본 텍스트에서 문장 간 구분자 추출 (공백, 줄바꿈 등)
        # pysbd는 문장 분리 시 원본 spacing을 제거하므로, 원본에서 패턴 추출 필요
        
        # 인덱스 기준으로 정렬된 교정된 문장들 수집
        corrected_sentences = []
        max_index = max(len(original_sentences) - 1, max(sentence_corrections.keys()) if sentence_corrections else 0)
        
        for i in range(max_index + 1):
            if i in sentence_corrections:
                corrected = sentence_corrections[i].get("corrected", "")
                if corrected:
                    corrected_sentences.append(corrected)
                    logger.debug(f"Sentence {i}: Using corrected version")
                else:
                    # corrected가 비어있으면 원본 사용
                    if i < len(original_sentences):
                        corrected_sentences.append(original_sentences[i])
                        logger.warning(f"Sentence {i}: Corrected is empty, using original")
                    else:
                        logger.warning(f"Sentence {i}: Index out of range, skipping")
            else:
                # 해당 인덱스의 교정이 없으면 원본 사용
                if i < len(original_sentences):
                    corrected_sentences.append(original_sentences[i])
                    logger.debug(f"Sentence {i}: No correction found, using original")
                else:
                    logger.warning(f"Sentence {i}: No correction and index out of range, skipping")
        
        # 문장들을 공백으로 연결 (기본적으로 단일 공백)
        full_text = " ".join(corrected_sentences)
        
        logger.info(f"Built full_text from {len(corrected_sentences)} sentences (total length: {len(full_text)} chars)")
        return full_text
        
    except Exception as e:
        logger.error(f"Error building full_text from corrections: {e}")
        # 에러 시 원본 문장들을 그대로 연결
        return " ".join(original_sentences)

def _parse_sentence_corrections(openai_response: str, original_sentences: List[str], sentence_violations_map: dict) -> tuple:
    """
    OpenAI 응답에서 문장별 교정 정보를 파싱하고 전체 텍스트 생성
    
    Args:
        openai_response: OpenAI에서 받은 전체 응답 텍스트
        original_sentences: 원본 문장 리스트
        sentence_violations_map: 문장별 violations 정보
        
    Returns:
        tuple: (sentence_corrections dict, full_text str)
    """
    sentence_corrections = {}
    
    try:
        response_length = len(openai_response)
        logger.info(f"Parsing OpenAI response (length: {response_length} chars)")
        
        # JSON 응답에서 corrected_sentences 추출 시도
        if "{" in openai_response and "}" in openai_response:
            # JSON 부분만 추출
            json_start = openai_response.find("{")
            json_end = openai_response.rfind("}") + 1
            json_part = openai_response[json_start:json_end]
            
            logger.debug(f"Extracted JSON part (length: {len(json_part)})")
            
            try:
                parsed_response = json.loads(json_part)
                corrected_sentences = parsed_response.get("corrected_sentences", [])
                
                if not corrected_sentences:
                    logger.warning("No corrected_sentences found in JSON response")
                    # Try alternative field names
                    for alt_field in ["sentences", "corrections", "results"]:
                        if alt_field in parsed_response:
                            corrected_sentences = parsed_response[alt_field]
                            logger.info(f"Found corrections in alternative field: {alt_field}")
                            break
                
                logger.info(f"Processing {len(corrected_sentences)} sentence corrections")
                
                for i, correction in enumerate(corrected_sentences):
                    if isinstance(correction, dict):
                        idx = correction.get("index")
                        if idx is None:
                            # Try alternative index field names
                            idx = correction.get("sentence_index", correction.get("id", i))
                        
                        if idx is not None and isinstance(idx, int):
                            original = correction.get("original", correction.get("before", ""))
                            corrected = correction.get("corrected", correction.get("after", ""))
                            
                            if not original and not corrected:
                                logger.warning(f"Both original and corrected text are empty for index {idx}")
                            
                            sentence_corrections[idx] = {
                                "original": original,
                                "corrected": corrected,
                                "violations": sentence_violations_map.get(idx, {}).get("violations", [])
                            }
                            logger.debug(f"Added correction for sentence {idx}: '{original[:50]}...' -> '{corrected[:50]}...'")
                        else:
                            logger.warning(f"Invalid or missing index for correction {i}: {correction}")
                    else:
                        logger.warning(f"Correction {i} is not a dictionary: {type(correction)}")
                            
                logger.info(f"Successfully parsed JSON response with {len(sentence_corrections)} corrections")
                
                # Build full_text from corrections
                full_text = _build_full_text_from_corrections(sentence_corrections, original_sentences)
                logger.info(f"Built full_text from corrections: {len(full_text)} chars")
                
                return sentence_corrections, full_text
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse OpenAI response as JSON: {e}")
                logger.debug(f"JSON parse failed for: {json_part[:200]}...")
        else:
            logger.warning("No JSON structure found in OpenAI response, falling back to text parsing")
    
    except Exception as e:
        logger.warning(f"JSON parsing failed: {e}, falling back to text parsing")
    
    # Fallback: 텍스트 기반 파싱 또는 원본 문장 사용
    logger.info("Using fallback: mapping original sentences to corrected text")
    
    # 응답을 문장 단위로 분리해서 매핑 시도
    response_lines = [line.strip() for line in openai_response.split('\n') if line.strip()]
    
    for idx, original_sentence in enumerate(original_sentences):
        # 기본적으로 원본 문장을 그대로 사용 (교정이 필요없을 수도 있음)
        corrected_sentence = original_sentence
        
        # 응답에서 해당 문장과 유사한 교정된 문장 찾기 시도
        if idx < len(response_lines):
            corrected_sentence = response_lines[idx]
        
        sentence_corrections[idx] = {
            "original": sentence_violations_map.get(idx, {}).get("text", original_sentence),
            "corrected": corrected_sentence,
            "violations": sentence_violations_map.get(idx, {}).get("violations", [])
        }
    
    logger.info(f"Fallback parsing completed with {len(sentence_corrections)} corrections")
    
    # Build full_text from fallback corrections
    full_text = _build_full_text_from_corrections(sentence_corrections, original_sentences)
    logger.info(f"Built full_text from fallback corrections: {len(full_text)} chars")
    
    return sentence_corrections, full_text

async def call_ai_correction_stream(
    prompt: str,
    text: str,
    category: ArticleCategory,
    db: AsyncSession
) -> AsyncGenerator[str, None]:
    """
    AI 텍스트 교정을 위한 기본 스트리밍 파이프라인.

    1. 텍스트를 문장 단위로 분리하고 AI 서버 배치 분석으로 위반 규칙 탐지
    2. 위반 규칙 정보를 바탕으로 스타일 가이드 상세 정보를 DB에서 조회
    3. OpenAI 스트리밍으로 실제 교정을 수행 (필요 시 DeepL 번역 정보 포함)
    4. 실패 시 단일 텍스트 분석 또는 OpenAI 단독 스트리밍으로 폴백
    """
    # 1. AI Server (분석기 역할)
    if settings.USE_AI_SERVER and settings.AI_SERVER_URL:
        try:
            # Step 1: 텍스트를 문장 단위로 분리
            logger.info("🤖 Step 1: Segmenting text into sentences...")
            seg = pysbd.Segmenter(language="en", clean=False)
            sentences = seg.segment(text)
            logger.debug(f"Segmented sentences: {sentences}")
            logger.info(f"Segmented into {len(sentences)} sentences")
            
            # Step 2: AI 서버에서 배치 분석
            ai_server_start = time.time()
            logger.info("🤖 Step 2: Analyzing sentences with AI Server (batch)...")
            
            # batch API 직접 호출해서 원본 응답 보존
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    valid_sentences = [s.strip() for s in sentences if s.strip()]
                    batch_data = {
                        "items": [
                            {"text": sentence, "category": map_category_to_ai_server(category)}
                            for sentence in valid_sentences
                        ]
                    }
                    
                    response = await client.post(
                        f"{settings.AI_SERVER_URL}/batch",
                        json=batch_data,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    batch_response = response.json()
                    
                    # 문장별 violations 정보 저장
                    sentence_violations_map = {}
                    all_violations = set()
                    
                    results = batch_response.get("results", [])
                    for idx, result in enumerate(results):
                        if isinstance(result, dict):
                            violations = result.get("violations", [])
                            sentence_violations_map[idx] = {
                                "text": result.get("text", sentences[idx] if idx < len(sentences) else ""),
                                "violations": violations,
                                "confidence": result.get("confidence", 0.0)
                            }
                            all_violations.update(violations)
                    
                    applicable_rules = list(all_violations)
                    
            except Exception as e:
                logger.error(f"Batch API error: {e}")
                raise
                
            ai_server_time = time.time() - ai_server_start
            
            if not applicable_rules:
                logger.info("No applicable style rules found. Returning original text.")
                for char in text:
                    yield json.dumps({"choices": [{"delta": {"content": char}}]})
                return
            
            logger.info(f"✅ AI Server batch analysis completed in {ai_server_time:.3f}s. Found rules: {applicable_rules}")
            logger.info(f"Processed {len(results)}/{len(sentences)} sentences with violations map: {len(sentence_violations_map)}")

            # Step 3: 서비스 레이어를 통해 DB에서 규칙 상세 정보 조회
            db_lookup_start = time.time()
            logger.info("🤖 Step 3: Fetching style guide details via service...")
            style_guides_details = await style_guide_service.get_guides_by_applicable_rules(
                db, rule_ids=applicable_rules
            )
            db_lookup_time = time.time() - db_lookup_start
            
            if not style_guides_details:
                raise ValueError(f"Could not find DB details for rules {applicable_rules}.")
            
            logger.info(f"✅ DB lookup completed in {db_lookup_time:.3f}s. Found {len(style_guides_details)} guide(s).")
             
            # Step 4: 분석 결과 JSON을 첫 번째로 yield
            logger.info("🤖 Step 4: Yielding analysis result...")
            analysis_data = {
                "applicable_rules": applicable_rules,
                "style_guide_violations": [
                    {
                        "id": f"{g.category}_SG{g.number:03d}", # 파싱 가능한 ID 형식으로
                        "category": g.category,
                        "number": g.number,
                        "description": ' '.join(g.content) if isinstance(g.content, list) else g.content
                    }
                    for g in style_guides_details
                ],
                "style_ids": [g.id for g in style_guides_details],  # DB primary key IDs for history storage
                "sentence_violations": sentence_violations_map,  # 문장별 violations 정보 추가
                "batch_info": {
                    "total_sentences": len(sentences),
                    "processed_sentences": len(results),
                    "avg_confidence": sum(s.get("confidence", 0.0) for s in sentence_violations_map.values()) / len(sentence_violations_map) if sentence_violations_map else 0.0
                }
            }
            yield json.dumps({"type": "analysis", "data": analysis_data})

            # Step 5: OpenAI에 보낼 문장별 교정 프롬프트 생성
            logger.info("🤖 Step 5: Generating sentence-level correction prompt for OpenAI.")
            final_openai_prompt = _generate_sentence_level_correction_prompt(
                style_guides_details, sentences, sentence_violations_map, prompt
            )
            
            # 생성된 프롬프트 로깅 (디버그 레벨)
            logger.debug("="*80)
            logger.debug("📝 Generated Sentence-Level Prompt for OpenAI:")
            logger.debug(f"Prompt Length: {len(final_openai_prompt)} characters")
            logger.debug(f"Number of Sentences: {len(sentences)}")
            logger.debug(f"Number of Style Guides: {len(style_guides_details)}")
            logger.debug("-"*40)
            logger.debug(f"Full Prompt (first 2500 chars):\n{final_openai_prompt[:2500]}...")
            if len(final_openai_prompt) > 2500:
                logger.debug(f"... [truncated {len(final_openai_prompt) - 2500} more characters]")
            logger.debug("="*80)
            
            # OpenAI API 시간 측정
            openai_start = time.time()
            logger.info("🤖 Starting OpenAI API correction stream (REAL STREAMING)...")

            # Step 5: OpenAI 스트림을 즉시 전달하면서 동시에 수집
            collected_chunks = []
            chunk_count = 0

            # 프롬프트 파일 덤프 (디버깅/검증용)
            try:
                dump_prompt(
                    "correction_sentence_level",
                    final_openai_prompt,
                    {
                        "category": category.value if hasattr(category, "value") else str(category),
                        "style_guides": [
                            {
                                "id": g.id,
                                "number": g.number,
                                "category": str(getattr(g, "category", ""))
                            }
                            for g in style_guides_details
                        ],
                        "sentences": len(sentences),
                    },
                )
            except Exception:
                pass
            
            async for chunk_str in call_openai_stream_native(final_openai_prompt, ""):
                try:
                    delta_chunk = json.loads(chunk_str)
                    
                    # 즉시 프론트엔드로 스트리밍
                    yield json.dumps({
                        "type": "delta",
                        "data": delta_chunk
                    })
                    
                    # 동시에 내용 수집 (나중 처리를 위해)
                    choices = delta_chunk.get("choices", [])
                    if choices and "delta" in choices[0] and "content" in choices[0]["delta"]:
                        content = choices[0]["delta"]["content"]
                        collected_chunks.append(content)
                        chunk_count += 1
                        
                except json.JSONDecodeError:
                    pass
            
            openai_time = time.time() - openai_start
            full_openai_response = "".join(collected_chunks).strip()
            
            # 로깅 간소화
            logger.info(f"✅ OpenAI streaming completed: {openai_time:.3f}s, {chunk_count} chunks, {len(full_openai_response)} chars")
            
            # Step 6: 스트리밍 완료 후 처리
            # OpenAI 응답은 교정된 전체 텍스트
            final_text = full_openai_response
            
            # 교정된 텍스트를 문장으로 분리하여 원본과 매핑
            corrected_seg = pysbd.Segmenter(language="en", clean=False)
            corrected_sentences = corrected_seg.segment(final_text)
            
            # 문장별 교정 정보 재구성 (AI 서버의 violations 정보 활용)
            sentence_corrections = {}
            for idx in range(min(len(sentences), len(corrected_sentences))):
                sentence_corrections[idx] = {
                    "original": sentence_violations_map.get(idx, {}).get("text", sentences[idx] if idx < len(sentences) else ""),
                    "corrected": corrected_sentences[idx] if idx < len(corrected_sentences) else "",
                    "violations": sentence_violations_map.get(idx, {}).get("violations", [])
                }
            
            # 스트리밍 완료 메타데이터 전송
            logger.info(f"Parsed {len(sentence_corrections)} sentence corrections")
            
            # 문장별 교정 정보 전송
            yield json.dumps({
                "type": "sentence_corrections", 
                "data": {
                    "sentence_corrections": sentence_corrections,
                    "total_sentences": len(sentences),
                    "corrected_sentences": len(sentence_corrections),
                    "full_text": final_text or full_openai_response  # 파싱된 텍스트 또는 원본
                }
            })
            
            # Step 7: 최종 분석 결과 생성 (DB 저장용 데이터 포함)
            final_analysis_result = analysis_data.copy()
            final_analysis_result["sentence_corrections"] = sentence_corrections
            final_analysis_result["full_text"] = final_text or full_openai_response
            
            yield json.dumps({"type": "final_analysis", "data": final_analysis_result})
            
            
            return

        except Exception as e:
            logger.warning(f"AI Server batch flow failed: {e}. Trying single sentence fallback...")
            
            # Fallback: 단일 텍스트로 처리
            try:
                logger.info("🤖 Fallback: Using single text analysis...")
                analysis_response = await call_ai_server(text, category)
                
                if analysis_response.get("success"):
                    result = analysis_response.get("result", {})
                    applicable_rules = result.get("applicable_rules", [])
                    
                    if applicable_rules:
                        logger.info(f"✅ Single text analysis successful. Found rules: {applicable_rules}")
                        
                        # DB 조회 및 OpenAI 처리 (동일한 로직)
                        style_guides_details = await style_guide_service.get_guides_by_applicable_rules(
                            db, rule_ids=applicable_rules
                        )
                        
                        if style_guides_details:
                            # 분석 결과 yield
                            analysis_data = {
                                "applicable_rules": applicable_rules,
                                "style_guide_violations": [
                                    {
                                        "id": f"{g.category}_SG{g.number:03d}",
                                        "category": g.category,
                                        "number": g.number,
                                        "description": ' '.join(g.content) if isinstance(g.content, list) else g.content
                                    }
                                    for g in style_guides_details
                                ],
                                "style_ids": [g.id for g in style_guides_details],
                                "fallback_mode": True,
                                "single_text_info": {
                                    "confidence": result.get("confidence", 0.0),
                                    "adapter_version": result.get("adapter_version", "unknown")
                                }
                            }
                            yield json.dumps({"type": "analysis", "data": analysis_data})
                            
                            # OpenAI 교정
                            final_openai_prompt = _generate_correction_prompt(style_guides_details, text, prompt)
                            
                            # Fallback 모드 프롬프트 로깅
                            logger.info("="*80)
                            logger.info("📝 Fallback Mode OpenAI Prompt:")
                            logger.info(f"Prompt Length: {len(final_openai_prompt)} characters")
                            logger.info(f"Style Guides Count: {len(style_guides_details)}")
                            logger.info("-"*40)
                            logger.info(f"Prompt (first 1500 chars):\n{final_openai_prompt[:1500]}...")
                            if len(final_openai_prompt) > 1500:
                                logger.info(f"... [truncated {len(final_openai_prompt) - 1500} more characters]")
                            logger.info("="*80)
                            # 프롬프트 파일 덤프 (폴백용)
                            try:
                                dump_prompt(
                                    "correction_fallback",
                                    final_openai_prompt,
                                    {
                                        "category": category.value if hasattr(category, "value") else str(category),
                                        "style_guides": [
                                            {
                                                "id": g.id,
                                                "number": g.number,
                                                "category": str(getattr(g, "category", ""))
                                            }
                                            for g in style_guides_details
                                        ],
                                        "fallback": True,
                                    },
                                )
                            except Exception:
                                pass

                            async for chunk_str in call_openai_stream_native(final_openai_prompt, ""):
                                try:
                                    delta_chunk = json.loads(chunk_str)
                                    yield json.dumps({"type": "delta", "data": delta_chunk})
                                except json.JSONDecodeError:
                                    pass
                            
                            logger.info("✅ Fallback processing completed.")
                            return
                            
            except Exception as fallback_error:
                logger.warning(f"Single text fallback also failed: {fallback_error}. Using OpenAI directly.")

    # 2. OpenAI 기본 교정 시도 (AI 서버 Flow 실패 시 Fallback)
    if settings.USE_OPENAI and settings.OPENAI_API_KEY:
        try:
            logger.info("🤖 Using OpenAI-only correction pipeline as fallback.")
            async for payload in call_openai_correction_stream(prompt, text, category, db):
                yield payload
            return
        except Exception as e:
            logger.warning(f"OpenAI-only pipeline failed: {e}. Will try Gemini if enabled.")

    # 2.5 Gemini 교정 시도 (OpenAI 실패 또는 비활성 시)
    if settings.USE_GEMINI and settings.GEMINI_API_KEY:
        try:
            logger.info("🟢 Using Gemini-only correction pipeline as fallback.")
            from .services.gemini_correction import call_gemini_correction_stream
            async for payload in call_gemini_correction_stream(prompt, text, category, db):
                yield payload
            return
        except Exception as e:
            logger.warning(f"Gemini-only pipeline failed: {e}. Falling back to Mock.")
    
    # 3. Mock 응답 (최종 Fallback)
    logger.info("📝 Using Mock response (all AI services unavailable).")
    mock_response = f"{text}\n\n[This is a mock correction as AI services are unavailable.]"
    for char in mock_response:
        yield json.dumps({"choices": [{"delta": {"content": char}}]})


async def call_ai_correction_stream_openai_only(
    prompt: str,
    text: str,
    category: ArticleCategory | None = None,
    db: AsyncSession | None = None,
):
    """OpenAI API만을 사용한 스트리밍 파이프라인 (테스트/폴백 용도)."""

    if db is None:
        raise ValueError("AsyncSession 'db' is required for OpenAI-only correction stream")

    resolved_category = category or ArticleCategory.BODY

    from .services.openai_correction import call_openai_correction_stream

    async for payload in call_openai_correction_stream(prompt, text, resolved_category, db):
        yield payload


# Backwards compatibility alias (legacy naming)
call_ai_correction_stream_before = call_ai_correction_stream_openai_only

def _parse_metadata_and_clean(text: str) -> tuple[str, list[int]]:
    """
    - 텍스트 끝부분의 'METADATA: {...}' 라인을 찾아 style_guide_ids를 추출하고,
      해당 라인을 제거한 본문을 반환.
    - 폴백: 'StyleGuide: 1,3,5' 패턴도 지원.
    """
    style_ids: list[int] = []

    # 1) METADATA: JSON 라인 찾기 (마지막 줄 우선)
    lines = text.splitlines()
    for i in range(len(lines)-1, -1, -1):
        line = lines[i].strip()
        if line.startswith("METADATA:"):
            payload = line[len("METADATA:"):].strip()
            try:
                obj = json.loads(payload)
                ids = obj.get("style_guide_ids", [])
                if isinstance(ids, list):
                    style_ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
            except Exception:
                pass
            # 메타데이터 라인은 본문에서 제거
            lines.pop(i)
            return ("\n".join(lines).rstrip(), style_ids)

    # 2) 폴백: "StyleGuide: 1,3,5" 패턴
    m = re.search(r"StyleGuide\s*:\s*([\d,\s]+)", text, re.IGNORECASE)
    if m:
        nums = [s.strip() for s in m.group(1).split(",")]
        style_ids = [int(s) for s in nums if s.isdigit()]
        # 해당 패턴 제거
        cleaned = re.sub(r"StyleGuide\s*:\s*[\d,\s]+", "", text, flags=re.IGNORECASE).rstrip()
        return (cleaned, style_ids)

    return (text, style_ids)

async def _attach_style_guides(db: AsyncSession, *, history_id: int, style_ids: list[int]) -> None:
    """전체 문서에 대한 스타일가이드 연결 (문장별이 아닌 경우)"""
    if not style_ids:
        return
    res = await db.execute(select(StyleGuide.id).where(StyleGuide.id.in_(style_ids)))
    exist_ids = set(res.scalars().all())
    for sid in exist_ids:
        db.add(TextCorrectionHistoryStyle(
            history_id=history_id, 
            style_id=sid,
            sentence_index=-1,  # -1 indicates whole document correction
            note="전체 문서 교정"
        ))


async def next_version_by_news(db: AsyncSession, news_key: str, category: ArticleCategory) -> int:
    """news_key + category 기준으로 다음 버전 번호 계산"""
    q = select(func.coalesce(func.max(TextCorrectionHistory.version), 0)).where(
        TextCorrectionHistory.news_key==news_key,
        TextCorrectionHistory.category==category
    )
    (maxv,) = (await db.execute(q)).one()
    return int(maxv)+1

async def create_article_with_history(
    db: AsyncSession, *,
    news_key: str,
    category: ArticleCategory,
    user_id: int,
    before_text: str,
    after_text: str,
    prompt: str,
    style_ids: list[int],
    operation_type: OperationType = OperationType.TRANSLATION_CORRECTION,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    original_text: Optional[str] = None,
    sentence_corrections: Optional[dict] = None  # 문장별 교정 정보
) -> TextCorrectionHistory:
    """교정 성공 후 TextCorrectionHistory만 생성 """
    
    # 1. news_key + category 기준으로 version 계산
    version = await next_version_by_news(db, news_key, category)
    
    # 2. TextCorrectionHistory 생성
    history = TextCorrectionHistory(
        news_key=news_key,
        category=category,
        version=version,     # news_key + category 기준 버전
        original_text=original_text,  # 완전 원본 텍스트 (번역+교정시 원본 한국어)
        before_text=before_text,
        after_text=after_text,
        prompt=prompt,
        operation_type=operation_type,
        source_lang=source_lang,
        target_lang=target_lang,
        created_by_user_id=user_id,
    )
    db.add(history)
    await db.flush()  # history.id를 얻기 위해 flush
    
    # 3. 문장별 교정 정보 저장 (있는 경우)
    if sentence_corrections:
        sentence_count = len(sentence_corrections)
        logger.info(f"Storing sentence-level corrections for {sentence_count} sentences")

        # StyleGuide의 number → id 매핑 생성 (한번만 조회)
        style_number_to_id = {}
        if style_ids:
            style_guides = await db.execute(
                select(StyleGuide).where(StyleGuide.id.in_(style_ids))
            )
            for sg in style_guides.scalars().all():
                style_number_to_id[sg.number] = sg.id
            logger.debug(f"Loaded {len(style_number_to_id)} style guides for mapping")

        try:
            for idx, sentence_data in enumerate(sentence_corrections):
                # 각 문장에 대한 스타일가이드별 교정 정보 저장
                sentence_index = sentence_data.get("sentence_index", idx)  # Default to idx if not provided
                before_text = sentence_data.get("before_text", "")
                after_text = sentence_data.get("after_text", "")
                violations = sentence_data.get("violations", [])
                
                # 데이터 유효성 검증
                if sentence_index is None:
                    logger.warning(f"Missing sentence_index for sentence {idx}, using index {idx}")
                    sentence_index = idx
                
                # sentence_index가 숫자가 아닌 경우 처리
                try:
                    sentence_index = int(sentence_index) if sentence_index is not None else idx
                except (ValueError, TypeError):
                    logger.warning(f"Invalid sentence_index type: {type(sentence_index)}, using index {idx}")
                    sentence_index = idx
                
                if not before_text and not after_text:
                    logger.warning(f"Both before_text and after_text are empty for sentence {sentence_index}")
                
                # 해당 문장에 적용된 스타일가이드들에 대해 개별 레코드 생성
                applied_style_ids = set()

                # violations는 두 가지 형식을 지원:
                # 1. dict 형식: {"style_guide_id": 1, ...} (AI server)
                # 2. string 형식: "articles_SG042" (OpenAI)
                for violation in violations:
                    if isinstance(violation, dict):
                        # dict 형식: style_guide_id 직접 사용
                        style_guide_id = violation.get("style_guide_id")
                        if style_guide_id and style_guide_id in style_ids:
                            applied_style_ids.add(style_guide_id)
                    elif isinstance(violation, str):
                        # string 형식: rule_id에서 style_guide_id 추출
                        # 예: "articles_SG042" → SG042 → 42번 스타일가이드
                        try:
                            # SG 뒤의 숫자 추출 (예: "articles_SG042" → 42)
                            match = re.search(r'SG(\d+)', violation)
                            if match:
                                sg_number = int(match.group(1))
                                # number → id 매핑에서 style_id 찾기
                                if sg_number in style_number_to_id:
                                    style_id = style_number_to_id[sg_number]
                                    applied_style_ids.add(style_id)
                                    logger.debug(f"Mapped rule_id '{violation}' → style_id {style_id}")
                                else:
                                    logger.warning(f"Style guide SG{sg_number:03d} not found in current style_ids")
                            else:
                                # A/H/C 코드 지원: A36/H05/C2 등 (공백/0패딩 허용)
                                match2 = re.match(r'^\s*([AHC])\s*0*(\d+)\s*$', str(violation), re.IGNORECASE)
                                if match2:
                                    sg_number = int(match2.group(2))
                                    if sg_number in style_number_to_id:
                                        style_id = style_number_to_id[sg_number]
                                        applied_style_ids.add(style_id)
                                        logger.debug(f"Mapped rule_code '{violation}' → style_id {style_id}")
                                    else:
                                        logger.warning(f"Style guide number {sg_number} not found in current style_ids")
                        except Exception as e:
                            logger.warning(f"Failed to parse rule_id from violation: {violation}, error: {e}")

                # 위반이 없는 문장은 스타일가이드 레코드를 저장하지 않음
                if not applied_style_ids:
                    logger.debug(f"No violations found for sentence {sentence_index}, skipping style guide records")
                    continue  # 다음 문장으로 건너뛰기

                # 각 적용된 스타일가이드에 대해 레코드 생성 (벌크 삽입을 위해 리스트에 저장)
                for style_id in applied_style_ids:
                    try:
                        history_style = TextCorrectionHistoryStyle(
                            history_id=history.id,
                            style_id=style_id,
                            sentence_index=sentence_index,
                            before_text=before_text[:2000] if before_text else "",  # Limit text length
                            after_text=after_text[:2000] if after_text else "",    # Limit text length
                            violations=violations,  # 해당 문장의 모든 violations 저장
                            note=f"문장 {sentence_index + 1 if sentence_index is not None else 'N/A'} 교정"
                        )
                        db.add(history_style)
                        logger.debug(f"Prepared sentence-level record for sentence {sentence_index}, style {style_id}")
                    except Exception as e:
                        logger.error(f"Failed to create sentence-level record for sentence {sentence_index}, style {style_id}: {e}")
                        # Continue with other records even if one fails
                        continue
                        
            logger.info(f"Successfully processed sentence-level corrections for {sentence_count} sentences")
            
        except Exception as e:
            logger.error(f"Error processing sentence corrections: {e}")
            # Fall back to traditional style guide attachment if sentence processing fails
            logger.info("Falling back to traditional style guide attachment")
            await _attach_style_guides(db, history_id=history.id, style_ids=style_ids)
    else:
        # 4. 기존 방식: 전체 텍스트에 대한 스타일가이드 연결
        logger.info("No sentence-level corrections provided, using traditional style guide attachment")
        await _attach_style_guides(db, history_id=history.id, style_ids=style_ids)
    
    # 5. 모든 변경사항 커밋
    await db.commit()
    
    return history

async def save_translation_history(
    db: AsyncSession, *,
    news_key: str,
    category: ArticleCategory,
    user_id: int,
    original_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str
) -> TextCorrectionHistory:
    """번역 이력을 저장 """
    
    # 1. news_key + category 기준으로 version 계산
    version = await next_version_by_news(db, news_key, category)
    
    # 2. TextCorrectionHistory 생성
    history = TextCorrectionHistory(
        news_key=news_key,
        category=category,
        version=version,     # news_key + category 기준 버전
        original_text=original_text,  # 완전 원본 텍스트
        before_text=original_text,     # 번역의 경우 before_text도 원본과 같음
        after_text=translated_text,
        prompt=None,  # 번역에는 프롬프트 없음
        operation_type=OperationType.TRANSLATION,
        source_lang=source_lang,
        target_lang=target_lang,
        created_by_user_id=user_id,
    )
    db.add(history)
    
    # 3. 커밋
    await db.commit()
    await db.refresh(history)
    
    return history

async def list_history(
    db: AsyncSession, *, 
    news_key: str, 
    category: ArticleCategory,
    operation_type: Optional[OperationType] = None
) -> list[TextCorrectionHistory]:
    query = (
        select(TextCorrectionHistory)
        .options(
            selectinload(TextCorrectionHistory.applied_styles)
            .selectinload(TextCorrectionHistoryStyle.style_guide)
        )
        .where(
            TextCorrectionHistory.news_key==news_key, 
            TextCorrectionHistory.category==category
        )
    )
    
    if operation_type:
        query = query.where(TextCorrectionHistory.operation_type==operation_type)
    
    query = query.order_by(TextCorrectionHistory.version.desc())
    
    res = await db.execute(query)
    return res.scalars().all()

async def list_news_history(
    db: AsyncSession, *,
    news_key: str,
    category: Optional[ArticleCategory] = None,  # Accept ArticleCategory enum
    operation_type: Optional[OperationType] = None
) -> list:
    """뉴스 히스토리 조회 (카테고리별 필터링 가능, SEO 포함)

    Args:
        category: None이면 모든 카테고리 (articles_translator 포함), 특정 enum이면 해당 카테고리만
    """

    # Handle SEO category separately
    if category == ArticleCategory.SEO:
        from .models import SEOGenerationHistory
        import json

        query = (
            select(SEOGenerationHistory)
            .where(SEOGenerationHistory.news_key == news_key)
            .order_by(SEOGenerationHistory.created_at.desc())
        )

        res = await db.execute(query)
        seo_histories = res.scalars().all()

        # Convert SEO histories to similar format as TextCorrectionHistory
        result = []
        for idx, history in enumerate(seo_histories):
            # Parse seo_titles from JSON string
            try:
                seo_titles = json.loads(history.seo_titles) if history.seo_titles else []
            except json.JSONDecodeError:
                seo_titles = []

            # Create a mock TextCorrectionHistory-like object for compatibility
            result.append({
                "id": history.id,
                "news_key": history.news_key,
                "category": "SEO",
                "version": len(seo_histories) - idx,  # Reverse index as version
                "before_text": history.input_text,
                "after_text": history.edited_title,
                "operation_type": "CORRECTION",
                "created_at": history.created_at,
                "applied_styles": [],  # SEO doesn't have style guides
                "seo_titles": seo_titles,  # Additional field for SEO
                "raw_response": history.raw_response
            })

        return result

    # Handle regular categories (includes articles_translator)
    query = (
        select(TextCorrectionHistory)
        .options(
            selectinload(TextCorrectionHistory.applied_styles)
            .selectinload(TextCorrectionHistoryStyle.style_guide)
        )
        .where(TextCorrectionHistory.news_key==news_key)
    )

    # category 필터 추가 (None이면 모든 카테고리 포함, articles_translator 포함)
    if category is not None:
        query = query.where(TextCorrectionHistory.category == category)
    
    if operation_type:
        query = query.where(TextCorrectionHistory.operation_type==operation_type)

    # 날짜순으로 정렬 (최신순)
    query = query.order_by(TextCorrectionHistory.created_at.desc())

    res = await db.execute(query)
    return res.scalars().all()

async def gpt_generate_title(
    db: AsyncSession,
    input_text: str,
    selected_type: str,
    data_type: str,
    model: str,
    guideline_text: str = None,
    news_key: str = None,
    user_id: int = None
) -> dict:
    """
    GPT를 사용하여 SEO 최적화된 제목을 생성합니다.
    
    Args:
        input_text: 원본 제목 텍스트
        selected_type: 선택된 유형 (현재 사용되지 않음)
        data_type: 헤드라인 작성 규칙
        model: 사용할 GPT 모델명
        guideline_text: 추가 가이드라인 텍스트 (선택사항)
    
    Returns:
        생성된 제목들이 포함된 딕셔너리 (edited_title, seo_titles, raw_response)
    """
    logger.info("Starting GPT title generation")
    
    # 오늘 날짜를 가져오기
    from datetime import datetime
    today_date = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"Today's date: {today_date}")
    
    prompt = f"""
    You are an SEO and editorial expert. Your tasks are as follows:
    
    Write a headline within 15 words according to the headline writing rules, and write three additional SEO-optimized headlines that include popular keywords while maintaining readability and relevance to the content. Please write according to the format below. If a Korean title is entered, please write it in English. Do not include any other descriptions or phrases.
    
    Note : Today's date is as follows, and if you need to create an title using today's date, please refer to the following date and create the title.
    
    #Input Title : "{input_text}"
    
    Edited Title:
    
    SEO Title 1:
    
    SEO Title 2:
    
    SEO Title 3:
    
    #Today's date : {today_date}
    
    #Headline writing rules : {data_type}
    """

    dump_prompt(
        "seo",
        prompt,
        {
            "news_key": news_key,
            "model": model,
            "data_type": data_type,
            "user_id": user_id,
        },
    )

    try:
        # OpenAI 클라이언트 가져오기
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_SEO_KEY)
        
        input_messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "You are an SEO and editorial expert."}],
            }
        ]

        if guideline_text:
            input_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": f"추가적인 지침: {guideline_text}"}],
                }
            )

        input_messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        completion = await client.responses.create(
            model=model,
            input=input_messages,
            temperature=1,
        )

        result = ""
        for item in completion.output or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    result += content.text
        logger.info(f"GPT title generation result: {result[:200]}...")  # 처음 200자만 로그
        
        # Parse the result to extract titles
        lines = result.strip().split('\n')
        edited_title = ""
        seo_titles = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Edited Title:'):
                edited_title = line.replace('Edited Title:', '').strip()
            elif line.startswith('SEO Title 1:'):
                seo_titles.append(line.replace('SEO Title 1:', '').strip())
            elif line.startswith('SEO Title 2:'):
                seo_titles.append(line.replace('SEO Title 2:', '').strip())
            elif line.startswith('SEO Title 3:'):
                seo_titles.append(line.replace('SEO Title 3:', '').strip())
        
        # Save to SEOGenerationHistory
        if db:
            from .models import SEOGenerationHistory
            import json
            
            history = SEOGenerationHistory(
                news_key=news_key,
                input_text=input_text,
                edited_title=edited_title or result,  # Fallback to full result if parsing failed
                seo_titles=json.dumps(seo_titles),  # Store as JSON string
                raw_response=result,
                prompt_used=prompt,
                model=model,
                data_type=data_type,
                guideline_text=guideline_text,
                created_by_user_id=user_id
            )
            db.add(history)
            await db.commit()
            logger.info(f"Saved SEO generation history with ID: {history.id}")
        
        return {
            "edited_title": edited_title or result,
            "seo_titles": seo_titles,
            "raw_response": result
        }
    
    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e)}"
        return {
            "edited_title": error_msg,
            "seo_titles": [],
            "raw_response": error_msg
        }


# CMS 연동을 위한 서비스 함수들
async def save_cms_article(
    db: AsyncSession,
    news_key: str,
    category: ArticleCategory,
    content: str,
    author_name: str
) -> Article:
    """CMS에서 전송한 단일 Article 저장/수정 (upsert)
    
    Args:
        db: 데이터베이스 세션
        news_key: 뉴스 키
        category: 카테고리
        content: 콘텐츠
        author_name: CMS 작성자 이름
        
    Returns:
        생성/수정된 Article
    """
    import json

    # 입력 category가 문자열로 들어오는 경우(관리자/외부 연동) 안전하게 Enum으로 보정
    if isinstance(category, str):
        raw = category.strip().upper().replace("-", "_")
        alias = {
            "SEO": ArticleCategory.SEO,
            "SEO_TITLE": ArticleCategory.SEO,
            "TITLE": ArticleCategory.TITLE,
            "HEADLINE": ArticleCategory.TITLE,
            "HEADLINES": ArticleCategory.TITLE,
            "BODY": ArticleCategory.BODY,
            "ARTICLE": ArticleCategory.BODY,
            "ARTICLES": ArticleCategory.BODY,
            "ARTICLE_TRANSLATOR": ArticleCategory.BODY,
            "ARTICLES_TRANSLATOR": ArticleCategory.BODY,
            "CAPTION": ArticleCategory.CAPTION,
            "CAPTIONS": ArticleCategory.CAPTION,
        }
        category = alias.get(raw, ArticleCategory.BODY)
    
    # Caption 처리: 여러 개의 caption을 JSON 형식으로 저장
    if category == ArticleCategory.CAPTION:
        try:
            # content가 이미 JSON 형식인지 확인
            caption_data = json.loads(content)
            # JSON이지만 올바른 구조가 아닌 경우 재구성
            if not isinstance(caption_data, dict) or "captions" not in caption_data:
                if isinstance(caption_data, list):
                    # 배열이면 captions로 감싸기
                    caption_data = {"captions": caption_data}
                else:
                    # 그 외의 경우 단일 caption으로 처리
                    caption_data = {"captions": [{"index": 0, "text": str(caption_data)}]}
        except (json.JSONDecodeError, ValueError):
            # JSON이 아니면 단일 caption으로 처리
            # 구분자가 있는지 확인 (예: |||)
            if "|||" in content:
                captions = content.split("|||")
                caption_data = {
                    "captions": [
                        {"index": i, "text": cap.strip()} 
                        for i, cap in enumerate(captions) if cap.strip()
                    ]
                }
            else:
                caption_data = {"captions": [{"index": 0, "text": content}]}
        
        content = json.dumps(caption_data, ensure_ascii=False)
        logger.info(f"Processed caption data for news_key={news_key}: {len(caption_data.get('captions', []))} captions")
    
    # CMS 시스템 사용자 ID (고정값)
    CMS_SYSTEM_USER_ID = 1
    
    # 기존 article 조회 (news_key + category 조합으로)
    result = await db.execute(
        select(Article).where(
            and_(
                Article.news_key == news_key,
                Article.category == category
            )
        )
    )
    article = result.scalar_one_or_none()
    
    if article:
        # 업데이트
        article.text = content
        article.cms_author = author_name
        article.updated_at = func.now()
        logger.info(f"Updated CMS article: {news_key}-{category} by {author_name}")
    else:
        # 생성
        article = Article(
            news_key=news_key,
            category=category,
            text=content,
            user_id=CMS_SYSTEM_USER_ID,  # 고정된 시스템 사용자 ID
            cms_author=author_name,
            status=ArticleStatus.DRAFT
        )
        db.add(article)
        logger.info(f"Created new CMS article: {news_key}-{category} by {author_name}")
    
    await db.commit()
    await db.refresh(article)
    return article


async def get_article_by_public_id(
    db: AsyncSession,
    public_id: str
) -> Article:
    """Article public_id(UUID)로 단일 Article 조회"""
    result = await db.execute(
        select(Article).where(Article.public_id == public_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


async def get_article_by_news_key_and_category(
    db: AsyncSession,
    news_key: str,
    category: ArticleCategory
) -> Article:
    """news_key와 category로 Article 조회"""
    result = await db.execute(
        select(Article).where(
            Article.news_key == news_key,
            Article.category == category
        )
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail=f"Article not found for news_key={news_key}, category={category}")
    return article


async def get_article_by_identifier(
    db: AsyncSession,
    identifier: str,
    category: Optional[ArticleCategory] = None
) -> Article:
    """identifier가 news_key인지 public_id인지 자동 판단하여 조회
    
    Args:
        identifier: news_key 또는 public_id
        category: category가 제공되면 news_key로 간주하여 조회
    """
    # category가 제공되면 news_key + category로 조회
    if category:
        return await get_article_by_news_key_and_category(db, identifier, category)
    
    # UUID 형식이면 public_id로 조회
    try:
        import uuid
        uuid.UUID(identifier)
        return await get_article_by_public_id(db, identifier)
    except ValueError:
        # UUID가 아니면 news_key로 간주 (첫 번째 매칭 반환)
        result = await db.execute(
            select(Article).where(Article.news_key == identifier).limit(1)
        )
        article = result.scalar_one_or_none()
        if not article:
            raise HTTPException(status_code=404, detail=f"Article not found for identifier={identifier}")
        return article


def parse_caption_content(content: str) -> dict:
    """Caption 콘텐츠를 파싱하여 구조화된 데이터 반환
    
    Args:
        content: Caption 텍스트 (JSON 또는 일반 텍스트)
    
    Returns:
        dict: {"captions": [{"index": 0, "text": "..."}]}
    """
    import json
    
    try:
        # JSON 형식인 경우
        caption_data = json.loads(content)
        if isinstance(caption_data, dict) and "captions" in caption_data:
            return caption_data
        elif isinstance(caption_data, list):
            return {"captions": caption_data}
        else:
            return {"captions": [{"index": 0, "text": str(caption_data)}]}
    except (json.JSONDecodeError, ValueError):
        # JSON이 아닌 경우
        if "|||" in content:
            # 구분자로 분리
            captions = content.split("|||")
            return {
                "captions": [
                    {"index": i, "text": cap.strip()}
                    for i, cap in enumerate(captions) if cap.strip()
                ]
            }
        else:
            # 단일 caption
            return {"captions": [{"index": 0, "text": content}]}


async def get_all_captions_for_news(
    db: AsyncSession,
    news_key: str
) -> List[dict]:
    """특정 news_key의 모든 caption 조회
    
    Returns:
        List[dict]: Caption 목록
    """
    try:
        article = await get_article_by_news_key_and_category(
            db, news_key, ArticleCategory.CAPTION
        )
        caption_data = parse_caption_content(article.text)
        return caption_data.get("captions", [])
    except HTTPException:
        # Caption이 없는 경우 빈 리스트 반환
        return []

# 하위 호환성을 위한 함수 (필요시)
async def get_article_by_id(
    db: AsyncSession,
    article_id: int
) -> Article:
    """Article ID로 단일 Article 조회 (내부용)"""
    result = await db.execute(
        select(Article).where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article
