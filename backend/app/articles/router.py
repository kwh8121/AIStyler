# backend/app/articles/router.py
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from ..database import SessionDep
from ..users.dependencies import resolve_actor_user
from ..users.models import User
from .schemas import (
    ArticleCorrectionRequest,
    ArticleCorrectionResponse,
    TranslationRequest,
    TranslationResponse,
    NewsHistoryResponse,
    TitleGenerationRequest,
    TitleGenerationResponse,
    CMSSaveRequest,
    CMSSaveResponse,
    CMSGetResponse,
    AppliedStyleGuide,
    HistoryRestoreRequest,
    HistoryRestoreResponse
)
from .models import ArticleCategory, OperationType
from . import service
from .services.translation import translate_title
from ..config import settings
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("/correct/stream")
async def correct_article_stream(body: ArticleCorrectionRequest, db: SessionDep, actor: User = Depends(resolve_actor_user)):
    # 사용자가 직접 입력한 추가 프롬프트 사용 (없으면 None)
    additional_prompt = body.prompt if body.prompt and body.prompt.strip() else None
    
    # 전체 처리 시작 시간
    start_time = time.time()
    logger.info(f"Starting correction stream for news_key={body.news_key}, category={body.category}, user_id={actor.id}")

    # 저장 선행 강제: local-로 시작하지 않는 news_key는 DB에 Article가 존재해야 함
    try:
        is_local_mode_check = body.news_key.lower().startswith("local-")
        if not is_local_mode_check:
            # 존재하지 않으면 400으로 안내
            try:
                await service.get_article_by_news_key_and_category(db, body.news_key, body.category)
            except HTTPException as e:
                if e.status_code == 404:
                    raise HTTPException(
                        status_code=400,
                        detail="Article not found. Save the article first via POST /articles/save"
                    )
                raise
    except Exception:
        # 제너레이터 시작 전에 예외를 그대로 전달
        raise

    async def generator():
        nonlocal db, additional_prompt, actor
        chunks: list[str] = []
        before_en = None
        source_lang = None
        target_lang = None
        sentence_corrections = {}  # 문장별 교정 정보 저장
        analysis_result = None
        
        if body.category == ArticleCategory.SEO:
            try:
                yield f"data: {json.dumps({'status': 'seo_generating', 'message': 'SEO 제목 생성중...'}, ensure_ascii=False)}\n\n"

                is_local_mode = body.news_key.lower().startswith("local-")
                seo_db = None if is_local_mode else db
                seo_request = TitleGenerationRequest(
                    news_key=None if is_local_mode else body.news_key,
                    input_text=body.text,
                    data_type="headline",
                    model=settings.OPENAI_MODEL or "o4-mini",
                    selected_type=None,
                    guideline_text=None,
                )

                seo_result = await service.gpt_generate_title(
                    db=seo_db,
                    input_text=seo_request.input_text,
                    selected_type=seo_request.selected_type or "",
                    data_type=seo_request.data_type,
                    model=seo_request.model,
                    guideline_text=seo_request.guideline_text,
                    news_key=seo_request.news_key,
                    user_id=actor.id if actor else None,
                )

                edited_title = seo_result.get("edited_title", "") if isinstance(seo_result, dict) else str(seo_result)
                seo_titles = seo_result.get("seo_titles", []) if isinstance(seo_result, dict) else []
                seo_titles = seo_titles[:3] if isinstance(seo_titles, list) else []
                combined_parts = [edited_title, *seo_titles]
                combined_text = "\n".join(part for part in combined_parts if part) or edited_title

                if combined_text:
                    delta_payload = {
                        "type": "delta",
                        "data": {"choices": [{"delta": {"content": combined_text}}]},
                    }
                    yield f"data: {json.dumps(delta_payload, ensure_ascii=False)}\n\n"

                result_payload = {
                    "status": "seo_complete",
                    "message": "SEO 제목 생성 완료",
                    "edited_title": edited_title,
                    "seo_titles": seo_titles,
                }
                yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'status': 'complete', 'message': '교정 완료'}, ensure_ascii=False)}\n\n"
            except Exception as seo_error:
                logger.error("SEO generation failed: %s", seo_error)
                error_payload = {
                    "status": "error",
                    "message": f"처리 중 오류가 발생했습니다: {seo_error}",
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                raise
            finally:
                yield "data: [DONE]\n\n"
            return

        # v2 파이프라인 사용 여부에 따라 스트림 함수 선택 (아래 공통 래핑 로직 사용)
        if settings and getattr(settings, "GPT5_V2_ENABLED", False):
            from .services.gpt5v2_correction import call_gpt5v2_correction_stream as _stream_fn
            logger.info("Using GPT-5 v2 correction stream in unified wrapper")
        else:
            _stream_fn = service.call_ai_correction_stream

        try:
            # 1. 번역 시작 메시지 전송
            yield f"data: {json.dumps({'status': 'translating', 'message': '번역중...'}, ensure_ascii=False)}\n\n"
            
            # 2. 실제 번역 수행 (언어 정보 포함)
            translation_start = time.time()
            logger.info(f"Starting translation for text length: {len(body.text)} chars")
            
            # 자동 감지를 위해 source_lang=None으로 전달 (영어 텍스트는 번역 건너뜀)
            before_en, source_lang, target_lang = await service.translate_text(
                body.text,
                source_lang=None,
                target_lang="EN-US"
            )
            
            translation_time = time.time() - translation_start
            logger.info(f"Translation completed in {translation_time:.3f}s, result length: {len(before_en)} chars")

            # 3. 번역 완료 메시지 전송 (처리 시간 포함)
            yield f"data: {json.dumps({'status': 'translation_complete', 'message': '번역 완료', 'elapsed': round(translation_time, 3)}, ensure_ascii=False)}\n\n"
            
            # 4. 스타일 가이드 적용 시작 메시지 전송
            yield f"data: {json.dumps({'status': 'applying_style', 'message': '스타일 가이드 적용중...'}, ensure_ascii=False)}\n\n"
            
            # 5. AI 교정 스트리밍 시작
            ai_processing_start = time.time()
            logger.info(f"Starting AI correction stream for text length: {len(before_en)} chars")
            
            async for payload in _stream_fn(additional_prompt, before_en, body.category, db):
                parsed_payload = json.loads(payload)
                payload_type = parsed_payload.get("type")
                payload_data = parsed_payload.get("data")

                if payload_type == "analysis":
                    analysis_result = payload_data
                    yield f"data: {json.dumps({'status': 'analysis_complete', 'message': '스타일 가이드 분석 완료', 'analysis': analysis_result}, ensure_ascii=False)}\n\n"
                
                elif payload_type == "delta":
                    yield f"data: {json.dumps(payload_data)}\n\n"
                    try:
                        delta = payload_data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            chunks.append(delta)
                    except Exception:
                        pass
                
                elif payload_type == "sentence_corrections":
                    # 문장별 교정 정보 처리 (스트리밍 완료 후 받음)
                    sentence_corrections = payload_data.get("sentence_corrections", {})
                    full_text = payload_data.get("full_text", "")
                    
                    # 전체 텍스트가 있으면 chunks를 대체
                    if full_text:
                        chunks.clear()
                        chunks.append(full_text)
                    
                    logger.info(f"Received sentence corrections: {len(sentence_corrections)} sentences")
                    yield f"data: {json.dumps({'status': 'sentence_parsing_complete', 'message': '문장별 교정 파싱 완료'}, ensure_ascii=False)}\n\n"
                
                elif payload_type == "final_analysis":
                    # 최종 분석 결과 (DB 저장용 데이터)
                    analysis_result = payload_data
                    sentence_corrections = payload_data.get("sentence_corrections", {})
                    logger.info(f"Final analysis ready for DB save")
                
                elif payload_type == "error":
                    # 서비스에서 발생한 에러 처리
                    error_message = payload_data.get("message", "Unknown error in service")
                    raise Exception(error_message)
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error occurred after {error_time:.3f}s: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'처리 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)}\n\n"
            raise
        finally:
            if before_en and chunks:
                ai_processing_time = time.time() - ai_processing_start
                logger.info(f"AI correction completed in {ai_processing_time:.3f}s, chunks collected: {len(chunks)}")

                model_out = "".join(chunks).strip()

                # METADATA 파싱하여 style_ids 추출하고 본문에서는 제거
                after_en, parsed_style_ids = service._parse_metadata_and_clean(model_out)
                logger.info(
                    f"METADATA parsing - style_ids: {parsed_style_ids}, cleaned text length: {len(after_en)}"
                )

                # AI 서버 분석에서 온 style_ids를 우선 사용, 없으면 METADATA에서 파싱한 것 사용
                style_ids = []
                if analysis_result:
                    style_ids = analysis_result.get("style_ids", [])
                if not style_ids and parsed_style_ids:
                    style_ids = parsed_style_ids

                is_local_mode = body.news_key.lower().startswith("local-")

                history = None
                db_save_time = 0.0

                if not is_local_mode:
                    # DB 저장 시간 측정
                    db_save_start = time.time()
                    history = await service.create_article_with_history(
                        db=db,
                        news_key=body.news_key,
                        category=body.category,
                        user_id=actor.id,
                        before_text=before_en,
                        after_text=after_en,
                        prompt=additional_prompt,
                        style_ids=style_ids,
                        operation_type=OperationType.TRANSLATION_CORRECTION,  # 번역+교정 타입
                        source_lang=source_lang,  # 언어 정보 추가
                        target_lang=target_lang,   # 언어 정보 추가
                        original_text=body.text,  # 완전 원본 텍스트 (번역 전 한국어)
                        sentence_corrections=[
                            {
                                "sentence_index": idx,
                                "before_text": data.get("original", ""),
                                "after_text": data.get("corrected", ""),
                                "violations": data.get("violations", [])
                            }
                            for idx, data in sentence_corrections.items()
                        ] if sentence_corrections and isinstance(sentence_corrections, dict) else None
                    )
                    db_save_time = time.time() - db_save_start
                    total_time = time.time() - start_time

                    logger.info(f"DB save completed in {db_save_time:.3f}s, history_id: {history.id}")
                    logger.info(
                        f"Total processing completed in {total_time:.3f}s - Translation: {translation_time:.3f}s, AI: {ai_processing_time:.3f}s, DB: {db_save_time:.3f}s"
                    )
                else:
                    total_time = time.time() - start_time
                    logger.info(
                        f"Local mode detected for news_key={body.news_key}; skipping DB persistence. Total processing: {total_time:.3f}s (Translation: {translation_time:.3f}s, AI: {ai_processing_time:.3f}s)"
                    )

                # 7. 최종 완료 메시지 전송 (로컬/서버 모드 동일)
                yield f"data: {json.dumps({'status': 'complete', 'message': '교정 완료'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache", 
        "Expires": "0",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        "X-Content-Type-Options": "nosniff",  # 브라우저 버퍼링 방지
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    })

@router.post("/correct/openai-stream")
async def correct_article_openai_stream(
    body: ArticleCorrectionRequest,
    db: SessionDep,
    actor: User = Depends(resolve_actor_user)
):
    """
    OpenAI API를 직접 사용한 교정 스트리밍
    - DeepL로 번역
    - OpenAI로 스타일가이드 분석 및 교정
    - 기존 correct/stream과 동일한 응답 포맷
    """
    # 사용자가 직접 입력한 추가 프롬프트 사용
    additional_prompt = body.prompt if body.prompt and body.prompt.strip() else None

    # 전체 처리 시작 시간
    start_time = time.time()
    logger.info(f"Starting OpenAI correction stream for news_key={body.news_key}, category={body.category}, user_id={actor.id}")

    # 저장 선행 강제: local-로 시작하지 않는 news_key는 DB에 Article가 존재해야 함
    try:
        is_local_mode_check = body.news_key.lower().startswith("local-")
        if not is_local_mode_check:
            try:
                await service.get_article_by_news_key_and_category(db, body.news_key, body.category)
            except HTTPException as e:
                if e.status_code == 404:
                    raise HTTPException(
                        status_code=400,
                        detail="Article not found. Save the article first via POST /articles/save"
                    )
                raise
    except Exception:
        raise

    async def generator():
        nonlocal db, additional_prompt, actor

        try:
            # 파이프라인 선택: v2가 활성화되면 v2 스트리밍 사용
            if settings and getattr(settings, "GPT5_V2_ENABLED", False):
                from .services.gpt5v2_correction import call_gpt5v2_correction_stream as _call_stream
                logger.info("### V2GPT GGGGOOOO ####")
            else:
                from .services.openai_correction import call_openai_correction_stream as _call_stream
                logger.info("### V1GPT GGGGOOOO ####")

            # 모든 처리를 선택된 모듈에 위임
            async for payload in _call_stream(
                additional_prompt,
                body.text,
                body.category,
                db
            ):
                # 스트리밍 데이터 그대로 전달
                yield f"data: {payload}\n\n"

                # final_analysis 이벤트에서 DB 저장
                try:
                    parsed = json.loads(payload)
                    if parsed.get("type") == "final_analysis":
                        is_local_mode = body.news_key.lower().startswith("local-")
                        if is_local_mode:
                            total_time = time.time() - start_time
                            logger.info(
                                f"Local mode detected for news_key={body.news_key}; skipping DB persistence. Total processing so far: {total_time:.3f}s"
                            )
                            continue
                        # DB 저장 로직
                        analysis_data = parsed.get("data", {})
                        style_ids = analysis_data.get("style_ids", [])
                        sentence_corrections = analysis_data.get("sentence_corrections", {})
                        full_text = analysis_data.get("full_text", "")

                        translation_data = analysis_data.get("translation", {}) or {}
                        before_en = translation_data.get("before_text")
                        source_lang = translation_data.get("source_lang")
                        target_lang = translation_data.get("target_lang")

                        if not before_en:
                            from .services.translation import translate_text
                            before_en, source_lang, target_lang = await translate_text(
                                body.text, source_lang="KO", target_lang="EN-US"
                            )
                        source_lang = source_lang or "UNKNOWN"
                        target_lang = target_lang or "EN-US"

                        # DB 히스토리 저장
                        history = await service.create_article_with_history(
                            db=db,
                            news_key=body.news_key,
                            category=body.category,
                            user_id=actor.id,
                            before_text=before_en,
                            after_text=full_text,
                            prompt=additional_prompt,
                            style_ids=style_ids,
                            operation_type=OperationType.TRANSLATION_CORRECTION,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            original_text=body.text,
                            sentence_corrections=[
                                {
                                    "sentence_index": idx,
                                    "before_text": data.get("original", ""),
                                    "after_text": data.get("corrected", ""),
                                    "violations": data.get("violations", [])
                                }
                                for idx, data in sentence_corrections.items()
                            ] if sentence_corrections else None
                        )

                        total_time = time.time() - start_time
                        logger.info(f"OpenAI correction completed in {total_time:.3f}s, history_id: {history.id}")

                except Exception as parse_error:
                    logger.debug(f"Payload parsing for DB save: {parse_error}")

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"OpenAI correction error after {error_time:.3f}s: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'처리 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)}\n\n"

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Content-Type-Options": "nosniff",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    })

@router.post("/correct/gemini-stream")
async def correct_article_gemini_stream(
    body: ArticleCorrectionRequest,
    db: SessionDep,
    actor: User = Depends(resolve_actor_user),
):
    """
    Google Gemini API를 사용한 교정 스트리밍
    - DeepL로 번역(자동 감지)
    - Gemini로 스타일가이드 분석 및 교정
    - 기존 스트리밍 포맷과 동일한 SSE 구조 유지
    """
    additional_prompt = body.prompt if body.prompt and body.prompt.strip() else None
    start_time = time.time()
    logger.info(
        f"Starting Gemini correction stream for news_key={body.news_key}, category={body.category}, user_id={actor.id}"
    )

    async def generator():
        try:
            from .services.gemini_correction import call_gemini_correction_stream

            async for payload in call_gemini_correction_stream(
                additional_prompt, body.text, body.category, db
            ):
                yield f"data: {payload}\n\n"

                # final_analysis에서 DB 저장
                try:
                    parsed = json.loads(payload)
                    if parsed.get("type") == "final_analysis":
                        is_local_mode = body.news_key.lower().startswith("local-")
                        if is_local_mode:
                            total_time = time.time() - start_time
                            logger.info(
                                f"Local mode detected for news_key={body.news_key}; skipping DB persistence. Total processing so far: {total_time:.3f}s"
                            )
                            continue

                        analysis_data = parsed.get("data", {})
                        style_ids = analysis_data.get("style_ids", [])
                        sentence_corrections = analysis_data.get("sentence_corrections", {})
                        full_text = analysis_data.get("full_text", "")

                        translation_data = analysis_data.get("translation", {}) or {}
                        before_en = translation_data.get("before_text")
                        source_lang = translation_data.get("source_lang")
                        target_lang = translation_data.get("target_lang")

                        if not before_en:
                            from .services.translation import translate_text
                            before_en, source_lang, target_lang = await translate_text(
                                body.text, source_lang=None, target_lang="EN-US"
                            )
                        source_lang = source_lang or "UNKNOWN"
                        target_lang = target_lang or "EN-US"

                        history = await service.create_article_with_history(
                            db=db,
                            news_key=body.news_key,
                            category=body.category,
                            user_id=actor.id,
                            before_text=before_en,
                            after_text=full_text,
                            prompt=additional_prompt,
                            style_ids=style_ids,
                            operation_type=OperationType.TRANSLATION_CORRECTION,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            original_text=body.text,
                            sentence_corrections=[
                                {
                                    "sentence_index": idx,
                                    "before_text": data.get("original", ""),
                                    "after_text": data.get("corrected", ""),
                                    "violations": data.get("violations", []),
                                }
                                for idx, data in sentence_corrections.items()
                            ]
                            if sentence_corrections
                            else None,
                        )
                        total_time = time.time() - start_time
                        logger.info(
                            f"Gemini correction completed in {total_time:.3f}s, history_id: {history.id}"
                        )
                except Exception as parse_error:
                    logger.debug(f"Payload parsing for DB save (Gemini): {parse_error}")

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Gemini correction error after {error_time:.3f}s: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'처리 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

@router.post("/translate", response_model=TranslationResponse)
async def translate_article(
    body: TranslationRequest, 
    db: SessionDep, 
    actor: User = Depends(resolve_actor_user)
):
    """독립적인 번역 API endpoint"""
    # 번역 수행: 제목/SEO는 전용 단일 단계, 그 외는 일반 번역(프로바이더 설정 따름)
    if body.category in (ArticleCategory.TITLE, ArticleCategory.SEO):
        logger.info("[translate] Using translate_title (single-pass) for headline/SEO endpoint request")
        translated_text, source_lang, target_lang = await translate_title(body.text)
    else:
        translated_text, source_lang, target_lang = await service.translate_text(
            text=body.text,
            source_lang=body.source_lang,
            target_lang=body.target_lang
        )
    
    # 번역 이력 저장
    history = await service.save_translation_history(
        db=db,
        news_key=body.news_key,
        category=body.category,
        user_id=actor.id,
        original_text=body.text,
        translated_text=translated_text,
        source_lang=source_lang,
        target_lang=target_lang
    )
    
    return TranslationResponse(
        translated_text=translated_text,
        source_lang=source_lang,
        target_lang=target_lang,
        history_id=history.id,
        version=history.version
    )

@router.get("/{news_key}/history", response_model=list[NewsHistoryResponse])
async def get_news_history(
    news_key: str,
    db: SessionDep,
    category: Optional[str] = Query(None, description="카테고리 필터 (headlines, articles, captions, seo)"),
    operation_type: Optional[OperationType] = Query(None, description="필터링할 작업 타입")
):
    """뉴스 이력 조회 (category 파라미터로 필터링 가능)"""
    # Pass category as string to service (handles "seo" specially)
    histories = await service.list_news_history(
        db,
        news_key=news_key,
        category=map_frontend_category_to_enum(category),  # Pass string directly
        operation_type=operation_type
    )
    
    # Handle both dict (SEO) and object (regular) results
    result = []
    for h in histories:
        if isinstance(h, dict):
            # SEO history returns dict format
            result.append(NewsHistoryResponse(
                history_id=h["id"],
                news_key=h["news_key"],
                category=h["category"],
                version=h["version"],
                original_text=h.get("before_text", ""),
                before_text=h["before_text"],
                after_text=h["after_text"],
                operation_type=h.get("operation_type"),
                source_lang=h.get("source_lang"),
                target_lang=h.get("target_lang"),
                created_at=h["created_at"],
                applied_styles=[]  # SEO doesn't have style guides
            ))
        else:
            # Regular TextCorrectionHistory object
            result.append(NewsHistoryResponse(
                history_id=h.id,
                news_key=h.news_key,
                category=h.category,
                version=h.version,
                original_text=h.original_text or h.before_text,
                before_text=h.before_text,
                after_text=h.after_text,
                operation_type=h.operation_type,
                source_lang=h.source_lang,
                target_lang=h.target_lang,
                created_at=h.created_at,
                applied_styles=[
                    AppliedStyleGuide(
                        style_id=applied.style_guide.id,
                        number=applied.style_guide.number,
                        name=applied.style_guide.name,
                        category=applied.style_guide.category,
                        docs=applied.style_guide.docs,
                        applied_at=applied.applied_at,
                        note=applied.note,
                        sentence_index=applied.sentence_index,
                        before_text=applied.before_text,
                        after_text=applied.after_text,
                        violations=applied.violations
                    )
                    for applied in h.applied_styles
                ]
            ))
    
    return result

@router.get("/{news_key}/{category}/history", response_model=list[ArticleCorrectionResponse])
async def get_history(
    news_key: str, 
    category: ArticleCategory, 
    db: SessionDep,
    operation_type: Optional[OperationType] = Query(None, description="필터링할 작업 타입")
):
    """특정 카테고리 이력 조회 (operation_type으로 필터링 가능)"""
    histories = await service.list_history(
        db, 
        news_key=news_key, 
        category=category,
        operation_type=operation_type
    )
    
    return [
        ArticleCorrectionResponse(
            history_id=h.id,
            news_key=h.news_key,
            category=h.category,
            version=h.version,
            before_text=h.before_text,
            after_text=h.after_text,
            operation_type=h.operation_type,
            created_at=h.created_at,
            applied_styles=[
                AppliedStyleGuide(
                    style_id=applied.style_guide.id,
                    number=applied.style_guide.number,
                    name=applied.style_guide.name,
                    category=applied.style_guide.category,
                    docs=applied.style_guide.docs,
                    applied_at=applied.applied_at,
                    note=applied.note,
                    sentence_index=applied.sentence_index,
                    before_text=applied.before_text,
                    after_text=applied.after_text,
                    violations=applied.violations
                )
                for applied in h.applied_styles
            ]
        )
        for h in histories
    ]

def _parse_seo_response(raw_response, default_title):
    """Helper function to parse SEO response"""
    import re
    
    raw_response = str(raw_response)
    edited_title = ""
    seo_titles = []
    
    # Edited Title 추출
    edited_match = re.search(r"Edited Title:\s*(.+?)(?:\n|$)", raw_response, re.IGNORECASE)
    if edited_match:
        edited_title = edited_match.group(1).strip()
    
    # SEO Title 1, 2, 3 추출
    for i in range(1, 4):
        pattern = rf"SEO Title {i}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            seo_titles.append(match.group(1).strip())
    
    # 만약 파싱이 실패했을 경우 대체 처리
    if not edited_title and not seo_titles:
        # 전체 응답을 줄 단위로 분리하여 처리
        lines = raw_response.strip().split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if non_empty_lines:
            edited_title = non_empty_lines[0] if len(non_empty_lines) > 0 else default_title
            seo_titles = non_empty_lines[1:4] if len(non_empty_lines) > 1 else []
    
    # 최소 응답 보장
    if not edited_title:
        edited_title = default_title
    
    # SEO 제목이 부족한 경우 채우기
    while len(seo_titles) < 3:
        seo_titles.append(f"{edited_title} - Version {len(seo_titles) + 1}")
    
    return TitleGenerationResponse(
        edited_title=edited_title,
        seo_titles=seo_titles[:3],  # 최대 3개만 반환
        raw_response=raw_response
    )

@router.post("/{news_key}/seo", response_model=TitleGenerationResponse)
async def generate_seo_title_with_key(
    news_key: str,
    db: SessionDep,
    body: TitleGenerationRequest,
    actor: User = Depends(resolve_actor_user)
):
    """
    GPT를 사용하여 SEO 최적화된 제목을 생성합니다 (news_key 포함).
    
    - 원본 제목을 개선하여 15단어 이내의 편집된 제목 생성
    - SEO에 최적화된 3개의 추가 제목 생성
    - 한국어 제목 입력 시 영어로 변환
    - URL의 news_key를 사용하여 히스토리 저장
    """
    import re
    
    # URL의 news_key를 우선 사용, 없으면 body의 news_key 사용
    final_news_key = news_key or body.news_key
    
    # gpt_generate_title 함수 호출
    result = await service.gpt_generate_title(
        db=db,
        input_text=body.input_text,
        selected_type=body.selected_type or "",
        data_type=body.data_type,
        model=body.model,
        guideline_text=body.guideline_text,
        news_key=final_news_key,
        user_id=actor.id if actor else None
    )
    
    # The service now returns a dict with parsed results
    if isinstance(result, dict):
        return TitleGenerationResponse(
            edited_title=result.get("edited_title", body.input_text),
            seo_titles=result.get("seo_titles", [])[:3],  # 최대 3개만 반환
            raw_response=result.get("raw_response", "")
        )
    
    # Fallback for backward compatibility (if service returns string)
    return _parse_seo_response(result, body.input_text)

@router.post("/seo", response_model=TitleGenerationResponse)
async def generate_seo_title(
    db: SessionDep,
    body: TitleGenerationRequest,
    actor: User = Depends(resolve_actor_user)
):
    """
    GPT를 사용하여 SEO 최적화된 제목을 생성합니다.
    
    - 원본 제목을 개선하여 15단어 이내의 편집된 제목 생성
    - SEO에 최적화된 3개의 추가 제목 생성
    - 한국어 제목 입력 시 영어로 변환
    """
    import re
    
    # gpt_generate_title 함수 호출
    result = await service.gpt_generate_title(
        db=db,
        input_text=body.input_text,
        selected_type=body.selected_type or "",
        data_type=body.data_type,
        model=body.model,
        guideline_text=body.guideline_text,
        news_key=body.news_key,
        user_id=actor.id if actor else None
    )
    
    # The service now returns a dict with parsed results
    if isinstance(result, dict):
        return TitleGenerationResponse(
            edited_title=result.get("edited_title", body.input_text),
            seo_titles=result.get("seo_titles", [])[:3],  # 최대 3개만 반환
            raw_response=result.get("raw_response", "")
        )
    
    # Fallback for backward compatibility (if service returns string)
    return _parse_seo_response(result, body.input_text)

def map_frontend_category_to_enum(frontend_category: Optional[str]) -> Optional[ArticleCategory]:
    """프론트엔드 category 값을 ArticleCategory enum으로 매핑"""
    # None이면 모든 카테고리를 의미 (필터링 없음)
    if frontend_category is None:
        return None

    normalized = frontend_category.strip().lower()

    mapping = {
        "headlines": ArticleCategory.TITLE,
        "headline": ArticleCategory.TITLE,
        "title": ArticleCategory.TITLE,
        "articles": ArticleCategory.BODY,
        "article": ArticleCategory.BODY,
        "body": ArticleCategory.BODY,
        "captions": ArticleCategory.CAPTION,
        "caption": ArticleCategory.CAPTION,
        "articles_translator": ArticleCategory.BODY,
        "article_translator": ArticleCategory.BODY,
        "translator": ArticleCategory.BODY,
        "seo": ArticleCategory.SEO,
        "seo_title": ArticleCategory.SEO,
    }

    if normalized not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{frontend_category}'. Must be one of: {list(mapping.keys())}"
        )

    return mapping[normalized]

@router.post("/save", response_model=CMSSaveResponse)
async def save_cms_data(
    body: CMSSaveRequest,
    db: SessionDep
):
    """
    CMS에서 전송한 데이터를 저장합니다.
    
    - news_key와 category로 단일 Article 저장/수정
    - category: headlines, articles, captions, articles_translator, seo 중 하나
    - author_id로 CMS 작성자 추적
    - 고정된 시스템 사용자(ID=1)로 저장하여 외래키 제약 조건 해결
    - 기존 데이터가 있으면 업데이트, 없으면 생성
    """
    # article_id 길이 검증 (DB 컬럼이 255자 제한)
    if len(body.article_id) > 255:
        raise HTTPException(
            status_code=400,
            detail=f"article_id must be 255 characters or less (got {len(body.article_id)} characters)"
        )
    
    # 프론트엔드 category를 enum으로 변환
    article_category = map_frontend_category_to_enum(body.category)
    
    # article_id를 news_key로 사용 (클라이언트가 제공한 ID 그대로 사용)
    article = await service.save_cms_article(
        db,
        body.article_id,  # article_id를 news_key로 사용
        article_category,
        body.content,
        body.author_id
    )
    # 응답용 카테고리 문자열 결정
    # 요구사항: CMS가 articles_translator로 저장 요청하면 DB에는 BODY로 저장하되,
    # Styler 페이지를 열 때는 번역 탭을 활성화하기 위해 응답 category는 'articles_translator'로 돌려준다.
    original_cat = (body.category or "").strip().lower()
    if original_cat in {"article_translator", "articles_translator", "translator"}:
        resp_category = "article_translator"
    else:
        # 소문자 단수형으로 매핑 (headline/article/caption/seo)
        cat_map = {
            ArticleCategory.TITLE: "headline",
            ArticleCategory.BODY: "article",
            ArticleCategory.CAPTION: "caption",
            ArticleCategory.SEO: "seo",
        }
        resp_category = cat_map.get(article.category, str(article.category).lower())
    return CMSSaveResponse(
        article_id=article.news_key,  # 동일한 값 반환
        news_key=article.news_key,
        category=resp_category,
    )

@router.get("/{article_identifier}", response_model=CMSGetResponse)
async def get_article(
    article_identifier: str,
    db: SessionDep,
    category: Optional[str] = Query(None, description="카테고리 (headlines, articles, captions, articles_translator, seo)")
):
    """
    Article을 조회합니다. news_key 또는 public_id 모두 지원합니다.
    
    - article_identifier: news_key 또는 public_id(UUID)
    - category가 제공되면 news_key + category로 조회
    - category가 없으면 identifier 타입을 자동 판단하여 조회
    - 하위 호환성: 기존 public_id 조회도 계속 지원
    """
    # category가 제공되면 enum으로 변환
    article_category = None
    if category:
        article_category = map_frontend_category_to_enum(category)
    
    # 유연한 조회 (news_key 또는 public_id 자동 판단)
    article = await service.get_article_by_identifier(
        db, 
        article_identifier, 
        article_category
    )
    
    # Caption인 경우 JSON 파싱하여 반환
    content = article.text
    if article.category == ArticleCategory.CAPTION:
        try:
            import json
            caption_data = json.loads(content)
            # 하위 호환성: 기존처럼 text로 반환하되, 구조화된 데이터 포함
            if isinstance(caption_data, dict) and "captions" in caption_data:
                # 첫 번째 caption을 기본으로 반환 (하위 호환성)
                captions = caption_data.get("captions", [])
                if captions:
                    content = captions[0].get("text", content)
                # 전체 caption 데이터는 별도 필드로 추가 가능
        except:
            pass  # JSON 파싱 실패 시 원본 텍스트 그대로 사용
    
    return CMSGetResponse(
        article_id=article.news_key,  # news_key를 article_id로 반환
        news_key=article.news_key,
        category=article.category.value,
        content=content,
        created_at=article.created_at,
        updated_at=article.updated_at
    )

@router.post("/history/restore", response_model=HistoryRestoreResponse)
async def restore_history(
    body: HistoryRestoreRequest,
    db: SessionDep,
    actor: User = Depends(resolve_actor_user)
):
    """
    히스토리 복원: 선택한 히스토리의 내용을 새로운 히스토리로 저장
    operation_type은 RESTORATION으로 설정됨
    """
    from .models import TextCorrectionHistory

    # 1. 원본 히스토리 조회
    stmt = select(TextCorrectionHistory).where(TextCorrectionHistory.id == body.history_id)
    result = await db.execute(stmt)
    original_history = result.scalar_one_or_none()

    if not original_history:
        raise HTTPException(status_code=404, detail=f"History ID {body.history_id} not found")

    # 2. 복원 히스토리 생성 (새로운 버전으로)
    restored_history = await service.create_article_with_history(
        db=db,
        news_key=body.news_key,
        category=body.category,
        user_id=actor.id,
        before_text=original_history.before_text,
        after_text=original_history.after_text,
        prompt=f"Restored from history #{original_history.id}",
        style_ids=[],  # 복원 시에는 스타일가이드 재적용 안 함
        operation_type=OperationType.RESTORATION,
        source_lang=original_history.source_lang,
        target_lang=original_history.target_lang,
        original_text=original_history.original_text,
        sentence_corrections=None  # 복원 시에는 문장별 교정 정보 없음
    )

    logger.info(f"History #{body.history_id} restored as new history #{restored_history.id} by user {actor.id}")

    return HistoryRestoreResponse(
        history_id=restored_history.id,
        version=restored_history.version,
        before_text=restored_history.before_text,
        after_text=restored_history.after_text
    )


@router.post("/correct/gpt5v2-stream")
async def correct_article_gpt5v2_stream(
    body: ArticleCorrectionRequest,
    db: SessionDep,
    actor: User = Depends(resolve_actor_user)
):
    """
    GPT-5 v2 파이프라인을 사용한 교정 스트리밍
    - DeepL 번역(자동 감지)
    - v2 Styler(3-Experts)로 분석 및 교정
    - 기존 SSE 포맷 유지, final_analysis에서 DB 저장
    """
    additional_prompt = body.prompt if body.prompt and body.prompt.strip() else None
    start_time = time.time()
    logger.info(
        f"Starting GPT-5 v2 correction stream for news_key={body.news_key}, category={body.category}, user_id={actor.id}"
    )

    async def generator():
        try:
            from .services.gpt5v2_correction import call_gpt5v2_correction_stream

            async for payload in call_gpt5v2_correction_stream(
                additional_prompt, body.text, body.category, db
            ):
                yield f"data: {payload}\n\n"

                # final_analysis에서 DB 저장
                try:
                    parsed = json.loads(payload)
                    if parsed.get("type") == "final_analysis":
                        is_local_mode = body.news_key.lower().startswith("local-")
                        if is_local_mode:
                            total_time = time.time() - start_time
                            logger.info(
                                f"Local mode detected for news_key={body.news_key}; skipping DB persistence. Total processing so far: {total_time:.3f}s"
                            )
                            continue

                        analysis_data = parsed.get("data", {})
                        style_ids = analysis_data.get("style_ids", [])
                        sentence_corrections = analysis_data.get("sentence_corrections", {})
                        full_text = analysis_data.get("full_text", "")

                        translation_data = analysis_data.get("translation", {}) or {}
                        before_en = translation_data.get("before_text")
                        source_lang = translation_data.get("source_lang")
                        target_lang = translation_data.get("target_lang")

                        if not before_en:
                            from .services.translation import translate_text
                            before_en, source_lang, target_lang = await translate_text(
                                body.text, source_lang=None, target_lang="EN-US"
                            )
                        source_lang = source_lang or "UNKNOWN"
                        target_lang = target_lang or "EN-US"

                        history = await service.create_article_with_history(
                            db=db,
                            news_key=body.news_key,
                            category=body.category,
                            user_id=actor.id,
                            before_text=before_en,
                            after_text=full_text,
                            prompt=additional_prompt,
                            style_ids=style_ids,
                            operation_type=OperationType.TRANSLATION_CORRECTION,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            original_text=body.text,
                            sentence_corrections=[
                                {
                                    "sentence_index": idx,
                                    "before_text": data.get("original", ""),
                                    "after_text": data.get("corrected", ""),
                                    "violations": data.get("violations", []),
                                }
                                for idx, data in sentence_corrections.items()
                            ] if sentence_corrections else None,
                        )

                        total_time = time.time() - start_time
                        logger.info(
                            f"GPT-5 v2 correction completed in {total_time:.3f}s, history_id: {history.id}"
                        )

                except Exception as parse_error:
                    logger.debug(f"Payload parsing for DB save (gpt5v2): {parse_error}")

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"GPT-5 v2 correction error after {error_time:.3f}s: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'처리 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )
