from fastapi import APIRouter, Depends, HTTPException, status

from ..auth.dependencies import CurrentUser, require_admin
from ..database import SessionDep
from .models import StyleCategory
from . import service
from .schemas import (
    BulkResult,
    BulkStyleGuideCreate,
    StyleGuideCreate,
    StyleGuideJSONInput,
    StyleGuideOut,
    StyleGuideUpdate,
)

router = APIRouter(prefix="/styleguides", tags=["styleguides"])


@router.post("/", response_model=StyleGuideOut, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
async def create_styleguide(body: StyleGuideCreate, db: SessionDep):
    return await service.create_styleguide(db, body)


@router.get("/", response_model=list[StyleGuideOut])
async def list_styleguides(
    db: SessionDep,
    category: StyleCategory | None = None,
    name: str | None = None,
    version: int | None = None,
    q: str | None = None,
    include_deleted: bool = False,
    skip: int = 0,
    limit: int = 50,
):
    return await service.list_styleguides(
        db,
        category=category,
        name=name,
        version=version,
        q=q,
        include_deleted=include_deleted,
        skip=skip,
        limit=limit,
    )


@router.get("/{style_id}", response_model=StyleGuideOut)
async def get_styleguide(style_id: int, db: SessionDep):
    sg = await service.get_by_id(db, style_id)
    if not sg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="StyleGuide not found")
    return sg


@router.patch("/{style_id}", response_model=StyleGuideOut, dependencies=[Depends(require_admin)])
async def update_styleguide(style_id: int, body: StyleGuideUpdate, db: SessionDep):
    sg = await service.get_by_id(db, style_id, include_deleted=True)
    if not sg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="StyleGuide not found")
    return await service.update_styleguide(db, sg, body)


@router.delete("/{style_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
async def delete_styleguide(style_id: int, db: SessionDep, hard: bool = False):
    sg = await service.get_by_id(db, style_id, include_deleted=True)
    if not sg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="StyleGuide not found")
    if hard:
        await service.hard_delete(db, sg)
    else:
        await service.soft_delete(db, sg)
    return


@router.post("/{style_id}/restore", response_model=StyleGuideOut, dependencies=[Depends(require_admin)])
async def restore_styleguide(style_id: int, db: SessionDep):
    sg = await service.get_by_id(db, style_id, include_deleted=True)
    if not sg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="StyleGuide not found")
    return await service.restore(db, sg)


@router.post("/bulk", response_model=BulkResult, dependencies=[Depends(require_admin)])
async def bulk_styleguides(body: BulkStyleGuideCreate, db: SessionDep):
    """기존 형식과 JSON 형식 모두 지원하는 대량 임포트"""
    if body.style_guides:
        # JSON 형식 처리
        json_data = {"style_guides": [item.model_dump() for item in body.style_guides]}
        return await service.bulk_import_json(db, json_data, mode=body.mode)
    elif body.items:
        # 기존 형식 처리
        return await service.bulk_import(db, body.items, mode=body.mode)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either 'items' or 'style_guides' must be provided")


@router.post("/import-json", response_model=BulkResult, dependencies=[Depends(require_admin)])
async def import_json_styleguides(json_data: dict, db: SessionDep):
    """JSON 형식 전용 대량 임포트 엔드포인트"""
    return await service.bulk_import_json(db, json_data, mode="error")

