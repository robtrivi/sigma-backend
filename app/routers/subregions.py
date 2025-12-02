from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.schemas import SubregionHistoryResponse
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/subregions", tags=["subregions"])
segments_service = SegmentsService()


@router.get("/{subregion_id}/history", response_model=SubregionHistoryResponse)
def subregion_history(subregion_id: str, db: Session = Depends(get_db)) -> SubregionHistoryResponse:
    try:
        return segments_service.subregion_history(db, subregion_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
