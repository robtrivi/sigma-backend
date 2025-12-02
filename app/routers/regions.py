from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.schemas import RegionPeriodItem, RegionSummaryResponse
from app.services.aggregations_service import AggregationsService
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/regions", tags=["regions"])
segments_service = SegmentsService()
aggregations_service = AggregationsService()


@router.get("/{region_id}/periods", response_model=list[RegionPeriodItem])
def list_region_periods(
    region_id: str,
    from_period: str | None = Query(None, alias="from"),
    to_period: str | None = Query(None, alias="to"),
    db: Session = Depends(get_db),
) -> list[RegionPeriodItem]:
    return segments_service.region_periods(db, region_id, from_period, to_period)


@router.get("/{region_id}/summary", response_model=RegionSummaryResponse)
def region_summary(
    region_id: str,
    periodo: str = Query(...),
    segmentIds: list[str] | None = Query(default=None),
    db: Session = Depends(get_db),
) -> RegionSummaryResponse:
    try:
        return aggregations_service.region_summary(db, region_id, periodo, segmentIds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
