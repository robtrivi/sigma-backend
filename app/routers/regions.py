from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Region
from app.schemas.schemas import RegionCreate, RegionPeriodItem, RegionRead, RegionSummaryResponse
from app.services.aggregations_service import AggregationsService
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/regions", tags=["regions"])
segments_service = SegmentsService()
aggregations_service = AggregationsService()


@router.get("/", response_model=list[RegionRead])
def list_regions(db: Session = Depends(get_db)) -> list[RegionRead]:
    regions = db.execute(select(Region)).scalars().all()
    return [
        RegionRead(
            id=region.id,
            name=region.name,
            geometry=None,
        )
        for region in regions
    ]


@router.post("/", response_model=RegionRead, status_code=201)
def create_region(
    region_data: RegionCreate,
    db: Session = Depends(get_db),
) -> RegionRead:
    existing = db.get(Region, region_data.id)
    if existing:
        raise HTTPException(status_code=400, detail="Region already exists")
    
    region = Region(
        id=region_data.id,
        name=region_data.name,
        geometry=None,
    )
    db.add(region)
    db.commit()
    db.refresh(region)
    
    return RegionRead(
        id=region.id,
        name=region.name,
        geometry=None,
    )


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
