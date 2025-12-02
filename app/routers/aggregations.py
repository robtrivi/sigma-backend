from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.schemas import (
    AggregationRebuildRequest,
    AggregationSummaryRead,
    DistributionItem,
    TrendItem,
)
from app.services.aggregations_service import AggregationsService

router = APIRouter(prefix="/aggregations", tags=["aggregations"])
service = AggregationsService()


@router.post("/rebuild", response_model=AggregationSummaryRead)
def rebuild_aggregation(
    payload: AggregationRebuildRequest,
    db: Session = Depends(get_db),
) -> AggregationSummaryRead:
    try:
        summary = service.rebuild(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    distribution = [
        DistributionItem(
            classId=item["classId"],
            className=item["className"],
            percentage=item["percentage"],
            areaM2=item["areaM2"],
        )
        for item in summary.distribution_json
    ]
    trend = [TrendItem(periodo=item["periodo"], value=item["value"]) for item in summary.trend_json]
    return AggregationSummaryRead(
        regionId=summary.region_id,
        periodo=summary.periodo,
        totalAreaM2=summary.total_area_m2,
        greenCoverage=summary.green_coverage,
        distribution=distribution,
        trend=trend,
    )
