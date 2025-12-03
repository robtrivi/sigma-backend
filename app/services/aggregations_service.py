from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import AggregationSummary, Segment
from app.schemas.schemas import (
    AggregationRebuildRequest,
    DistributionItem,
    RegionSummaryResponse,
    TrendItem,
)

logger = logging.getLogger(__name__)


class AggregationsService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rebuild(
        self, db: Session, request: AggregationRebuildRequest
    ) -> AggregationSummary:
        summary = self.snapshot_period(db, request.regionId, request.periodo)
        existing = (
            db.execute(
                select(AggregationSummary).where(
                    AggregationSummary.region_id == request.regionId,
                    AggregationSummary.periodo == request.periodo,
                )
            )
            .scalars()
            .first()
        )
        if existing:
            existing.total_area_m2 = summary["total_area"]
            existing.green_coverage = summary["green_coverage"]
            existing.distribution_json = summary["distribution"]
            existing.trend_json = summary["trend"]
            db.add(existing)
            db.commit()
            db.refresh(existing)
            return existing
        new_summary = AggregationSummary(
            region_id=request.regionId,
            periodo=request.periodo,
            total_area_m2=summary["total_area"],
            green_coverage=summary["green_coverage"],
            distribution_json=summary["distribution"],
            trend_json=summary["trend"],
        )
        db.add(new_summary)
        db.commit()
        db.refresh(new_summary)
        return new_summary

    def region_summary(
        self,
        db: Session,
        region_id: str,
        periodo: str,
        segment_ids: List[str] | None = None,
    ) -> RegionSummaryResponse:
        distribution_data: List[dict]
        trend_data: List[dict]
        green_coverage: float
        summary = self._get_summary(db, region_id, periodo)
        if segment_ids:
            distribution_data, green_coverage = self._aggregate_filtered(
                db, region_id, periodo, segment_ids
            )
            trend_data = summary.trend_json if summary else []
        else:
            if not summary:
                request = AggregationRebuildRequest(regionId=region_id, periodo=periodo)
                summary = self.rebuild(db, request)
            distribution_data = summary.distribution_json if summary else []
            green_coverage = summary.green_coverage if summary else 0.0
            trend_data = summary.trend_json if summary else []
        count_stmt = select(func.count(Segment.id)).where(
            Segment.region_id == region_id,
            Segment.periodo == periodo,
        )
        if segment_ids:
            parsed_ids = [uuid.UUID(seg_id) for seg_id in segment_ids]
            count_stmt = count_stmt.where(Segment.id.in_(parsed_ids))
        segments_visible = db.execute(count_stmt).scalar_one()
        distribution = [
            DistributionItem(
                classId=item["classId"],
                className=item["className"],
                percentage=item["percentage"],
                areaM2=item["areaM2"],
            )
            for item in distribution_data
        ]
        trend = [
            TrendItem(periodo=item["periodo"], value=item["value"])
            for item in trend_data
        ]
        messages: list[str] = []
        if green_coverage < 20:
            messages.append("Cobertura verde por debajo del umbral recomendado.")
        if green_coverage > 60:
            messages.append("Cobertura verde saludable en el periodo.")
        return RegionSummaryResponse(
            regionId=region_id,
            periodo=periodo,
            segmentsVisible=segments_visible,
            coberturaVerde=green_coverage,
            distribution=distribution,
            trend=trend,
            messages=messages,
        )

    def _aggregate_filtered(
        self,
        db: Session,
        region_id: str,
        periodo: str,
        segment_ids: List[str],
    ) -> tuple[List[dict], float]:
        parsed_ids = [uuid.UUID(seg_id) for seg_id in segment_ids]
        rows = db.execute(
            select(Segment.class_id, Segment.class_name, func.sum(Segment.area_m2))
            .where(
                Segment.region_id == region_id,
                Segment.periodo == periodo,
                Segment.id.in_(parsed_ids),
            )
            .group_by(Segment.class_id, Segment.class_name)
        ).all()
        total_area = sum(row[2] or 0 for row in rows)
        if total_area == 0:
            return ([], 0.0)
        distribution = []
        green_area = 0.0
        for class_id, class_name, area in rows:
            value = area or 0.0
            if class_id in self.settings.green_class_ids:
                green_area += value
            distribution.append(
                {
                    "classId": class_id,
                    "className": class_name,
                    "percentage": (value / total_area) * 100,
                    "areaM2": value,
                }
            )
        coverage = (green_area / total_area) * 100 if total_area else 0.0
        return distribution, coverage

    def _get_summary(
        self, db: Session, region_id: str, periodo: str
    ) -> AggregationSummary | None:
        return (
            db.execute(
                select(AggregationSummary).where(
                    AggregationSummary.region_id == region_id,
                    AggregationSummary.periodo == periodo,
                )
            )
            .scalars()
            .first()
        )

    def snapshot_period(
        self, db: Session, region_id: str, periodo: str
    ) -> Dict[str, Any]:
        rows = db.execute(
            select(
                Segment.class_id,
                Segment.class_name,
                func.sum(Segment.area_m2).label("area"),
            )
            .where(
                Segment.region_id == region_id,
                Segment.periodo == periodo,
            )
            .group_by(Segment.class_id, Segment.class_name)
        ).all()
        total_area = sum(row.area or 0.0 for row in rows)
        distribution = []
        green_area = 0.0
        for row in rows:
            area_value = row.area or 0.0
            if row.class_id in self.settings.green_class_ids:
                green_area += area_value
            percentage = (area_value / total_area) * 100 if total_area else 0.0
            distribution.append(
                {
                    "classId": row.class_id,
                    "className": row.class_name,
                    "percentage": percentage,
                    "areaM2": area_value,
                }
            )
        coverage = (green_area / total_area) * 100 if total_area else 0.0
        trend = self._build_trend(db, region_id)
        return {
            "total_area": total_area,
            "distribution": distribution,
            "green_coverage": coverage,
            "trend": trend,
        }

    def _build_trend(self, db: Session, region_id: str) -> List[dict]:
        green_case = case(
            (Segment.class_id.in_(self.settings.green_class_ids), Segment.area_m2),
            else_=0.0,
        )
        stmt: Select = (
            select(
                Segment.periodo,
                func.sum(green_case).label("green_area"),
                func.sum(Segment.area_m2).label("total_area"),
            )
            .where(Segment.region_id == region_id)
            .group_by(Segment.periodo)
            .order_by(Segment.periodo.desc())
            .limit(6)
        )
        rows = db.execute(stmt).all()
        trend = []
        for row in reversed(rows):
            total = row.total_area or 0.0
            green_area = row.green_area or 0.0
            value = (green_area / total) * 100 if total else 0.0
            trend.append({"periodo": row.periodo, "value": value})
        return trend
