from __future__ import annotations

import logging
import uuid
from typing import List

from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
from shapely.geometry import MultiPolygon
from sqlalchemy import Select, bindparam, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import ClassCatalog, Region, Scene, Segment, Subregion
from app.schemas.schemas import (
    GeoJSONFeatureCollection,
    RegionPeriodItem,
    SegmentFeature,
    SegmentFeatureCollection,
    SegmentProperties,
    SegmentUpdateRequest,
    SegmentsImportResponse,
    SubregionHistoryItem,
    SubregionHistoryResponse,
)
from app.utils.geo import load_multipolygon, parse_bbox, to_geojson, to_wkb

logger = logging.getLogger(__name__)


class SegmentsService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def import_segments(
        self, db: Session, payload: GeoJSONFeatureCollection
    ) -> SegmentsImportResponse:
        segment_ids: list[str] = []
        for feature in payload.features:
            geom = self._prepare_geometry(feature.geometry.model_dump())
            scene = self._get_scene(db, feature.properties.sceneId)
            class_catalog = self._get_class(db, feature.properties.classId)
            segment = Segment(
                scene_id=scene.id,
                region_id=scene.region_id,
                class_id=class_catalog.id,
                class_name=class_catalog.name,
                periodo=feature.properties.periodo,
                confidence=feature.properties.confidence,
                area_m2=feature.properties.areaM2,
                source=feature.properties.source or "import",
                geometry=to_wkb(geom, srid=self.settings.default_epsg),
            )
            self._validate_against_region(db, segment.region_id, geom)
            db.add(segment)
            db.flush()
            segment_ids.append(str(segment.id))
        db.commit()
        return SegmentsImportResponse(inserted=len(segment_ids), segmentIds=segment_ids)

    def update_segment(
        self, db: Session, segment_id: str, payload: SegmentUpdateRequest
    ) -> Segment:
        segment = db.get(Segment, uuid.UUID(segment_id))
        if not segment:
            raise ValueError("Segment not found")
        if payload.classId and payload.classId != segment.class_id:
            catalog = self._get_class(db, payload.classId)
            segment.class_id = catalog.id
            segment.class_name = catalog.name
        if payload.confidence is not None:
            segment.confidence = payload.confidence
        if payload.notes is not None:
            segment.notes = payload.notes
        segment.is_manual_edited = True
        db.add(segment)
        db.commit()
        db.refresh(segment)
        return segment

    def segments_geojson(
        self,
        db: Session,
        region_id: str,
        periodo: str | None,
        class_ids: List[str] | None,
        bbox: str | None,
    ) -> SegmentFeatureCollection:
        stmt: Select[tuple[Segment]] = select(Segment).where(
            Segment.region_id == region_id,
        )
        if periodo:
            stmt = stmt.where(Segment.periodo == periodo)
        if class_ids:
            stmt = stmt.where(Segment.class_id.in_(class_ids))
        if bbox:
            min_lon, min_lat, max_lon, max_lat = parse_bbox(bbox)
            envelope = func.ST_MakeEnvelope(
                min_lon, min_lat, max_lon, max_lat, self.settings.default_epsg
            )
            stmt = stmt.where(func.ST_Intersects(Segment.geometry, envelope))
        segments = db.execute(stmt).scalars().all()
        features = [
            SegmentFeature(
                geometry=to_geojson(segment.geometry),
                properties=SegmentProperties(
                    segmentId=str(segment.id),
                    regionId=segment.region_id,
                    classId=segment.class_id,
                    className=segment.class_name,
                    areaM2=segment.area_m2,
                    periodo=segment.periodo,
                    confidence=segment.confidence,
                    source=segment.source,
                ),
            )
            for segment in segments
        ]
        return SegmentFeatureCollection(type="FeatureCollection", features=features)

    def region_periods(
        self,
        db: Session,
        region_id: str,
        from_period: str | None,
        to_period: str | None,
    ) -> List[RegionPeriodItem]:
        stmt = (
            select(
                Segment.periodo,
                func.count(Segment.id).label("segment_count"),
                func.max(Segment.updated_at).label("last_updated"),
            )
            .where(Segment.region_id == region_id)
            .group_by(Segment.periodo)
            .order_by(Segment.periodo)
        )
        if from_period:
            stmt = stmt.where(Segment.periodo >= from_period)
        if to_period:
            stmt = stmt.where(Segment.periodo <= to_period)
        rows = db.execute(stmt).all()
        return [
            RegionPeriodItem(
                periodo=row.periodo,
                regionId=region_id,
                segmentCount=row.segment_count,
                lastUpdated=row.last_updated,
            )
            for row in rows
        ]

    def subregion_history(
        self, db: Session, subregion_id: str
    ) -> SubregionHistoryResponse:
        subregion = db.get(Subregion, uuid.UUID(subregion_id))
        if not subregion:
            raise ValueError("Subregion not found")
        subgeom = bindparam(
            "subgeom", subregion.geometry, type_=Geometry("MULTIPOLYGON", 4326)
        )
        area_expr = func.ST_Area(
            func.ST_Transform(
                func.ST_Intersection(Segment.geometry, func.ST_GeomFromEWKB(subgeom)),
                3857,
            )
        )
        stmt = (
            select(
                Segment.periodo,
                Segment.class_id,
                Segment.class_name,
                func.sum(area_expr).label("area"),
            )
            .where(Segment.region_id == subregion.region_id)
            .where(func.ST_Intersects(Segment.geometry, func.ST_GeomFromEWKB(subgeom)))
            .group_by(Segment.periodo, Segment.class_id, Segment.class_name)
            .order_by(Segment.periodo)
        )
        rows = db.execute(stmt, {"subgeom": subregion.geometry}).all()
        history_map: dict[str, dict[str, tuple[str, float]]] = {}
        for row in rows:
            period_map = history_map.setdefault(row.periodo, {})
            period_map[row.class_id] = (row.class_name, row.area or 0.0)
        history_items: list[SubregionHistoryItem] = []
        ordered_periods = sorted(history_map.keys())
        prev_total = None
        for periodo in ordered_periods:
            classes = history_map[periodo]
            dominant_class_id, (dominant_class_name, area) = max(
                classes.items(), key=lambda item: item[1][1]
            )
            total_area = sum(area for _, area in classes.values())
            delta = None
            if prev_total is not None and prev_total > 0:
                delta = ((total_area - prev_total) / prev_total) * 100
            prev_total = total_area
            history_items.append(
                SubregionHistoryItem(
                    periodo=periodo,
                    areaM2=total_area,
                    dominantClassId=dominant_class_id,
                    dominantClassName=dominant_class_name,
                    deltaVsPrev=delta,
                )
            )
        return SubregionHistoryResponse(
            subregionId=str(subregion.id),
            regionId=subregion.region_id,
            history=history_items,
        )

    def _prepare_geometry(self, geometry: dict) -> MultiPolygon:
        return load_multipolygon(geometry)

    def _get_scene(self, db: Session, scene_id: str) -> Scene:
        try:
            scene_uuid = uuid.UUID(scene_id)
        except ValueError as exc:
            raise ValueError("sceneId must be a valid UUID") from exc
        scene = db.get(Scene, scene_uuid)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")
        return scene

    def _get_class(self, db: Session, class_id: str) -> ClassCatalog:
        catalog = db.get(ClassCatalog, class_id)
        if not catalog:
            raise ValueError(f"Class {class_id} not found in catalog")
        return catalog

    def _validate_against_region(
        self, db: Session, region_id: str, geometry: MultiPolygon
    ) -> None:
        region = db.get(Region, region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")
        if region.geometry is None:
            return
        region_geom = to_shape(region.geometry)
        if not region_geom.contains(geometry):
            logger.warning("Segment geometry slightly outside region %s", region_id)
