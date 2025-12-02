from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.schemas import SceneSegmentRequest, SegmentFeatureCollection, SegmentUpdateRequest, SegmentsImportResponse
from app.services.segmentation_raster_service import SegmentationRasterService
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/segments", tags=["segments"])
segments_service = SegmentsService()
segmentation_service = SegmentationRasterService()


@router.put("/{segment_id}")
def update_segment(
    segment_id: str,
    payload: SegmentUpdateRequest,
    db: Session = Depends(get_db),
):
    try:
        segment = segments_service.update_segment(db, segment_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "segmentId": str(segment.id),
        "classId": segment.class_id,
        "className": segment.class_name,
        "confidence": segment.confidence,
        "notes": segment.notes,
        "updatedAt": segment.updated_at,
    }


@router.get("/tiles", response_model=SegmentFeatureCollection)
def get_segments_tiles(
    regionId: str = Query(...),
    periodo: str = Query(...),
    classId: list[str] | None = Query(default=None),
    bbox: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> SegmentFeatureCollection:
    try:
        return segments_service.segments_geojson(db, regionId, periodo, classId, bbox)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/scenes/{scene_id}/segment", response_model=SegmentsImportResponse)
def segment_scene(
    scene_id: str,
    payload: SceneSegmentRequest,
    db: Session = Depends(get_db),
) -> SegmentsImportResponse:
    try:
        return segmentation_service.segment_scene(db, scene_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
