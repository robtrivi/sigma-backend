from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session
import uuid

from app.core.db import get_db
from app.models import SegmentationResult, SegmentationSummary
from app.schemas.schemas import (
    SegmentFeatureCollection,
    SegmentUpdateRequest,
    SegmentsImportResponse,
    SegmentationCoverageRead,
    SegmentationCoverageSummary,
)
from app.services.dl_segmentation_service import DLSegmentationService
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/segments", tags=["segments"])
segments_service = SegmentsService()
dl_segmentation_service = DLSegmentationService()


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
async def segment_scene(
    scene_id: str,
    tiff_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> SegmentsImportResponse:
    try:
        tiff_bytes = await tiff_file.read()
        return dl_segmentation_service.segment_scene(db, scene_id, tiff_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(exc)}") from exc


@router.get("/coverage/{scene_id}", response_model=SegmentationCoverageRead)
def get_coverage(
    scene_id: str,
    db: Session = Depends(get_db),
):
    """
    Obtiene cobertura por píxeles previamente calculada y guardada en BD.
    
    Args:
        scene_id: UUID de la escena
        
    Returns:
        SegmentationCoverageRead con detalles de cobertura por clase
    """
    try:
        scene_uuid = uuid.UUID(scene_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scene_id format")
    
    result = db.query(SegmentationResult).filter(
        SegmentationResult.scene_id == scene_uuid
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No segmentation results found for scene {scene_id}"
        )
    
    return SegmentationCoverageRead(
        scene_id=str(result.scene_id),
        total_pixels=result.total_pixels,
        image_resolution=result.image_resolution,
        coverage_by_class=result.coverage_by_class,
        created_at=result.created_at,
    )


@router.get("/coverage-summary/{scene_id}", response_model=SegmentationCoverageSummary)
def get_coverage_summary(
    scene_id: str,
    db: Session = Depends(get_db),
):
    """
    Obtiene resumen rápido de cobertura dominante (sin parsear JSON completo).
    Ideal para dashboards con muchas escenas.
    
    Args:
        scene_id: UUID de la escena
        
    Returns:
        SegmentationCoverageSummary con clases dominantes
    """
    try:
        scene_uuid = uuid.UUID(scene_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scene_id format")
    
    result = db.query(SegmentationResult).filter(
        SegmentationResult.scene_id == scene_uuid
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No segmentation results found for scene {scene_id}"
        )
    
    summary = db.query(SegmentationSummary).filter(
        SegmentationSummary.segmentation_result_id == result.id
    ).first()
    
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No coverage summary found for scene {scene_id}"
        )
    
    return SegmentationCoverageSummary(
        scene_id=str(result.scene_id),
        dominant_class=summary.dominant_class_name,
        dominant_percentage=summary.dominant_class_percentage,
        secondary_class=summary.second_class_name,
        secondary_percentage=summary.second_class_percentage,
    )
