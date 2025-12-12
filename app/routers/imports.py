from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.models import Region, Scene
from app.schemas.schemas import (
    GeoJSONFeatureCollection,
    SceneUploadResponse,
    SegmentsImportResponse,
)
from app.services.dl_segmentation_service import DLSegmentationService
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/imports", tags=["imports"])
settings = get_settings()
segments_service = SegmentsService(settings)
dl_segmentation_service = DLSegmentationService(settings)


@router.post("/scenes", response_model=SceneUploadResponse)
async def upload_scene(
    sceneFile: UploadFile = File(...),
    captureDate: str = Form(...),
    epsg: int = Form(...),
    sensor: str = Form(...),
    regionId: str = Form(...),
    db: Session = Depends(get_db),
) -> SceneUploadResponse:
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not regionId or not regionId.strip():
        raise HTTPException(status_code=400, detail="regionId is required")
    if not captureDate or not captureDate.strip():
        raise HTTPException(status_code=400, detail="captureDate is required")
    if not sensor or not sensor.strip():
        raise HTTPException(status_code=400, detail="sensor is required")
    
    logger.info(f"Upload scene request: regionId={regionId}, captureDate={captureDate}, epsg={epsg}, sensor={sensor}")
    
    try:
        # Convert captureDate string to date object
        from datetime import datetime
        capture_date_obj = datetime.strptime(captureDate, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Expected YYYY-MM-DD, got {captureDate}"
        )
    
    region = db.get(Region, regionId)
    if not region:
        logger.error(f"Region not found: {regionId}")
        raise HTTPException(status_code=404, detail=f"Region not found: {regionId}")
    
    scene = Scene(
        region_id=regionId,
        capture_date=capture_date_obj,
        epsg=epsg,
        sensor=sensor,
        raster_path="",
    )
    db.add(scene)
    db.flush()
    scene_dir = Path(settings.scenes_dir) / str(scene.id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(sceneFile.filename or "scene.tif").suffix or ".tif"
    target_path = scene_dir / f"scene{extension}"
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(sceneFile.file, buffer)
    scene.raster_path = str(target_path)
    db.add(scene)
    db.commit()
    db.refresh(scene)
    
    try:
        sceneFile.file.seek(0)
        tiff_bytes = await sceneFile.read()
        dl_segmentation_service.segment_scene(db, str(scene.id), tiff_bytes)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Automatic segmentation failed for scene {scene.id}: {e}")
    
    return SceneUploadResponse(
        sceneId=str(scene.id),
        regionId=scene.region_id,
        captureDate=scene.capture_date,
        epsg=scene.epsg,
        sensor=scene.sensor,
        rasterPath=scene.raster_path,
    )


@router.post("/segments", response_model=SegmentsImportResponse)
async def import_segments(
    feature_collection: GeoJSONFeatureCollection,
    db: Session = Depends(get_db),
) -> SegmentsImportResponse:
    try:
        return segments_service.import_segments(db, feature_collection)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
