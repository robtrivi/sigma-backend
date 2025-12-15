from __future__ import annotations

import asyncio
import json
import shutil
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import threading

from app.core.config import get_settings
from app.core.db import get_db
from app.models import Region, Scene
from app.schemas.schemas import (
    GeoJSONFeatureCollection,
    SceneUploadResponse,
    SegmentsImportResponse,
)
from app.services.dl_segmentation_service import DLSegmentationService
from app.services.progress_service import get_progress_service
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
    
    # Start segmentation in background thread
    try:
        sceneFile.file.seek(0)
        tiff_bytes_content = await sceneFile.read()
        
        # Create a new database session for the background thread
        def run_segmentation():
            from app.core.db import SessionLocal
            db_bg = SessionLocal()
            try:
                dl_segmentation_service.segment_scene(db_bg, str(scene.id), tiff_bytes_content)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Automatic segmentation failed for scene {scene.id}: {e}")
                from app.services.progress_service import get_progress_service
                progress_service = get_progress_service()
                progress_service.error_progress(str(scene.id), str(e))
            finally:
                db_bg.close()
        
        # Execute in background thread
        thread = threading.Thread(target=run_segmentation, daemon=True)
        thread.start()
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error starting segmentation for scene {scene.id}: {e}")
    
    return SceneUploadResponse(
        sceneId=str(scene.id),
        regionId=scene.region_id,
        captureDate=scene.capture_date,
        epsg=scene.epsg,
        sensor=scene.sensor,
        rasterPath=scene.raster_path,
    )


@router.get("/progress/{scene_id}")
async def stream_progress(scene_id: str):
    """Stream progress updates for a scene using Server-Sent Events."""
    progress_service = get_progress_service()
    
    async def event_generator():
        # Enviar progreso inicial
        progress = progress_service.get_progress(scene_id)
        if progress:
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Esperar actualizaciones cada 200ms hasta que esté completado
        max_iterations = 300  # ~60 segundos máximo
        iteration = 0
        
        while iteration < max_iterations:
            await asyncio.sleep(0.2)
            iteration += 1
            
            progress = progress_service.get_progress(scene_id)
            if progress:
                yield f"data: {json.dumps(progress.to_dict())}\n\n"
                
                # Detener si se completó o hubo error
                if progress.status in ("completed", "error"):
                    break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
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
