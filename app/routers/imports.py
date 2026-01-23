from __future__ import annotations

import asyncio
import json
import shutil
import time
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

# ✅ Progress cache for reducing frequent memory/DB hits
_progress_cache = {}
_cache_timestamp = {}


@router.post("/scenes", response_model=SceneUploadResponse)
async def upload_scene(
    scene_file: UploadFile = File(...),
    capture_date: str = Form(...),
    epsg: int = Form(...),
    sensor: str = Form(...),
    region_id: str = Form(...),
    db: Session = Depends(get_db),
) -> SceneUploadResponse:
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not region_id or not region_id.strip():
        raise HTTPException(status_code=400, detail="region_id is required")
    if not capture_date or not capture_date.strip():
        raise HTTPException(status_code=400, detail="capture_date is required")
    if not sensor or not sensor.strip():
        raise HTTPException(status_code=400, detail="sensor is required")
    
    logger.info(f"Upload scene request: region_id={region_id}, capture_date={capture_date}, epsg={epsg}, sensor={sensor}")
    
    try:
        # Convert capture_date string to date object
        from datetime import datetime
        capture_date_obj = datetime.strptime(capture_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Expected YYYY-MM-DD, got {capture_date}"
        )
    
    region = db.get(Region, region_id)
    if not region:
        logger.error(f"Region not found: {region_id}")
        raise HTTPException(status_code=404, detail=f"Region not found: {region_id}")
    
    scene = Scene(
        region_id=region_id,
        capture_date=capture_date_obj,
        epsg=epsg,
        sensor=sensor,
        raster_path="",
    )
    db.add(scene)
    db.flush()
    scene_dir = Path(settings.scenes_dir) / str(scene.id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(scene_file.filename or "scene.tif").suffix or ".tif"
    target_path = scene_dir / f"scene{extension}"
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(scene_file.file, buffer)
    scene.raster_path = str(target_path)
    db.add(scene)
    db.commit()
    db.refresh(scene)
    
    # Start segmentation in background thread
    try:
        scene_file.file.seek(0)
        tiff_bytes_content = await scene_file.read()
        
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
async def get_progress(scene_id: str):
    """Get progress for a scene (polling-based, with aggressive caching)."""
    progress_service = get_progress_service()
    
    # ✅ Cache with 500ms TTL to reduce backend hits on tunnel latency
    cache_key = f"progress_{scene_id}"
    now = time.time()
    
    if cache_key in _progress_cache:
        cached_time = _cache_timestamp.get(cache_key, 0)
        if (now - cached_time) < 0.5:  # 500ms cache TTL
            return _progress_cache[cache_key]
    
    progress = progress_service.get_progress(scene_id)
    
    if progress:
        result = progress.to_dict()
    else:
        result = {
            "sceneId": scene_id,
            "status": "pending",
            "currentStep": 0,
            "totalSteps": 0,
            "steps": [],
            "errorMessage": "",
            "result": None
        }
    
    # ✅ Store in cache
    _progress_cache[cache_key] = result
    _cache_timestamp[cache_key] = now
    
    # ✅ Cleanup old cache entries (older than 2 minutes)
    keys_to_delete = [key for key in _progress_cache if (now - _cache_timestamp.get(key, 0)) > 120]
    for key in keys_to_delete:
        del _progress_cache[key]
        del _cache_timestamp[key]
    
    return result


@router.post("/segments", response_model=SegmentsImportResponse)
async def import_segments(
    feature_collection: GeoJSONFeatureCollection,
    db: Session = Depends(get_db),
) -> SegmentsImportResponse:
    try:
        return segments_service.import_segments(db, feature_collection)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/clear-all-data")
async def clear_all_data(db: Session = Depends(get_db)):
    """
    Elimina todos los datos: escenas, segmentaciones y archivos del directorio sigma-backend/data/scenes
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Obtener todas las escenas
        all_scenes = db.query(Scene).all()
        
        deleted_count = 0
        for scene in all_scenes:
            try:
                # Eliminar carpeta del disco si existe
                scene_dir = Path(settings.scenes_dir) / str(scene.id)
                if scene_dir.exists():
                    shutil.rmtree(scene_dir)
                    logger.info(f"Deleted scene directory: {scene_dir}")
                
                # Usar SQLAlchemy para eliminar la escena por ID (mejor forma de manejar cascadas)
                db.query(Scene).filter(Scene.id == scene.id).delete(synchronize_session=False)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting scene {scene.id}: {e}")
                raise
        
        # Confirmar cambios en la base de datos
        db.commit()
        
        logger.info(f"Successfully cleared all data. Deleted {deleted_count} scenes and their associated files.")
        
        return {
            "message": "All data successfully deleted",
            "deleted_scenes_count": deleted_count
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing all data: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")
