from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.models import Region, Scene
from app.schemas.schemas import GeoJSONFeatureCollection, SceneUploadResponse, SegmentsImportResponse
from app.services.segments_service import SegmentsService

router = APIRouter(prefix="/imports", tags=["imports"])
settings = get_settings()
segments_service = SegmentsService(settings)


@router.post("/scenes", response_model=SceneUploadResponse)
async def upload_scene(
    sceneFile: UploadFile = File(...),
    captureDate: date = Form(...),
    epsg: int = Form(...),
    sensor: str = Form(...),
    regionId: str = Form(...),
    db: Session = Depends(get_db),
) -> SceneUploadResponse:
    region = db.get(Region, regionId)
    if not region:
        raise HTTPException(status_code=404, detail="Region not found")
    scene = Scene(
        region_id=regionId,
        capture_date=captureDate,
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
