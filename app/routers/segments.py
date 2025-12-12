from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import uuid
from pathlib import Path

from app.core.db import get_db
from app.core.config import get_settings
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


@router.get("/mask/{scene_id}")
def get_predicted_mask(scene_id: str, db: Session = Depends(get_db)):
    """
    Devuelve la máscara RGB predicha como imagen GeoTIFF
    """
    settings = get_settings()
    
    # Construir ruta de la máscara
    mask_path = Path(settings.data_dir) / "scenes" / scene_id / "mask_predicted_rgb.tif"
    
    if not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Predicted mask not found for scene {scene_id}"
        )
    
    return FileResponse(
        path=mask_path,
        media_type="image/tiff",
        filename=f"mask_{scene_id}.tif"
    )


@router.get("/debug/scenes-dir")
def debug_scenes_dir(db: Session = Depends(get_db)):
    """
    Endpoint de debug para verificar el directorio de escenas
    """
    import logging
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    scenes_dir = Path(settings.data_dir) / "scenes"
    
    logger.info(f"Scenes dir: {scenes_dir}")
    logger.info(f"Scenes dir exists: {scenes_dir.exists()}")
    
    if scenes_dir.exists():
        subdirs = list(scenes_dir.iterdir())
        logger.info(f"Subdirectories: {[d.name for d in subdirs]}")
        
        return {
            "scenes_dir": str(scenes_dir),
            "exists": True,
            "subdirectories": [d.name for d in subdirs if d.is_dir()],
            "total": len(subdirs)
        }
    else:
        return {
            "scenes_dir": str(scenes_dir),
            "exists": False
        }


@router.get("/mask-info/{scene_id}")
def get_mask_info(scene_id: str, db: Session = Depends(get_db)):
    """
    Devuelve metadatos de georeferenciación de la máscara con imagen en base64
    """
    import rasterio
    import base64
    import numpy as np
    import io
    import logging
    from PIL import Image
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    try:
        mask_path = Path(settings.data_dir) / "scenes" / scene_id / "mask_predicted_rgb.tif"
        logger.info(f"[MASK-INFO] Buscando máscara en: {mask_path}")
        logger.info(f"[MASK-INFO] Path exists: {mask_path.exists()}")
        
        if not mask_path.exists():
            logger.warning(f"[MASK-INFO] Máscara no encontrada")
            raise HTTPException(status_code=404, detail=f"Mask not found for scene {scene_id}")
        
        # Abrir GeoTIFF
        with rasterio.open(str(mask_path)) as src:
            logger.info(f"[MASK-INFO] GeoTIFF abierto: {src.width}x{src.height}")
            
            # Obtener georeferenciación
            bbox = src.bounds
            crs_epsg = None
            
            try:
                if src.crs is not None:
                    crs_epsg = src.crs.to_epsg()
                    logger.info(f"[MASK-INFO] CRS obtenido del archivo: EPSG:{crs_epsg}")
            except Exception as crs_error:
                logger.warning(f"[MASK-INFO] Error al leer CRS: {crs_error}")
            
            # Si no hay CRS pero los bounds parecen ser UTM, inferir el código EPSG
            if crs_epsg is None:
                logger.info(f"[MASK-INFO] CRS no detectado, intentando inferir desde bounds")
                # Los bounds UTM para Ecuador son típicamente:
                # Zona 17S: X entre 400000-800000, Y entre 9600000-10000000
                # Zona 18S: X entre 100000-500000, Y entre 9600000-10000000
                if 600000 < bbox.left < 700000 and 9700000 < bbox.bottom < 9800000:
                    crs_epsg = 32717  # UTM Zona 17S
                    logger.info(f"[MASK-INFO] Inferido EPSG:32717 (UTM 17S)")
                elif 100000 < bbox.left < 400000 and 9600000 < bbox.bottom < 10000000:
                    crs_epsg = 32718  # UTM Zona 18S
                    logger.info(f"[MASK-INFO] Inferido EPSG:32718 (UTM 18S)")
                else:
                    crs_epsg = 32717  # Default para Ecuador
                    logger.info(f"[MASK-INFO] Default EPSG:32717 (UTM 17S)")
            
            logger.info(f"[MASK-INFO] CRS final: EPSG:{crs_epsg}")
            
            # Leer bandas RGB
            try:
                data = src.read([1, 2, 3])
                logger.info(f"[MASK-INFO] Datos leídos: shape={data.shape}")
            except:
                # Si no hay 3 bandas, leer todas
                data = src.read()
                if data.shape[0] >= 3:
                    data = data[:3]
                elif data.shape[0] == 1:
                    # Replicar banda única a RGB
                    data = np.repeat(data, 3, axis=0)
                logger.info(f"[MASK-INFO] Fallback: datos={data.shape}")
            
            # Convertir a (H, W, C)
            img_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
            logger.info(f"[MASK-INFO] Array: {img_array.shape}, min={img_array.min()}, max={img_array.max()}")
            
            # Convertir a PNG
            img = Image.fromarray(img_array, 'RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.info(f"[MASK-INFO] PNG encoded: {len(img_base64)} chars")
            
            return {
                "width": src.width,
                "height": src.height,
                "crs": crs_epsg,
                "bounds": {
                    "minX": float(bbox.left),
                    "minY": float(bbox.bottom),
                    "maxX": float(bbox.right),
                    "maxY": float(bbox.top)
                },
                "image": f"data:image/png;base64,{img_base64}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MASK-INFO] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mask-filtered/{scene_id}")
def get_mask_filtered(scene_id: str, classes: str = "", db: Session = Depends(get_db)):
    """
    Devuelve máscara RGB filtrada mostrando solo clases seleccionadas
    
    Args:
        scene_id: UUID de la escena
        classes: Clases separadas por comas (ej: "vegetation,grass,tree")
    """
    import rasterio
    import base64
    import numpy as np
    import io
    import logging
    from PIL import Image
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    try:
        # Mapeo de nombres de clase a colores RGB
        CLASS_COLORS_RGB = {
            0: (0, 0, 0),      # unlabeled
            1: (128, 64, 128),  # paved-area
            2: (130, 76, 0),    # dirt
            3: (0, 102, 0),     # grass
            4: (112, 103, 87),  # gravel
            5: (28, 42, 168),   # water
            6: (48, 41, 30),    # rocks
            7: (0, 50, 89),     # pool
            8: (107, 142, 35),  # vegetation
            9: (70, 70, 70),    # roof
            10: (102, 102, 156), # wall
            11: (254, 228, 12), # window
            12: (254, 148, 12), # door
            13: (190, 153, 153), # fence
            14: (153, 153, 153), # fence-pole
            15: (255, 22, 96),  # person
            16: (102, 51, 0),   # dog
            17: (9, 143, 150),  # car
            18: (119, 11, 32),  # bicycle
            19: (51, 51, 0),    # tree
            20: (190, 250, 190), # bald-tree
            21: (112, 150, 146), # ar-marker
            22: (2, 135, 115),  # obstacle
            23: (255, 0, 0),    # conflicting
        }
        
        CLASS_IDS = {
            'unlabeled': 0, 'paved-area': 1, 'dirt': 2, 'grass': 3, 'gravel': 4,
            'water': 5, 'rocks': 6, 'pool': 7, 'vegetation': 8, 'roof': 9,
            'wall': 10, 'window': 11, 'door': 12, 'fence': 13, 'fence-pole': 14,
            'person': 15, 'dog': 16, 'car': 17, 'bicycle': 18, 'tree': 19,
            'bald-tree': 20, 'ar-marker': 21, 'obstacle': 22, 'conflicting': 23
        }
        
        mask_path = Path(settings.data_dir) / "scenes" / scene_id / "mask_predicted_rgb.tif"
        logger.info(f"[MASK-FILTERED] Buscando máscara en: {mask_path}")
        
        if not mask_path.exists():
            raise HTTPException(status_code=404, detail=f"Mask not found for scene {scene_id}")
        
        # Parsear clases solicitadas
        selected_classes = [c.strip().lower() for c in classes.split(',') if c.strip()]
        selected_class_ids = [CLASS_IDS[c] for c in selected_classes if c in CLASS_IDS]
        logger.info(f"[MASK-FILTERED] Clases seleccionadas: {selected_class_ids}")
        
        # Abrir GeoTIFF original
        with rasterio.open(str(mask_path)) as src:
            logger.info(f"[MASK-FILTERED] GeoTIFF abierto: {src.width}x{src.height}")
            
            # Leer bandas RGB
            data = src.read([1, 2, 3])
            
            # Convertir a (H, W, C)
            img_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
            
            # Crear imagen filtrada - mostrar solo píxeles de clases seleccionadas
            filtered_array = np.zeros_like(img_array)
            
            for class_id in selected_class_ids:
                color = CLASS_COLORS_RGB[class_id]
                # Encontrar píxeles con este color
                mask = (img_array[:, :, 0] == color[0]) & \
                       (img_array[:, :, 1] == color[1]) & \
                       (img_array[:, :, 2] == color[2])
                filtered_array[mask] = color
            
            # Convertir a PNG
            img = Image.fromarray(filtered_array.astype(np.uint8), 'RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.info(f"[MASK-FILTERED] PNG encoded: {len(img_base64)} chars")
            
            # Obtener bounds
            bbox = src.bounds
            crs_epsg = src.crs.to_epsg() if src.crs else None
            if crs_epsg is None:
                crs_epsg = 32717  # Default para Ecuador
            
            return {
                "width": src.width,
                "height": src.height,
                "crs": crs_epsg,
                "bounds": {
                    "minX": float(bbox.left),
                    "minY": float(bbox.bottom),
                    "maxX": float(bbox.right),
                    "maxY": float(bbox.top)
                },
                "image": f"data:image/png;base64,{img_base64}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MASK-FILTERED] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




