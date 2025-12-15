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
    PixelCoverageItem,
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
    
    # Convertir coverage_by_class a modelos de Pydantic para validar y rellenar campos faltantes
    coverage_items = []
    try:
        # Manejar ambos casos: lista de diccionarios o diccionario
        coverage_list = result.coverage_by_class
        if isinstance(coverage_list, dict):
            # Si es diccionario, convertir a lista
            coverage_list = list(coverage_list.values()) if coverage_list else []
        
        if isinstance(coverage_list, list):
            for item in coverage_list:
                if isinstance(item, dict):
                    # Rellenar campos que podrían faltar
                    class_id = item.get('class_id', 0)
                    class_name = item.get('class_name', f'class_{class_id}')
                    pixel_count = item.get('pixel_count', 0)
                    coverage_percentage = item.get('coverage_percentage', 0.0)
                    
                    # Calcular area_m2 si falta
                    area_m2 = item.get('area_m2')
                    if area_m2 is None:
                        area_m2 = round(float(pixel_count * result.pixel_area_m2), 2)
                    
                    # Crear el modelo Pydantic con todos los campos
                    coverage_item = PixelCoverageItem(
                        class_id=int(class_id),
                        class_name=str(class_name),
                        pixel_count=int(pixel_count),
                        coverage_percentage=float(coverage_percentage),
                        area_m2=float(area_m2)
                    )
                    coverage_items.append(coverage_item)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error procesando coverage_by_class para scene {scene_id}: {type(e).__name__}: {str(e)}")
        # Devolver lista vacía si hay error en procesamiento
        coverage_items = []
    
    return SegmentationCoverageRead(
        scene_id=str(result.scene_id),
        total_pixels=result.total_pixels,
        total_area_m2=result.total_area_m2,
        pixel_area_m2=result.pixel_area_m2,
        image_resolution=result.image_resolution,
        coverage_by_class=coverage_items,
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


@router.get("/masks-by-period/{region_id}")
def get_masks_by_period(region_id: str, periodo: str = Query(...), classes: str = Query(None), db: Session = Depends(get_db)):
    """
    Devuelve todas las máscaras RGB de las escenas de un período específico.
    
    Args:
        region_id: ID de la región
        periodo: Período en formato YYYY-MM (ej: "2025-12")
        classes: Clases a mostrar en la máscara (separadas por comas, ej: "1,2,3")
        
    Returns:
        Lista de máscaras con metadatos de georeferenciación
    """
    import rasterio
    import base64
    import numpy as np
    import io
    import logging
    from datetime import date
    from PIL import Image
    from sqlalchemy import select, and_
    from app.models import Scene
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    try:
        # Obtener todas las escenas de la región cuya capture_date está en el período
        parts = periodo.split('-')
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Period must be in YYYY-MM format")
        
        year = int(parts[0])
        month = int(parts[1])
        
        # Calcular fecha de inicio y fin del mes
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
        
        stmt = select(Scene).where(
            and_(
                Scene.region_id == region_id,
                Scene.capture_date >= start_date,
                Scene.capture_date < end_date
            )
        )
        
        scenes = db.execute(stmt).scalars().all()
        
        if not scenes:
            logger.warning(f"[MASKS-BY-PERIOD] No scenes found for region {region_id}, period {periodo}")
            raise HTTPException(status_code=404, detail=f"No scenes found for period {periodo}")
        
        masks = []
        
        for scene in scenes:
            try:
                mask_path = Path(settings.data_dir) / "scenes" / str(scene.id) / "mask_predicted_rgb.tif"
                
                if not mask_path.exists():
                    logger.warning(f"[MASKS-BY-PERIOD] Máscara no encontrada para escena {scene.id}")
                    continue
                
                # Abrir GeoTIFF
                with rasterio.open(str(mask_path)) as src:
                    logger.info(f"[MASKS-BY-PERIOD] GeoTIFF abierto para escena {scene.id}: {src.width}x{src.height}")
                    
                    # Obtener georeferenciación
                    bbox = src.bounds
                    crs_epsg = None
                    
                    try:
                        if src.crs is not None:
                            crs_epsg = src.crs.to_epsg()
                    except Exception as crs_error:
                        logger.warning(f"[MASKS-BY-PERIOD] Error al leer CRS: {crs_error}")
                    
                    # Si no hay CRS pero los bounds parecen ser UTM, inferir el código EPSG
                    if crs_epsg is None:
                        if 600000 < bbox.left < 700000 and 9700000 < bbox.bottom < 9800000:
                            crs_epsg = 32717  # UTM Zona 17S
                        elif 100000 < bbox.left < 400000 and 9600000 < bbox.bottom < 10000000:
                            crs_epsg = 32718  # UTM Zona 18S
                        else:
                            crs_epsg = 32717  # Default para Ecuador
                    
                    # Leer bandas RGB
                    try:
                        data = src.read([1, 2, 3])
                    except:
                        data = src.read()
                        if data.shape[0] >= 3:
                            data = data[:3]
                        elif data.shape[0] == 1:
                            data = np.repeat(data, 3, axis=0)
                    
                    # Convertir a (H, W, C)
                    img_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
                    
                    # Aplicar filtro de clases si se especifica
                    if classes:
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
                            0: 'unlabeled', 1: 'paved-area', 2: 'dirt', 3: 'grass', 4: 'gravel',
                            5: 'water', 6: 'rocks', 7: 'pool', 8: 'vegetation', 9: 'roof',
                            10: 'wall', 11: 'window', 12: 'door', 13: 'fence', 14: 'fence-pole',
                            15: 'person', 16: 'dog', 17: 'car', 18: 'bicycle', 19: 'tree',
                            20: 'bald-tree', 21: 'ar-marker', 22: 'obstacle', 23: 'conflicting'
                        }
                        
                        # Parsear clases solicitadas (como números separados por comas)
                        selected_class_ids = []
                        try:
                            selected_class_ids = [int(c.strip()) for c in classes.split(',') if c.strip()]
                        except ValueError:
                            logger.warning(f"[MASKS-BY-PERIOD] Formato de clases inválido: {classes}")
                        
                        if selected_class_ids:
                            # Crear imagen filtrada - mostrar solo píxeles de clases seleccionadas
                            filtered_array = np.zeros_like(img_array)
                            
                            for class_id in selected_class_ids:
                                if class_id in CLASS_COLORS_RGB:
                                    color = CLASS_COLORS_RGB[class_id]
                                    # Encontrar píxeles con este color
                                    mask = (img_array[:, :, 0] == color[0]) & \
                                           (img_array[:, :, 1] == color[1]) & \
                                           (img_array[:, :, 2] == color[2])
                                    filtered_array[mask] = color
                            
                            img_array = filtered_array
                    
                    # Convertir a PNG
                    img = Image.fromarray(img_array, 'RGB')
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    masks.append({
                        "sceneId": str(scene.id),
                        "captureDate": str(scene.capture_date),
                        "sensor": scene.sensor,
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
                    })
                    
            except Exception as e:
                logger.warning(f"[MASKS-BY-PERIOD] Error procesando escena {scene.id}: {str(e)}")
                continue
        
        if not masks:
            logger.warning(f"[MASKS-BY-PERIOD] No valid masks found for region {region_id}, period {periodo}")
            raise HTTPException(status_code=404, detail="No valid masks found for period")
        
        return {
            "regionId": region_id,
            "periodo": periodo,
            "count": len(masks),
            "masks": masks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MASKS-BY-PERIOD] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pixel-coverage-aggregated/{region_id}")
def get_aggregated_pixel_coverage(region_id: str, periodo: str = Query(...), db: Session = Depends(get_db)):
    """
    Devuelve la cobertura de píxeles agregada (sumada) de todas las máscaras de un período.
    
    Args:
        region_id: ID de la región
        periodo: Período en formato YYYY-MM (ej: "2025-12")
        
    Returns:
        Cobertura agregada con píxeles y porcentajes sumados
    """
    import logging
    from datetime import date
    from sqlalchemy import select, and_
    from app.models import Scene, SegmentationResult, ClassCatalog
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    try:
        # Obtener todas las escenas del período
        parts = periodo.split('-')
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Period must be in YYYY-MM format")
        
        year = int(parts[0])
        month = int(parts[1])
        
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
        
        stmt = select(Scene).where(
            and_(
                Scene.region_id == region_id,
                Scene.capture_date >= start_date,
                Scene.capture_date < end_date
            )
        )
        
        scenes = db.execute(stmt).scalars().all()
        
        if not scenes:
            logger.warning(f"[PIXEL-COVERAGE-AGG] No scenes found for region {region_id}, period {periodo}")
            raise HTTPException(status_code=404, detail=f"No scenes found for period {periodo}")
        
        # Agregador de píxeles por clase
        pixel_aggregator = {}
        total_pixels = 0
        
        for scene in scenes:
            # Obtener resultado de segmentación para la escena
            result = db.query(SegmentationResult).filter(
                SegmentationResult.scene_id == scene.id
            ).first()
            
            if result and result.coverage_by_class:
                # result.coverage_by_class es una LISTA de diccionarios:
                # [{"class_name": "Vegetación", "pixel_count": X, ...}, ...]
                
                # Si es lista, iterar sobre elementos
                coverage_items = result.coverage_by_class
                if isinstance(coverage_items, list):
                    for class_data in coverage_items:
                        if isinstance(class_data, dict) and 'class_name' in class_data and 'pixel_count' in class_data:
                            class_name = class_data['class_name']
                            pixel_count = class_data['pixel_count']
                            
                            if class_name not in pixel_aggregator:
                                pixel_aggregator[class_name] = 0
                            
                            pixel_aggregator[class_name] += pixel_count
                            total_pixels += pixel_count
                # Si es diccionario, iterar sobre items (compatibilidad hacia atrás)
                elif isinstance(coverage_items, dict):
                    for class_name, class_data in coverage_items.items():
                        if isinstance(class_data, dict) and 'pixel_count' in class_data:
                            pixel_count = class_data['pixel_count']
                            
                            if class_name not in pixel_aggregator:
                                pixel_aggregator[class_name] = 0
                            
                            pixel_aggregator[class_name] += pixel_count
                            total_pixels += pixel_count
        
        if total_pixels == 0:
            logger.warning(f"[PIXEL-COVERAGE-AGG] No pixel data found for period {periodo}")
            raise HTTPException(status_code=404, detail="No pixel coverage data found")
        
        # Obtener pixel_area_m2 de cualquier escena disponible
        pixel_area_m2 = 1.0
        for scene in scenes:
            result = db.query(SegmentationResult).filter(
                SegmentationResult.scene_id == scene.id
            ).first()
            if result and result.pixel_area_m2:
                pixel_area_m2 = result.pixel_area_m2
                break
        
        # Calcular porcentajes y áreas
        coverage_data = []
        total_area_m2 = 0
        for class_name, pixel_count in sorted(pixel_aggregator.items()):
            coverage_percentage = (pixel_count / total_pixels) * 100
            area_m2 = round(float(pixel_count * pixel_area_m2), 2)
            total_area_m2 += area_m2
            
            coverage_data.append({
                "class_name": class_name,
                "pixel_count": pixel_count,
                "coverage_percentage": coverage_percentage,
                "area_m2": area_m2
            })
        
        logger.info(f"[PIXEL-COVERAGE-AGG] Aggregated {len(coverage_data)} classes, total {total_pixels} pixels, {total_area_m2} m²")
        
        return {
            "regionId": region_id,
            "periodo": periodo,
            "totalPixels": total_pixels,
            "totalAreaM2": total_area_m2,
            "pixelAreaM2": pixel_area_m2,
            "coverageByClass": coverage_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PIXEL-COVERAGE-AGG] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




