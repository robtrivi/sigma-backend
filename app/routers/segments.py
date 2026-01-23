from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import uuid
import logging
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

# Constants
PREDICTED_MASK_FILENAME = "mask_predicted_rgb.tif"

router = APIRouter(prefix="/segments", tags=["segments"])
segments_service = SegmentsService()
dl_segmentation_service = DLSegmentationService()
logger = logging.getLogger(__name__)


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
    region_id: str = Query(...),
    periodo: str = Query(...),
    class_id: list[str] | None = Query(default=None),
    bbox: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> SegmentFeatureCollection:
    try:
        return segments_service.segments_geojson(db, region_id, periodo, class_id, bbox)
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
logger = logging.getLogger(__name__)


def _normalize_coverage_list(coverage_data) -> list:
    """Convierte coverage_by_class en lista normalizada."""
    if isinstance(coverage_data, dict):
        return list(coverage_data.values()) if coverage_data else []
    return coverage_data if isinstance(coverage_data, list) else []


def _should_skip_item(class_id: int, class_name: str) -> bool:
    """Verifica si un item debe ser ignorado (ej: unlabeled)."""
    return class_id == 0 or class_name.lower() == 'unlabeled'


def _create_coverage_item(item: dict, pixel_area_m2: float) -> PixelCoverageItem | None:
    """Crea un PixelCoverageItem a partir de un diccionario, retorna None si debe ignorarse."""
    class_id = item.get('class_id', 0)
    class_name = item.get('class_name', f'class_{class_id}')
    
    if _should_skip_item(class_id, class_name):
        return None
    
    pixel_count = item.get('pixel_count', 0)
    coverage_percentage = item.get('coverage_percentage', 0.0)
    area_m2 = item.get('area_m2')
    
    if area_m2 is None:
        area_m2 = round(float(pixel_count * pixel_area_m2), 2)
    
    return PixelCoverageItem(
        class_id=int(class_id),
        class_name=str(class_name),
        pixel_count=int(pixel_count),
        coverage_percentage=float(coverage_percentage),
        area_m2=float(area_m2)
    )


def _process_coverage_items(result: SegmentationResult) -> list[PixelCoverageItem]:
    """Procesa la lista de items de cobertura."""
    coverage_items = []
    try:
        coverage_list = _normalize_coverage_list(result.coverage_by_class)
        
        for item in coverage_list:
            if not isinstance(item, dict):
                continue
            
            coverage_item = _create_coverage_item(item, result.pixel_area_m2)
            if coverage_item:
                coverage_items.append(coverage_item)
    except Exception as e:
        logger.error(f"Error procesando coverage_by_class: {type(e).__name__}: {str(e)}")
    
    return coverage_items


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
    
    coverage_items = _process_coverage_items(result)
    
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
    mask_path = Path(settings.data_dir) / "scenes" / scene_id / PREDICTED_MASK_FILENAME
    
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


def _infer_crs_from_bounds(bbox) -> int:
    """Infiere el código EPSG desde los bounds del GeoTIFF."""
    import logging
    logger = logging.getLogger(__name__)
    # Zona 17S: X entre 400000-800000, Y entre 9600000-10000000
    # Zona 18S: X entre 100000-500000, Y entre 9600000-10000000
    if 600000 < bbox.left < 700000 and 9700000 < bbox.bottom < 9800000:
        logger.info("[MASK-INFO] Inferido EPSG:32717 (UTM 17S)")
        return 32717
    elif 100000 < bbox.left < 400000 and 9600000 < bbox.bottom < 10000000:
        logger.info("[MASK-INFO] Inferido EPSG:32718 (UTM 18S)")
        return 32718
    else:
        logger.info("[MASK-INFO] Default EPSG:32717 (UTM 17S)")
        return 32717


def _get_crs_from_source(src) -> int | None:
    """Obtiene el CRS del GeoTIFF, infiriendo si es necesario."""
    import logging
    logger = logging.getLogger(__name__)
    crs_epsg = None
    
    try:
        if src.crs is not None:
            crs_epsg = src.crs.to_epsg()
            logger.info(f"[MASK-INFO] CRS obtenido del archivo: EPSG:{crs_epsg}")
    except Exception as crs_error:
        logger.warning(f"[MASK-INFO] Error al leer CRS: {crs_error}")
    
    if crs_epsg is None:
        logger.info("[MASK-INFO] CRS no detectado, intentando inferir desde bounds")
        crs_epsg = _infer_crs_from_bounds(src.bounds)
    
    return crs_epsg


def _read_mask_bands(src):
    """Lee las bandas RGB del GeoTIFF con manejo de excepciones."""
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)
    
    try:
        data = src.read([1, 2, 3])
        logger.info(f"[MASK-INFO] Datos leídos: shape={data.shape}")
    except Exception:
        # Si no hay 3 bandas, leer todas
        data = src.read()
        if data.shape[0] >= 3:
            data = data[:3]
        elif data.shape[0] == 1:
            # Replicar banda única a RGB
            data = np.repeat(data, 3, axis=0)
        logger.info(f"[MASK-INFO] Fallback: datos={data.shape}")
    
    return data


def _image_to_png_base64(img_array):
    """Convierte un array de imagen a PNG codificado en base64."""
    import base64
    import io
    import logging
    import numpy as np
    from PIL import Image
    
    logger = logging.getLogger(__name__)
    
    # Crear imagen RGBA con transparencia para píxeles blancos
    img_rgba = np.dstack([img_array, np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255])
    
    # Hacer transparentes los píxeles blancos (255, 255, 255)
    white_mask = np.all(img_array == [255, 255, 255], axis=2)
    img_rgba[white_mask, 3] = 0  # Canal alfa a 0 para píxeles blancos
    
    # Convertir a PNG con transparencia
    img = Image.fromarray(img_rgba, 'RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    logger.info(f"[MASK-INFO] PNG encoded: {len(img_base64)} chars")
    
    return img_base64


@router.get("/mask-info/{scene_id}")
def get_mask_info(scene_id: str, db: Session = Depends(get_db)):
    """
    Devuelve metadatos de georeferenciación de la máscara con imagen en base64
    """
    import rasterio
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    try:
        mask_path = Path(settings.data_dir) / "scenes" / scene_id / PREDICTED_MASK_FILENAME
        logger.info(f"[MASK-INFO] Buscando máscara en: {mask_path}")
        logger.info(f"[MASK-INFO] Path exists: {mask_path.exists()}")
        
        if not mask_path.exists():
            logger.warning("[MASK-INFO] Máscara no encontrada")
            raise HTTPException(status_code=404, detail=f"Mask not found for scene {scene_id}")
        
        # Abrir GeoTIFF
        with rasterio.open(str(mask_path)) as src:
            logger.info(f"[MASK-INFO] GeoTIFF abierto: {src.width}x{src.height}")
            
            # Obtener CRS
            crs_epsg = _get_crs_from_source(src)
            logger.info(f"[MASK-INFO] CRS final: EPSG:{crs_epsg}")
            
            # Leer bandas RGB
            data = _read_mask_bands(src)
            
            # Convertir a (H, W, C)
            img_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
            logger.info(f"[MASK-INFO] Array: {img_array.shape}, min={img_array.min()}, max={img_array.max()}")
            
            # Convertir a PNG
            img_base64 = _image_to_png_base64(img_array)
            
            bbox = src.bounds
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
        
        mask_path = Path(settings.data_dir) / "scenes" / scene_id / PREDICTED_MASK_FILENAME
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
            
            # Crear imagen filtrada con RGBA - mostrar solo píxeles de clases seleccionadas
            filtered_array_rgb = np.zeros_like(img_array)
            filtered_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
            
            for class_id in selected_class_ids:
                color = CLASS_COLORS_RGB[class_id]
                # Encontrar píxeles con este color
                mask = (img_array[:, :, 0] == color[0]) & \
                       (img_array[:, :, 1] == color[1]) & \
                       (img_array[:, :, 2] == color[2])
                filtered_array_rgb[mask] = color
                filtered_mask |= mask
            
            # Crear imagen RGBA con transparencia
            filtered_array_rgba = np.dstack([filtered_array_rgb, np.zeros((filtered_array_rgb.shape[0], filtered_array_rgb.shape[1]), dtype=np.uint8)])
            filtered_array_rgba[filtered_mask, 3] = 255  # Opaco para píxeles seleccionados
            
            # Convertir a PNG
            img = Image.fromarray(filtered_array_rgba.astype(np.uint8), 'RGBA')
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


def _get_color_mappings():
    """Retorna los mapeos de colores para clases."""
    class_colors_rgb = {
        0: (0, 0, 0), 1: (128, 64, 128), 2: (130, 76, 0), 3: (0, 102, 0), 4: (112, 103, 87),
        5: (28, 42, 168), 6: (48, 41, 30), 7: (0, 50, 89), 8: (107, 142, 35), 9: (70, 70, 70),
        10: (102, 102, 156), 11: (254, 228, 12), 12: (254, 148, 12), 13: (190, 153, 153), 14: (153, 153, 153),
        15: (255, 22, 96), 16: (102, 51, 0), 17: (9, 143, 150), 18: (119, 11, 32), 19: (51, 51, 0),
        20: (190, 250, 190), 21: (112, 150, 146), 22: (2, 135, 115), 23: (255, 0, 0),
    }
    
    class_name_to_id = {
        'unlabeled': 0, 'paved-area': 1, 'dirt': 2, 'grass': 3, 'gravel': 4,
        'water': 5, 'rocks': 6, 'pool': 7, 'vegetation': 8, 'roof': 9,
        'wall': 10, 'window': 11, 'door': 12, 'fence': 13, 'fence-pole': 14,
        'person': 15, 'dog': 16, 'car': 17, 'bicycle': 18, 'tree': 19,
        'bald-tree': 20, 'ar-marker': 21, 'obstacle': 22, 'conflicting': 23
    }
    
    return class_colors_rgb, class_name_to_id


def _apply_custom_colors(colors: str | None, class_colors_rgb: dict) -> None:
    """Aplica colores personalizados al diccionario de colores."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not colors:
        return
    
    _, class_name_to_id = _get_color_mappings()
    
    try:
        for pair in colors.split('|'):
            if ':' not in pair:
                continue
            class_name, hex_color = pair.split(':', 1)
            class_name, hex_color = class_name.strip(), hex_color.strip()
            
            if class_name in class_name_to_id and len(hex_color) == 6:
                class_id = class_name_to_id[class_name]
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                class_colors_rgb[class_id] = (r, g, b)
    except Exception as e:
        logger.warning(f"[MASKS-BY-PERIOD] Error al parsear colores: {str(e)}")


def _process_mask_with_filters(img_array, classes: str | None, colors: str | None, class_colors_rgb: dict, make_unlabeled_transparent: bool) -> tuple:
    """Procesa una máscara aplicando filtros de clases y colores."""
    import numpy as np
    
    unlabeled_mask = None
    if make_unlabeled_transparent:
        unlabeled_mask = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)
    
    if classes:
        selected_class_ids = _parse_class_ids(classes)
        if selected_class_ids:
            img_array = _filter_classes(img_array, selected_class_ids, class_colors_rgb)
        else:
            img_array = _apply_rgba_with_transparency(img_array)
    elif colors:
        img_array = _apply_color_mapping(img_array, class_colors_rgb)
    
    return img_array, unlabeled_mask


def _parse_class_ids(classes: str) -> list[int]:
    """Parsea IDs de clases desde string."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        return [int(c.strip()) for c in classes.split(',') if c.strip()]
    except ValueError:
        logger.warning(f"[MASKS-BY-PERIOD] Formato de clases inválido: {classes}")
        return []


def _filter_classes(img_array, selected_class_ids: list, class_colors_rgb: dict):
    """Filtra imagen para mostrar solo clases seleccionadas."""
    import numpy as np
    
    original_colors = _get_color_mappings()[0]
    filtered_array_rgb = np.zeros_like(img_array)
    filtered_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
    
    for class_id in selected_class_ids:
        if class_id not in original_colors:
            continue
        orig_color = original_colors[class_id]
        new_color = class_colors_rgb.get(class_id, orig_color)
        mask = (img_array[:, :, 0] == orig_color[0]) & (img_array[:, :, 1] == orig_color[1]) & (img_array[:, :, 2] == orig_color[2])
        filtered_array_rgb[mask] = new_color
        filtered_mask |= mask
    
    result = np.dstack([filtered_array_rgb, np.zeros((filtered_array_rgb.shape[0], filtered_array_rgb.shape[1]), dtype=np.uint8)])
    result[filtered_mask, 3] = 255
    return result


def _apply_rgba_with_transparency(img_array):
    """Convierte RGB a RGBA con transparencia para píxeles blancos."""
    import numpy as np
    
    img_rgba = np.dstack([img_array, np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255])
    white_mask = np.all(img_array == [255, 255, 255], axis=2)
    img_rgba[white_mask, 3] = 0
    return img_rgba


def _apply_color_mapping(img_array, class_colors_rgb: dict):
    """Aplica mapeo de colores a toda la imagen."""
    import numpy as np
    
    original_colors = _get_color_mappings()[0]
    new_img = np.zeros_like(img_array)
    
    for class_id, orig_color in original_colors.items():
        new_color = class_colors_rgb.get(class_id, orig_color)
        mask = (img_array[:, :, 0] == orig_color[0]) & (img_array[:, :, 1] == orig_color[1]) & (img_array[:, :, 2] == orig_color[2])
        new_img[mask] = new_color
    
    return new_img


def _mask_to_rgba(img_array, unlabeled_mask, make_unlabeled_transparent: bool):
    """Convierte imagen a RGBA con transparencia según configuración."""
    import numpy as np
    from PIL import Image
    
    if img_array.shape[2] == 3:
        img_rgba = np.dstack([img_array, np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255])
    else:
        img_rgba = img_array
    
    if make_unlabeled_transparent and unlabeled_mask is not None:
        img_rgba[unlabeled_mask, 3] = 0
    
    return img_rgba


def _mask_to_png_base64(img_rgba):
    """Convierte máscara RGBA a PNG en base64."""
    import base64
    import io
    from PIL import Image
    
    img = Image.fromarray(img_rgba, 'RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _process_scene_mask(scene, scene_id: str, classes: str | None, colors: str | None, make_unlabeled_transparent: bool):
    """Procesa una máscara de escena y retorna datos o None si hay error."""
    import rasterio
    import numpy as np
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    mask_path = Path(settings.data_dir) / "scenes" / str(scene_id) / PREDICTED_MASK_FILENAME
    
    if not mask_path.exists():
        logger.warning(f"[MASKS-BY-PERIOD] Máscara no encontrada para escena {scene_id}")
        return None
    
    with rasterio.open(str(mask_path)) as src:
        bbox = src.bounds
        crs_epsg = _get_crs_from_source(src)
        
        data = _read_mask_bands(src)
        img_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        
        class_colors_rgb, _ = _get_color_mappings()
        _apply_custom_colors(colors, class_colors_rgb)
        
        img_array, unlabeled_mask = _process_mask_with_filters(img_array, classes, colors, class_colors_rgb, make_unlabeled_transparent)
        img_rgba = _mask_to_rgba(img_array, unlabeled_mask, make_unlabeled_transparent)
        img_base64 = _mask_to_png_base64(img_rgba)
        
        return {
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
        }


@router.get("/masks-by-period/{region_id}")
def get_masks_by_period(region_id: str, periodo: str = Query(...), classes: str = Query(None), colors: str = Query(None), make_unlabeled_transparent: bool = Query(False), db: Session = Depends(get_db)):
    """
    Devuelve todas las máscaras RGB de las escenas de un período específico.
    """
    import logging
    from datetime import date
    from sqlalchemy import select, and_
    from app.models import Scene
    
    logger = logging.getLogger(__name__)
    
    try:
        # Parsear período
        parts = periodo.split('-')
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Period must be in YYYY-MM format")
        
        year, month = int(parts[0]), int(parts[1])
        start_date = date(year, month, 1)
        end_date = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        
        # Obtener escenas del período
        stmt = select(Scene).where(and_(Scene.region_id == region_id, Scene.capture_date >= start_date, Scene.capture_date < end_date))
        scenes = db.execute(stmt).scalars().all()
        
        if not scenes:
            logger.warning(f"[MASKS-BY-PERIOD] No scenes found for region {region_id}, period {periodo}")
            raise HTTPException(status_code=404, detail=f"No scenes found for period {periodo}")
        
        masks = []
        for scene in scenes:
            try:
                mask_data = _process_scene_mask(scene, scene.id, classes, colors, make_unlabeled_transparent)
                if mask_data:
                    masks.append(mask_data)
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


def _parse_period_to_dates(periodo: str) -> tuple:
    """Parsea período YYYY-MM a fechas de inicio y fin del mes."""
    from datetime import date
    
    parts = periodo.split('-')
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Period must be in YYYY-MM format")
    
    year, month = int(parts[0]), int(parts[1])
    start_date = date(year, month, 1)
    end_date = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start_date, end_date


def _aggregate_coverage_data(coverage_items, pixel_aggregator: dict, area_aggregator: dict) -> tuple:
    """Agrega datos de cobertura desde lista o diccionario."""
    total_pixels = 0
    total_area_m2 = 0
    
    if isinstance(coverage_items, list):
        total_pixels, total_area_m2 = _aggregate_from_list(coverage_items, pixel_aggregator, area_aggregator)
    elif isinstance(coverage_items, dict):
        total_pixels, total_area_m2 = _aggregate_from_dict(coverage_items, pixel_aggregator, area_aggregator)
    
    return total_pixels, total_area_m2


def _aggregate_from_list(coverage_items: list, pixel_aggregator: dict, area_aggregator: dict) -> tuple:
    """Agrega datos de cobertura desde lista."""
    total_pixels = 0
    total_area_m2 = 0
    
    for class_data in coverage_items:
        if isinstance(class_data, dict) and 'class_name' in class_data:
            class_name = class_data['class_name']
            pixel_count = class_data.get('pixel_count', 0)
            area_m2 = class_data.get('area_m2', 0)
            
            if class_name not in pixel_aggregator:
                pixel_aggregator[class_name] = 0
                area_aggregator[class_name] = 0
            
            pixel_aggregator[class_name] += pixel_count
            area_aggregator[class_name] += area_m2
            total_pixels += pixel_count
            total_area_m2 += area_m2
    
    return total_pixels, total_area_m2


def _aggregate_from_dict(coverage_items: dict, pixel_aggregator: dict, area_aggregator: dict) -> tuple:
    """Agrega datos de cobertura desde diccionario (compatibilidad hacia atrás)."""
    total_pixels = 0
    total_area_m2 = 0
    
    for class_name, class_data in coverage_items.items():
        if isinstance(class_data, dict):
            pixel_count = class_data.get('pixel_count', 0)
            area_m2 = class_data.get('area_m2', 0)
            
            if class_name not in pixel_aggregator:
                pixel_aggregator[class_name] = 0
                area_aggregator[class_name] = 0
            
            pixel_aggregator[class_name] += pixel_count
            area_aggregator[class_name] += area_m2
            total_pixels += pixel_count
            total_area_m2 += area_m2
    
    return total_pixels, total_area_m2


def _calculate_total_area_without_unlabeled(pixel_aggregator: dict, area_aggregator: dict) -> tuple:
    """Calcula total de píxeles y áreas excluyendo unlabeled."""
    total_pixels = 0
    total_area_m2 = 0
    
    for class_name, pixel_count in pixel_aggregator.items():
        if class_name.lower() != 'unlabeled' and class_name != 'Sin etiqueta':
            total_pixels += pixel_count
            total_area_m2 += area_aggregator.get(class_name, 0)
    
    return total_pixels, total_area_m2


def _build_coverage_item(class_name: str, pixel_count: int, area_m2: float, total_area_m2: float) -> dict:
    """Construye un item de cobertura con porcentaje calculado."""
    coverage_percentage = (area_m2 / total_area_m2 * 100) if total_area_m2 > 0 else 0
    
    return {
        "class_name": class_name,
        "pixel_count": pixel_count,
        "coverage_percentage": round(coverage_percentage, 2),
        "area_m2": round(float(area_m2), 2)
    }


@router.get("/pixel-coverage-aggregated/{region_id}")
def get_aggregated_pixel_coverage(region_id: str, periodo: str = Query(...), db: Session = Depends(get_db)):
    """
    Devuelve la cobertura de píxeles agregada de todas las máscaras de un período.
    """
    import logging
    from app.models import Scene, SegmentationResult, ClassCatalog
    
    logger = logging.getLogger(__name__)
    
    try:
        # Parsear período
        start_date, end_date = _parse_period_to_dates(periodo)
        
        # Validar y obtener escenas
        scenes = _validate_and_get_scenes(region_id, start_date, end_date, db, logger, periodo)
        
        # Agregadores
        pixel_aggregator = {}
        area_aggregator = {}
        pixel_area_m2 = 1.0
        
        # Agregar datos de escenas
        total_pixels, _, pixel_area_m2 = _aggregate_scenes_coverage(
            scenes, pixel_aggregator, area_aggregator, pixel_area_m2
        )
        
        # Validar datos agregados
        if total_pixels == 0:
            logger.warning(f"[PIXEL-COVERAGE-AGG] No pixel data found for period {periodo}")
            raise HTTPException(status_code=404, detail="No pixel coverage data found")
        
        # Calcular totales sin unlabeled
        total_pixels_without_unlabeled, total_area_m2_calculated = _calculate_total_area_without_unlabeled(
            pixel_aggregator, area_aggregator
        )
        
        # Construir datos de cobertura
        coverage_data = _build_coverage_response(pixel_aggregator, area_aggregator, total_area_m2_calculated)
        
        logger.info(f"[PIXEL-COVERAGE-AGG] Aggregated {len(coverage_data)} classes, total {total_pixels_without_unlabeled} pixels, {total_area_m2_calculated} m²")
        
        return {
            "regionId": region_id,
            "periodo": periodo,
            "totalPixels": total_pixels_without_unlabeled,
            "totalAreaM2": total_area_m2_calculated,
            "pixelAreaM2": pixel_area_m2,
            "coverageByClass": coverage_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PIXEL-COVERAGE-AGG] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _validate_and_get_scenes(region_id: str, start_date, end_date, db: Session, logger, periodo: str):
    """Valida y obtiene escenas del período especificado."""
    from sqlalchemy import select, and_
    from sqlalchemy.orm import joinedload
    from app.models import Scene
    
    stmt = select(Scene).where(
        and_(
            Scene.region_id == region_id,
            Scene.capture_date >= start_date,
            Scene.capture_date < end_date
        )
    ).options(joinedload(Scene.segmentation_results))
    
    scenes = db.execute(stmt).unique().scalars().all()
    
    if not scenes:
        logger.warning(f"[PIXEL-COVERAGE-AGG] No scenes found for region {region_id}, period {periodo}")
        raise HTTPException(status_code=404, detail=f"No scenes found for period {periodo}")
    
    return scenes


def _aggregate_scenes_coverage(scenes, pixel_aggregator: dict, area_aggregator: dict, pixel_area_m2: float):
    """Agrega cobertura de píxeles de todas las escenas."""
    total_pixels = 0
    total_area_m2 = 0
    EPSILON = 1e-9
    
    for scene in scenes:
        for result in scene.segmentation_results:
            if result and result.coverage_by_class:
                if abs(pixel_area_m2 - 1.0) < EPSILON and result.pixel_area_m2:
                    pixel_area_m2 = result.pixel_area_m2
                
                scene_pixels, scene_area = _aggregate_coverage_data(
                    result.coverage_by_class, pixel_aggregator, area_aggregator
                )
                total_pixels += scene_pixels
                total_area_m2 += scene_area
    
    return total_pixels, total_area_m2, pixel_area_m2


def _build_coverage_response(pixel_aggregator: dict, area_aggregator: dict, total_area_m2_calculated: float) -> list:
    """Construye lista de cobertura filtrando unlabeled."""
    coverage_data = []
    for class_name in sorted(pixel_aggregator.keys()):
        if class_name.lower() == 'unlabeled' or class_name == 'Sin etiqueta':
            continue
        
        pixel_count = pixel_aggregator[class_name]
        area_m2 = area_aggregator.get(class_name, 0)
        
        coverage_data.append(_build_coverage_item(class_name, pixel_count, area_m2, total_area_m2_calculated))
    
    return coverage_data




