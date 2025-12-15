from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import MultiPolygon, Polygon
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import ClassCatalog, Scene, Segment, SegmentationResult, SegmentationSummary
from app.schemas.schemas import SegmentsImportResponse, PixelCoverageItem
from app.services.progress_service import get_progress_service
from app.utils.geo import (
    geometry_area_m2,
    reproject_geometry,
    to_wkb,
)
from app.utils.tiff_validator import TiffValidator

logger = logging.getLogger(__name__)

_model = None

CLASS_MAPPING = {
    0: "unlabeled",
    1: "paved-area",
    2: "dirt",
    3: "grass",
    4: "gravel",
    5: "water",
    6: "rocks",
    7: "pool",
    8: "vegetation",
    9: "roof",
    10: "wall",
    11: "window",
    12: "door",
    13: "fence",
    14: "fence-pole",
    15: "person",
    16: "dog",
    17: "car",
    18: "bicycle",
    19: "tree",
    20: "bald-tree",
    21: "ar-marker",
    22: "obstacle",
    23: "conflicting",
}

CLASS_COLORS = {
    0: "#000000",   # unlabeled
    1: "#804080",   # paved-area
    2: "#824C00",   # dirt
    3: "#006600",   # grass
    4: "#706757",   # gravel
    5: "#1C2AA8",   # water
    6: "#30291E",   # rocks
    7: "#003259",   # pool
    8: "#6B8E23",   # vegetation
    9: "#464646",   # roof
    10: "#66669C",  # wall
    11: "#FEE40C",  # window
    12: "#FE940C",  # door
    13: "#BE9999",  # fence
    14: "#999999",  # fence-pole
    15: "#FF1660",  # person
    16: "#663300",  # dog
    17: "#098F96",  # car
    18: "#770B20",  # bicycle
    19: "#333300",  # tree
    20: "#BEFABE",  # bald-tree
    21: "#709692",  # ar-marker
    22: "#028773",  # obstacle
    23: "#FF0000",  # conflicting
}

CLASS_NAMES = {
    0: "Sin etiqueta",
    1: "Área pavimentada",
    2: "Tierra",
    3: "Césped",
    4: "Grava",
    5: "Agua",
    6: "Rocas",
    7: "Piscina",
    8: "Vegetación",
    9: "Techo",
    10: "Pared",
    11: "Ventana",
    12: "Puerta",
    13: "Cerca",
    14: "Poste de cerca",
    15: "Persona",
    16: "Perro",
    17: "Automóvil",
    18: "Bicicleta",
    19: "Árbol",
    20: "Árbol sin hojas",
    21: "Marcador AR",
    22: "Obstáculo",
    23: "Conflicto",
}


def get_model(settings: Settings):
    global _model
    if _model is None:
        try:
            import tensorflow as tf
            model_path = settings.segmentation_model_path
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            _model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info(f"Segmentation model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise
    return _model


def get_model_input_shape(model) -> Tuple[int, int]:
    input_shape = model.input_shape
    return (input_shape[1], input_shape[2])


def get_model_num_classes(model) -> int:
    output_shape = model.output_shape
    return output_shape[-1]


class DLSegmentationService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def segment_scene(
        self, db: Session, scene_id: str, tiff_bytes: bytes
    ) -> SegmentsImportResponse:
        progress_service = get_progress_service()
        progress_service.initialize_progress(scene_id)
        
        scene = db.get(Scene, uuid.UUID(scene_id))
        if not scene:
            error_msg = "Scene not found"
            progress_service.error_progress(scene_id, error_msg)
            raise ValueError(error_msg)

        # Step 0: Validate TIFF before processing
        try:
            progress_service.update_step(
                scene_id, 0, "in-progress",
                message="Validando archivo TIFF"
            )
            tiff_metadata = TiffValidator.validate(tiff_bytes)
            logger.info(
                f"Scene {scene_id} TIFF validated: "
                f"{tiff_metadata.width}x{tiff_metadata.height}, "
                f"EPSG:{tiff_metadata.epsg}"
            )
            progress_service.update_step(
                scene_id, 0, "completed",
                message=f"TIFF validado ({tiff_metadata.width}x{tiff_metadata.height})"
            )
        except Exception as e:
            error_msg = f"Validación del TIFF falló: {str(e)}"
            logger.error(f"TIFF validation failed for scene {scene_id}: {e}")
            progress_service.update_step(scene_id, 0, "error", error=error_msg)
            progress_service.error_progress(scene_id, error_msg)
            raise ValueError(error_msg) from e

        try:
            # Step 1: Read and load geospatial data
            progress_service.update_step(
                scene_id, 1, "in-progress",
                message="Leyendo datos geoespaciales"
            )
            model = get_model(self.settings)
            model_height, model_width = get_model_input_shape(model)
            num_classes = get_model_num_classes(model)

            image_data, transform, crs, original_shape = self._read_tiff(tiff_bytes)
            progress_service.update_step(
                scene_id, 1, "completed",
                message=f"Datos cargados ({original_shape[1]}x{original_shape[0]} px)"
            )
            
            # Step 2: Process with segmentation model
            progress_service.update_step(
                scene_id, 2, "in-progress",
                message="Procesando con modelo de segmentación"
            )
            preprocessed = self._preprocess_image(image_data, model_height, model_width)
            predictions = model.predict(preprocessed, verbose=0)
            mask_ids_model = np.argmax(predictions[0], axis=-1)
            progress_service.update_step(
                scene_id, 2, "completed",
                message="Predicción completada"
            )
            
            # Step 3: Generate predicted mask
            progress_service.update_step(
                scene_id, 3, "in-progress",
                message="Generando máscara predicha"
            )
            mask_ids_original = self._resize_mask(
                mask_ids_model, original_shape[0], original_shape[1]
            )
            self._save_mask_rgb(db, scene, mask_ids_original, image_data, transform, crs)
            logger.info(f"RGB mask saved for scene {scene_id}")
            progress_service.update_step(
                scene_id, 3, "completed",
                message="Máscara guardada"
            )
            
            # Step 4: Calculate pixels per class
            progress_service.update_step(
                scene_id, 4, "in-progress",
                message="Calculando píxeles por clase"
            )
            coverage_data = self._calculate_pixel_coverage(mask_ids_model, num_classes)
            total_pixels = mask_ids_model.size
            self._save_coverage_to_db(db, scene, coverage_data, total_pixels)
            logger.info(f"Pixel coverage analysis saved for scene {scene_id}")
            progress_service.update_step(
                scene_id, 4, "completed",
                message="Análisis de cobertura completado"
            )
            
            # Step 5: Create segments
            progress_service.update_step(
                scene_id, 5, "in-progress",
                message="Creando segmentos en base de datos"
            )
            stats = self._calculate_stats(mask_ids_original, num_classes)
            features_geo = self._vectorize_mask(mask_ids_original, transform, crs)
            segment_ids = self._persist_segments(
                db, scene, features_geo, stats, scene.epsg
            )
            db.commit()

            logger.info(
                f"Segmentation completed for scene {scene_id}. Created {len(segment_ids)} segments."
            )
            progress_service.update_step(
                scene_id, 5, "completed",
                message=f"Se crearon {len(segment_ids)} segmentos"
            )
            
            # Mark overall progress as completed
            progress_service.complete_progress(
                scene_id,
                result={
                    "sceneId": str(scene_id),
                    "segmentCount": len(segment_ids),
                    "totalPixels": int(total_pixels),
                }
            )
            
        except Exception as seg_error:
            error_msg = f"Error durante segmentación: {str(seg_error)}"
            logger.error(f"Error during segmentation of scene {scene_id}: {seg_error}", exc_info=True)
            progress_service.error_progress(scene_id, error_msg)
            raise
        return SegmentsImportResponse(inserted=len(segment_ids), segmentIds=segment_ids)

    def _read_tiff(
        self, tiff_bytes: bytes
    ) -> Tuple[np.ndarray, Affine, int | None, Tuple[int, int]]:
        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as src:
                data = src.read()
                transform = src.transform
                crs_epsg = None
                
                try:
                    if src.crs is not None:
                        crs_epsg = src.crs.to_epsg()
                except Exception as e:
                    logger.warning(f"[_read_tiff] Error reading CRS: {e}. Will attempt to infer from bounds.")
                
                # Si no hay CRS pero tenemos bounds, inferir del rango de coordenadas
                if crs_epsg is None:
                    try:
                        bbox = src.bounds
                        # Ecuador generalmente está en UTM Zonas 17-18 Sur
                        # Zona 17S: X 400000-800000, Y 9600000-10000000
                        # Zona 18S: X 100000-500000, Y 9600000-10000000
                        if bbox.left > 400000 and bbox.left < 800000:
                            crs_epsg = 32717  # UTM Zone 17S
                            logger.info(f"[_read_tiff] Inferido CRS: EPSG:32717 (UTM 17S)")
                        elif bbox.left > 100000 and bbox.left < 500000:
                            crs_epsg = 32718  # UTM Zone 18S
                            logger.info(f"[_read_tiff] Inferido CRS: EPSG:32718 (UTM 18S)")
                        else:
                            crs_epsg = 32717  # Default para Ecuador
                            logger.info(f"[_read_tiff] CRS no detectado, usando default EPSG:32717")
                    except Exception as e:
                        logger.warning(f"[_read_tiff] Error inferring CRS from bounds: {e}. Using default EPSG:32717")
                        crs_epsg = 32717
                
                original_shape = (src.height, src.width)

                if data.shape[0] >= 3:
                    image_rgb = data[:3].transpose(1, 2, 0)
                elif data.shape[0] == 1:
                    single_band = data[0]
                    image_rgb = np.stack([single_band, single_band, single_band], axis=-1)
                else:
                    raise ValueError(
                        f"Unsupported number of bands: {data.shape[0]}. Expected at least 3 or 1."
                    )

        logger.info(f"[_read_tiff] Imagen: {original_shape}, CRS: EPSG:{crs_epsg}")
        return image_rgb, transform, crs_epsg, original_shape

    def _preprocess_image(
        self, image: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        import tensorflow as tf
        
        # Convertir a tensor de TensorFlow para redimensionar (idéntico a notebook)
        image_tf = tf.convert_to_tensor(image.astype(np.float32))
        
        # Redimensionar con tf.image.resize (MISMO método que notebook)
        resized = tf.image.resize(image_tf, [target_height, target_width])
        
        # Normalizar con MobileNetV2 preprocessing (idéntico a notebook)
        normalized = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        
        # Agregar dimensión de batch
        batched = tf.expand_dims(normalized, axis=0)
        
        # Convertir a numpy para el modelo
        return batched.numpy()

    def _resize_mask(
        self, mask: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        resized = cv2.resize(
            mask.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized

    def _calculate_stats(
        self, mask: np.ndarray, num_classes: int
    ) -> dict[int, dict[str, float]]:
        total_pixels = mask.size
        stats = {}
        for class_id in range(num_classes):
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0.0
            stats[class_id] = {"count": int(count), "percentage": percentage}
        return stats

    def _vectorize_mask(
        self, mask: np.ndarray, transform: Affine, crs_epsg: int | None
    ) -> List[Tuple[int, MultiPolygon, float]]:
        features_list = []
        num_classes = len(np.unique(mask))

        for class_id in range(num_classes):
            class_mask = (mask == class_id).astype(np.uint8)
            if np.sum(class_mask) == 0:
                continue

            num_labels, labels = cv2.connectedComponents(class_mask)
            for component_id in range(1, num_labels):
                component_mask = (labels == component_id).astype(np.uint8)
                contours, _ = cv2.findContours(
                    component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    pixel_coords = contour.squeeze()
                    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
                        continue

                    geo_coords = []
                    for col, row in pixel_coords:
                        x, y = transform * (col, row)
                        geo_coords.append((x, y))

                    if len(geo_coords) < 3:
                        continue

                    try:
                        poly = Polygon(geo_coords)
                        if not poly.is_valid or poly.is_empty:
                            continue

                        multi_poly = (
                            MultiPolygon([poly])
                            if isinstance(poly, Polygon)
                            else poly
                        )
                        area_pixels = np.sum(component_mask)
                        features_list.append((class_id, multi_poly, area_pixels))
                    except Exception as e:
                        logger.warning(f"Failed to create polygon for contour: {e}")
                        continue

        return features_list

    def _persist_segments(
        self,
        db: Session,
        scene: Scene,
        features_geo: List[Tuple[int, MultiPolygon, float]],
        stats: dict,
        scene_epsg: int,
    ) -> List[str]:
        segment_ids = []
        segments_to_add = []
        
        # Cache ClassCatalogs para evitar queries repetidas
        catalog_cache = {}
        
        # Pre-crear todos los ClassCatalogs necesarios
        class_ids_needed = set(class_id for class_id, _, _ in features_geo)
        
        for class_id in class_ids_needed:
            class_label = CLASS_MAPPING.get(class_id, f"unknown_{class_id}")
            catalog_id = class_label
            
            # Intentar obtener del caché primero
            if catalog_id not in catalog_cache:
                catalog = db.get(ClassCatalog, catalog_id)
                if not catalog:
                    # Crear nuevo si no existe
                    color = CLASS_COLORS.get(class_id, "#9E9E9E")
                    class_name = CLASS_NAMES.get(class_id, class_label.replace("-", " ").replace("_", " ").title())
                    catalog = ClassCatalog(
                        id=catalog_id,
                        name=class_name,
                        color_hex=color,
                        icono_primeng="pi pi-map-marker",
                        description=f"Clase de segmentación: {class_name}",
                    )
                    db.add(catalog)
                
                catalog_cache[catalog_id] = catalog
        
        # Flush una sola vez para crear los catalogs
        db.flush()

        # Crear todos los segmentos en memoria primero
        for class_id, geom, area_pixels in features_geo:
            class_label = CLASS_MAPPING.get(class_id, f"unknown_{class_id}")
            catalog_id = class_label
            catalog = catalog_cache[catalog_id]

            geom_4326 = reproject_geometry(geom, scene_epsg, 4326)
            area_m2 = geometry_area_m2(geom, scene_epsg)

            confidence = stats.get(class_id, {}).get("percentage", 0.0) / 100.0

            segment = Segment(
                scene_id=scene.id,
                region_id=scene.region_id,
                class_id=catalog.id,
                class_name=catalog.name,
                periodo=scene.capture_date.strftime("%Y-%m"),
                confidence=confidence,
                area_m2=area_m2,
                source="dl_segmentation",
                geometry=to_wkb(geom_4326),
            )
            segments_to_add.append(segment)
            segment_ids.append(str(segment.id))

        # Hacer bulk insert de todos los segmentos
        db.add_all(segments_to_add)
        db.flush()

        return segment_ids

    def _calculate_pixel_coverage(
        self,
        mask: np.ndarray,
        num_classes: int,
    ) -> List[PixelCoverageItem]:
        """
        Calcula cobertura basada en píxeles para cada clase.
        Replika la lógica del notebook: predict_external_image
        
        Args:
            mask: Máscara de predicción con índices de clase (512x512)
            num_classes: Número total de clases
        
        Returns:
            Lista de PixelCoverageItem ordenada por cobertura descendente
        """
        total_pixels = mask.size  # 512 * 512 = 262144
        coverage_data = []
        
        for class_id in range(num_classes):
            # Contar píxeles de esta clase
            pixel_count = int(np.sum(mask == class_id))
            
            # Calcular porcentaje con 2 decimales
            percentage = round(float((pixel_count / total_pixels) * 100), 2)
            
            # Obtener nombre de la clase
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            
            coverage_data.append(
                PixelCoverageItem(
                    class_id=int(class_id),
                    class_name=class_name,
                    pixel_count=pixel_count,
                    coverage_percentage=percentage,
                )
            )
        
        # Ordenar por cobertura descendente
        coverage_data.sort(key=lambda x: x.coverage_percentage, reverse=True)
        
        return coverage_data

    def _save_coverage_to_db(
        self,
        db: Session,
        scene: Scene,
        coverage_data: List[PixelCoverageItem],
        total_pixels: int,
    ) -> SegmentationResult:
        """
        Guarda los datos de cobertura en la base de datos.
        
        Args:
            db: Sesión de base de datos
            scene: Escena a la que pertenece la segmentación
            coverage_data: Lista de PixelCoverageItem
            total_pixels: Total de píxeles analizados
        
        Returns:
            SegmentationResult guardado en BD
        """
        # Convertir PixelCoverageItem a dict para JSONB
        coverage_dict = [
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "pixel_count": item.pixel_count,
                "coverage_percentage": item.coverage_percentage,
            }
            for item in coverage_data
        ]
        
        # Crear registro de resultado de segmentación
        segmentation_result = SegmentationResult(
            scene_id=scene.id,
            total_pixels=total_pixels,
            image_resolution="512x512",
            coverage_by_class=coverage_dict,
        )
        db.add(segmentation_result)
        db.flush()  # Para obtener el ID
        
        # Crear resumen para búsquedas rápidas
        summary = SegmentationSummary(
            segmentation_result_id=segmentation_result.id,
            dominant_class_name=coverage_data[0].class_name,
            dominant_class_percentage=coverage_data[0].coverage_percentage,
            second_class_name=coverage_data[1].class_name if len(coverage_data) > 1 else None,
            second_class_percentage=coverage_data[1].coverage_percentage if len(coverage_data) > 1 else None,
        )
        db.add(summary)
        
        return segmentation_result
    def _save_mask_rgb(
        self,
        db: Session,
        scene: Scene,
        mask_ids: np.ndarray,
        original_image: np.ndarray,
        transform: Affine,
        crs: int | None,
    ) -> None:
        """
        Convierte la máscara de índices a RGB y la guarda como GeoTIFF.
        
        Args:
            db: Sesión de base de datos
            scene: Escena a la que pertenece la máscara
            mask_ids: Máscara con índices de clase (H, W)
            original_image: Imagen original para referencia
            transform: Transformación Affine para georeferenciación
            crs: EPSG code del CRS (puede ser None)
        """
        import rasterio
        from rasterio.transform import Affine
        
        # Si no hay CRS, asumir EPSG:32717 (Ecuador default)
        if crs is None:
            crs = 32717
            logger.warning(f"[_save_mask_rgb] CRS None, usando default EPSG:32717")
        
        # Obtener colores del backend (mismos que en notebook)
        class_colors_rgb = {
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
        
        # Crear imagen RGB
        mask_rgb = np.zeros((*mask_ids.shape, 3), dtype=np.uint8)
        
        for class_id, color_rgb in class_colors_rgb.items():
            mask_rgb[mask_ids == class_id] = color_rgb
        
        # Crear directorio de escena si no existe
        scene_dir = Path(self.settings.data_dir) / "scenes" / str(scene.id)
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar como GeoTIFF
        mask_path = scene_dir / "mask_predicted_rgb.tif"
        
        logger.info(f"[_save_mask_rgb] Guardando máscara RGB en {mask_path} con CRS EPSG:{crs}")
        
        try:
            # Intentar crear CRS con el código EPSG
            from rasterio.crs import CRS as Rasterio_CRS
            crs_obj = Rasterio_CRS.from_epsg(crs)
        except Exception as e:
            logger.warning(f"[_save_mask_rgb] Error al crear CRS EPSG:{crs}: {e}. Usando string directo.")
            crs_obj = f'EPSG:{crs}'
        
        try:
            with rasterio.open(
                str(mask_path),
                'w',
                driver='GTiff',
                height=mask_rgb.shape[0],
                width=mask_rgb.shape[1],
                count=3,
                dtype=mask_rgb.dtype,
                transform=transform,
                crs=crs_obj,
            ) as dst:
                dst.write(mask_rgb[:, :, 0], 1)  # R
                dst.write(mask_rgb[:, :, 1], 2)  # G
                dst.write(mask_rgb[:, :, 2], 3)  # B
            
            logger.info(f"[_save_mask_rgb] Máscara RGB guardada exitosamente")
        except Exception as e:
            # Si falla incluso sin CRS, guardar sin información de proyección
            logger.warning(f"[_save_mask_rgb] Error al escribir con CRS, intentando sin CRS: {e}")
            try:
                with rasterio.open(
                    str(mask_path),
                    'w',
                    driver='GTiff',
                    height=mask_rgb.shape[0],
                    width=mask_rgb.shape[1],
                    count=3,
                    dtype=mask_rgb.dtype,
                    transform=transform,
                ) as dst:
                    dst.write(mask_rgb[:, :, 0], 1)  # R
                    dst.write(mask_rgb[:, :, 1], 2)  # G
                    dst.write(mask_rgb[:, :, 2], 3)  # B
                
                logger.info(f"[_save_mask_rgb] Máscara RGB guardada sin información de proyección")
            except Exception as e2:
                logger.error(f"[_save_mask_rgb] Error crítico al guardar máscara: {e2}")
                raise