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
        scene = db.get(Scene, uuid.UUID(scene_id))
        if not scene:
            raise ValueError("Scene not found")

        # Validate TIFF before processing
        try:
            tiff_metadata = TiffValidator.validate(tiff_bytes)
            logger.info(
                f"Scene {scene_id} TIFF validated: "
                f"{tiff_metadata.width}x{tiff_metadata.height}, "
                f"EPSG:{tiff_metadata.epsg}"
            )
        except Exception as e:
            logger.error(f"TIFF validation failed for scene {scene_id}: {e}")
            raise ValueError(f"Invalid TIFF file: {e}") from e

        model = get_model(self.settings)
        model_height, model_width = get_model_input_shape(model)
        num_classes = get_model_num_classes(model)

        image_data, transform, crs, original_shape = self._read_tiff(tiff_bytes)
        preprocessed = self._preprocess_image(image_data, model_height, model_width)
        predictions = model.predict(preprocessed, verbose=0)
        mask_ids_model = np.argmax(predictions[0], axis=-1)
        
        # ========== NUEVO: Calcular cobertura por píxeles ==========
        coverage_data = self._calculate_pixel_coverage(mask_ids_model, num_classes)
        total_pixels = mask_ids_model.size
        
        # Guardar cobertura en BD
        self._save_coverage_to_db(db, scene, coverage_data, total_pixels)
        logger.info(f"Pixel coverage analysis saved for scene {scene_id}")
        # =========================================================
        
        mask_ids_original = self._resize_mask(
            mask_ids_model, original_shape[0], original_shape[1]
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
        return SegmentsImportResponse(inserted=len(segment_ids), segmentIds=segment_ids)

    def _read_tiff(
        self, tiff_bytes: bytes
    ) -> Tuple[np.ndarray, Affine, int | None, Tuple[int, int]]:
        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as src:
                data = src.read()
                transform = src.transform
                crs_epsg = None
                if src.crs is not None:
                    crs_epsg = src.crs.to_epsg()
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

        for class_id, geom, area_pixels in features_geo:
            class_label = CLASS_MAPPING.get(class_id, f"unknown_{class_id}")
            catalog_id = class_label

            catalog = db.get(ClassCatalog, catalog_id)
            if not catalog:
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
                db.flush()

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
            db.add(segment)
            db.flush()
            segment_ids.append(str(segment.id))

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
