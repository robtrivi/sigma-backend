from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import MultiPolygon, Polygon
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import ClassCatalog, Scene, Segment
from app.schemas.schemas import SegmentsImportResponse
from app.utils.geo import (
    geometry_area_m2,
    reproject_geometry,
    to_wkb,
)

logger = logging.getLogger(__name__)

_model = None

CLASS_MAPPING = {
    0: "obstaculos",
    1: "agua",
    2: "superficies_blandas",
    3: "objetos_en_movimiento",
    4: "zonas_aterrizables",
}

CLASS_COLORS = {
    0: "#FF5722",
    1: "#2196F3",
    2: "#4CAF50",
    3: "#FFC107",
    4: "#8BC34A",
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

        model = get_model(self.settings)
        model_height, model_width = get_model_input_shape(model)
        num_classes = get_model_num_classes(model)

        image_data, transform, crs, original_shape = self._read_tiff(tiff_bytes)
        preprocessed = self._preprocess_image(image_data, model_height, model_width)
        predictions = model.predict(preprocessed, verbose=0)
        mask_ids_model = np.argmax(predictions[0], axis=-1)
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
        resized = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        normalized = resized.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched

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
                catalog = ClassCatalog(
                    id=catalog_id,
                    name=class_label.replace("_", " ").title(),
                    color_hex=color,
                    icono_primeng="pi pi-map-marker",
                    description=f"Clase {class_label} generada por segmentación DL",
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
