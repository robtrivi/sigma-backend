from __future__ import annotations

import logging
import uuid
from typing import List

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio import features
from shapely.geometry import shape
from sklearn.cluster import KMeans
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import ClassCatalog, Scene, Segment
from app.schemas.schemas import SceneSegmentRequest, SegmentsImportResponse
from app.utils.geo import geometry_area_m2, load_multipolygon, reproject_geometry, to_wkb

logger = logging.getLogger(__name__)


class SegmentationRasterService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.palette = [
            "#4CAF50",
            "#FFC107",
            "#2196F3",
            "#9C27B0",
            "#FF5722",
            "#00BCD4",
            "#8BC34A",
        ]

    def segment_scene(self, db: Session, scene_id: str, payload: SceneSegmentRequest) -> SegmentsImportResponse:
        scene = db.get(Scene, uuid.UUID(scene_id))
        if not scene:
            raise ValueError("Scene not found")
        dataset, transform = self._read_raster(scene.raster_path, payload)
        clusters = self._cluster(dataset, payload.n_classes)
        features_geo = self._vectorize(clusters, transform)
        segment_ids: list[str] = []
        for geom_geojson, cluster_value in features_geo:
            geom = load_multipolygon(geom_geojson)
            geom_4326 = reproject_geometry(geom, scene.epsg, 4326)
            area_m2 = geometry_area_m2(geom, scene.epsg)
            class_id = f"auto_{int(cluster_value)}"
            catalog = db.get(ClassCatalog, class_id)
            if not catalog:
                color = self.palette[int(cluster_value) % len(self.palette)]
                catalog = ClassCatalog(
                    id=class_id,
                    name=f"Auto Cluster {int(cluster_value)}",
                    color_hex=color,
                    icono_primeng="pi pi-map",
                    description="Clase generada automáticamente",
                )
                db.add(catalog)
                db.flush()
            segment = Segment(
                scene_id=scene.id,
                region_id=scene.region_id,
                class_id=catalog.id,
                class_name=catalog.name,
                periodo=scene.capture_date.strftime("%Y-%m"),
                confidence=0.6,
                area_m2=area_m2,
                source=payload.method,
                geometry=to_wkb(geom_4326),
            )
            db.add(segment)
            db.flush()
            segment_ids.append(str(segment.id))
        db.commit()
        return SegmentsImportResponse(inserted=len(segment_ids), segmentIds=segment_ids)

    def _read_raster(self, path: str, payload: SceneSegmentRequest) -> tuple[np.ndarray, Affine]:
        with rasterio.open(path) as src:
            bands = payload.bands or list(range(1, min(4, src.count + 1)))
            data = src.read(bands)
            if payload.downscale_factor:
                factor = payload.downscale_factor
                new_height = max(1, int(src.height / factor))
                new_width = max(1, int(src.width / factor))
                data = src.read(
                    bands,
                    out_shape=(len(bands), new_height, new_width),
                    resampling=Resampling.bilinear,
                )
                scale_y = src.height / new_height
                scale_x = src.width / new_width
                transform = src.transform * Affine.scale(scale_x, scale_y)
            else:
                transform = src.transform
        return data, transform

    def _cluster(self, data: np.ndarray, n_classes: int) -> np.ndarray:
        bands, height, width = data.shape
        reshaped = data.reshape(bands, -1).T
        model = KMeans(n_clusters=n_classes, n_init="auto")
        labels = model.fit_predict(reshaped)
        return labels.reshape(height, width)

    def _vectorize(self, clusters: np.ndarray, transform: Affine) -> List[tuple[dict, int]]:
        shapes_iter = features.shapes(clusters.astype(np.int16), transform=transform)
        polygons: list[tuple[dict, int]] = []
        for geom, value in shapes_iter:
            if value is None:
                continue
            geom_geojson = (
                {"type": "MultiPolygon", "coordinates": [geom["coordinates"]]}
                if geom["type"] == "Polygon"
                else geom
            )
            polygons.append((geom_geojson, int(value)))
        return polygons
