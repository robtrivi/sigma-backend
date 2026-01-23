from __future__ import annotations

import logging
from typing import Iterable, Tuple

from geoalchemy2.shape import from_shape, to_shape
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as shapely_transform
from shapely.validation import explain_validity

logger = logging.getLogger(__name__)


def load_multipolygon(geometry: dict) -> MultiPolygon:
    geom = shape(geometry)
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    if not isinstance(geom, MultiPolygon):
        raise ValueError("Geometry must be Polygon or MultiPolygon")
    if not geom.is_valid:
        message = explain_validity(geom)
        raise ValueError(f"Invalid geometry: {message}")
    return geom


def to_wkb(geometry: MultiPolygon, srid: int = 4326) -> bytes:
    return from_shape(geometry, srid=srid)


def to_geojson(db_geometry) -> dict:
    if hasattr(db_geometry, 'data'):
        geom = to_shape(db_geometry)
    else:
        geom = wkb.loads(bytes(db_geometry))
    
    return geom.__geo_interface__


def parse_bbox(bbox: str) -> Tuple[float, float, float, float]:
    parts = [float(p) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must contain four comma-separated values")
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox coordinates are invalid")
    return min_lon, min_lat, max_lon, max_lat


def flatten(iterable: Iterable) -> list:
    return list(iterable)


def reproject_geometry(geometry, src_epsg: int, dst_epsg: int):
    if src_epsg == dst_epsg:
        return geometry
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def geometry_area_m2(geometry, src_epsg: int = 4326) -> float:
    projected = reproject_geometry(geometry, src_epsg, 3857)
    return projected.area
