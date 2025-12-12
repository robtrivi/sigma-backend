"""
TIFF Metadata Extraction Service

Provides comprehensive metadata extraction from GeoTIFF files using both
tifffile and rasterio libraries for complete introspection.
"""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

try:
    import tifffile
except ImportError:
    tifffile = None

import rasterio
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)


@dataclass
class BandInfo:
    """Information about a single band"""
    index: int
    dtype: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nodata: Optional[Any] = None
    colorinterp: Optional[str] = None


@dataclass
class TiffFileInfo:
    """Comprehensive TIFF file information"""
    # Raster dimensions
    width: int
    height: int
    band_count: int
    
    # CRS information
    crs_wkt: Optional[str]
    epsg_code: Optional[int]
    
    # Geospatial info
    bounds: dict[str, float]  # minx, miny, maxx, maxy
    bounds_wgs84: Optional[dict[str, float]]  # Reprojected to WGS84
    pixel_size: dict[str, float]  # width, height in map units
    
    # File format info
    compression: Optional[str]
    photometric: Optional[str]
    dtype_primary: str
    
    # Band details
    bands: list[BandInfo]
    
    # Metadata tags
    tags: dict[str, Any]
    
    # File statistics
    file_size_bytes: int
    file_size_mb: float
    
    # Warnings/notes
    warnings: list[str]


class TiffMetadataExtractor:
    """Extracts comprehensive metadata from GeoTIFF files"""
    
    @staticmethod
    def extract(tiff_bytes: bytes) -> TiffFileInfo:
        """
        Extracts comprehensive metadata from TIFF file.
        
        Args:
            tiff_bytes: Raw TIFF file content in bytes
            
        Returns:
            TiffFileInfo: Comprehensive metadata about the TIFF file
        """
        warnings = []
        file_size_mb = len(tiff_bytes) / (1024 * 1024)
        
        # Extract rasterio info
        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as src:
                width = src.width
                height = src.height
                band_count = src.count
                
                # CRS information
                crs_wkt = str(src.crs) if src.crs else None
                epsg_code = src.crs.to_epsg() if src.crs else None
                
                # Bounds
                bounds = {
                    "minx": src.bounds.left,
                    "miny": src.bounds.bottom,
                    "maxx": src.bounds.right,
                    "maxy": src.bounds.top,
                }
                
                # Reproject bounds to WGS84 if needed
                bounds_wgs84 = None
                if src.crs and epsg_code != 4326:
                    try:
                        from rasterio.warp import transform_bounds
                        wgs84_bounds = transform_bounds(
                            src.crs, "EPSG:4326", *src.bounds
                        )
                        bounds_wgs84 = {
                            "minx": wgs84_bounds[0],
                            "miny": wgs84_bounds[1],
                            "maxx": wgs84_bounds[2],
                            "maxy": wgs84_bounds[3],
                        }
                    except Exception as e:
                        warnings.append(f"Could not reproject bounds to WGS84: {e}")
                
                # Pixel size
                pixel_width = abs(src.transform.a)
                pixel_height = abs(src.transform.e)
                pixel_size = {"width": pixel_width, "height": pixel_height}
                
                # File format info
                compression = src.compression
                photometric = src.photometric
                dtype_primary = src.dtypes[0] if src.dtypes else "unknown"
                
                # Band information
                bands = []
                for i in range(1, band_count + 1):
                    try:
                        band_data = src.read(i)
                        colorinterp = src.colorinterp[i - 1] if src.colorinterp else None
                        
                        band_info = BandInfo(
                            index=i,
                            dtype=src.dtypes[i - 1],
                            min_value=float(band_data.min()),
                            max_value=float(band_data.max()),
                            nodata=src.nodata,
                            colorinterp=str(colorinterp) if colorinterp else None,
                        )
                        bands.append(band_info)
                    except Exception as e:
                        warnings.append(f"Could not read band {i}: {e}")
                        bands.append(
                            BandInfo(
                                index=i,
                                dtype=src.dtypes[i - 1] if i < len(src.dtypes) else "unknown",
                                colorinterp=str(src.colorinterp[i - 1]) if src.colorinterp and i < len(src.colorinterp) else None,
                            )
                        )
                
                # Extract tags
                tags = {}
                try:
                    # Try to get GeoTIFF tags
                    if hasattr(src, 'tags'):
                        # rasterio v1.3+
                        for tag_set in src.tags().values() if isinstance(src.tags(), dict) else []:
                            if isinstance(tag_set, dict):
                                tags.update(tag_set)
                except Exception as e:
                    logger.debug(f"Could not extract tags: {e}")
                
                # Additional info using tifffile if available
                if tifffile:
                    try:
                        with tifffile.TiffFile(io.BytesIO(tiff_bytes)) as tif:
                            # Extract TIFF tags from first image
                            if tif.series:
                                pages = tif.series[0].pages
                                if pages:
                                    page = pages[0]
                                    # Add TIFF-specific tags
                                    if hasattr(page, 'tags'):
                                        for tag_name, tag in page.tags.items():
                                            tags[tag_name] = str(tag.value)
                                    
                                    # Get page shape and dtype
                                    tags['tifffile_shape'] = page.shape
                                    tags['tifffile_dtype'] = str(page.dtype)
                    except Exception as e:
                        logger.debug(f"tifffile extraction failed: {e}")
        
        # Validate CRS
        if not crs_wkt:
            warnings.append("No CRS information found in TIFF")
        
        logger.info(
            f"Extracted metadata: {width}x{height}x{band_count}, "
            f"EPSG:{epsg_code}, {file_size_mb:.2f}MB"
        )
        
        return TiffFileInfo(
            width=width,
            height=height,
            band_count=band_count,
            crs_wkt=crs_wkt,
            epsg_code=epsg_code,
            bounds=bounds,
            bounds_wgs84=bounds_wgs84,
            pixel_size=pixel_size,
            compression=compression,
            photometric=photometric,
            dtype_primary=dtype_primary,
            bands=bands,
            tags=tags,
            file_size_bytes=len(tiff_bytes),
            file_size_mb=file_size_mb,
            warnings=warnings,
        )
    
    @staticmethod
    def to_dict(info: TiffFileInfo) -> dict[str, Any]:
        """Converts TiffFileInfo to dictionary for JSON serialization"""
        data = asdict(info)
        data['bands'] = [asdict(b) for b in info.bands]
        return data
    
    @staticmethod
    def extract_capture_date_from_tags(tags: dict[str, Any]) -> Optional[str]:
        """
        Attempts to extract capture date from TIFF tags.
        
        Looks for common TIFF tags like:
        - DateTime
        - ImageDescription
        - XMP metadata
        """
        # Try common datetime tags
        for tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
            if tag_name in tags:
                return str(tags[tag_name])
        
        # Try to parse from ImageDescription
        if 'ImageDescription' in tags:
            desc = str(tags['ImageDescription'])
            # Look for ISO date format
            import re
            match = re.search(r'\d{4}-\d{2}-\d{2}', desc)
            if match:
                return match.group(0)
        
        return None
    
    @staticmethod
    def get_sensor_hints(tags: dict[str, Any]) -> dict[str, str]:
        """
        Extracts sensor information from TIFF tags.
        
        Returns:
            dict with keys: model, software, camera_make
        """
        hints = {}
        
        for tag_name in ['Model', 'CameraModel']:
            if tag_name in tags:
                hints['model'] = str(tags[tag_name])
                break
        
        for tag_name in ['Software']:
            if tag_name in tags:
                hints['software'] = str(tags[tag_name])
                break
        
        for tag_name in ['Make', 'CameraMake']:
            if tag_name in tags:
                hints['camera_make'] = str(tags[tag_name])
                break
        
        return hints
