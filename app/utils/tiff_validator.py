"""
TIFF Validation and Metadata Extraction Service

Validates GeoTIFF files and extracts metadata including:
- CRS/EPSG information
- Raster dimensions and band count
- Geospatial bounds
- No data values
- Compression info
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import rasterio
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)


class TiffValidationError(Exception):
    """Base exception for TIFF validation errors"""
    pass


class InvalidTiffFileError(TiffValidationError):
    """Raised when file is not a valid TIFF"""
    pass


class UnsupportedCRSError(TiffValidationError):
    """Raised when CRS is not supported or missing"""
    pass


class InsufficientBandsError(TiffValidationError):
    """Raised when image doesn't have enough bands for processing"""
    pass


class ImageSizeTooSmallError(TiffValidationError):
    """Raised when image dimensions are too small"""
    pass


@dataclass
class TiffMetadata:
    """Container for TIFF metadata"""
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None
    epsg: int | None
    nodata_value: int | float | None
    bounds: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    pixel_width: float
    pixel_height: float
    compression: str | None
    photometric: str | None
    has_geotransform: bool
    is_valid: bool
    warnings: list[str]
    info: str


class TiffValidator:
    """Validates TIFF files and extracts metadata"""
    
    MIN_WIDTH = 256
    MIN_HEIGHT = 256
    SUPPORTED_BANDS = [1, 3, 4, 5]  # Grayscale, RGB, RGBN, 5-band
    SUPPORTED_DTYPES = ['uint8', 'uint16', 'uint32', 'int16', 'int32', 'float32', 'float64']
    
    @staticmethod
    def validate(tiff_bytes: bytes) -> TiffMetadata:
        """
        Validates a TIFF file and extracts metadata.
        
        Args:
            tiff_bytes: Raw TIFF file content in bytes
            
        Returns:
            TiffMetadata: Extracted metadata and validation status
            
        Raises:
            InvalidTiffFileError: If file is not a valid TIFF
            UnsupportedCRSError: If CRS is missing or unsupported
            InsufficientBandsError: If band count is not supported
            ImageSizeTooSmallError: If dimensions are too small
        """
        warnings = []
        
        try:
            with MemoryFile(tiff_bytes) as memfile:
                with memfile.open() as src:
                    # Extract basic info
                    width = src.width
                    height = src.height
                    bands = src.count
                    dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
                    
                    # Extract CRS info
                    crs = str(src.crs) if src.crs else None
                    epsg = src.crs.to_epsg() if src.crs else None
                    
                    # Extract nodata value
                    nodata_value = src.nodata
                    
                    # Extract bounds
                    bounds = src.bounds
                    
                    # Extract pixel dimensions
                    pixel_width = abs(src.transform.a)  # pixel width in map units
                    pixel_height = abs(src.transform.e)  # pixel height in map units
                    
                    # Extract compression info
                    compression = src.compression
                    photometric = src.photometric
                    
                    # Check if has geotransform
                    has_geotransform = src.transform is not None
                    
                    # Get full file info
                    info = src.profile
                    
                    # Validation checks
                    is_valid = True
                    
                    # Check dimensions
                    if width < TiffValidator.MIN_WIDTH or height < TiffValidator.MIN_HEIGHT:
                        is_valid = False
                        raise ImageSizeTooSmallError(
                            f"Image dimensions {width}x{height} are too small. "
                            f"Minimum required: {TiffValidator.MIN_WIDTH}x{TiffValidator.MIN_HEIGHT}"
                        )
                    
                    # Check number of bands
                    if bands not in TiffValidator.SUPPORTED_BANDS:
                        raise InsufficientBandsError(
                            f"Unsupported band count: {bands}. "
                            f"Supported: {TiffValidator.SUPPORTED_BANDS}"
                        )
                    
                    # Check data type
                    if dtype not in TiffValidator.SUPPORTED_DTYPES:
                        warnings.append(
                            f"Unusual data type: {dtype}. "
                            f"Supported: {TiffValidator.SUPPORTED_DTYPES}"
                        )
                    
                    # Check CRS
                    if not crs:
                        warnings.append("No CRS information found in TIFF")
                    elif epsg is None and crs:
                        warnings.append(
                            f"CRS found ({crs}) but cannot convert to EPSG code"
                        )
                    
                    # Check for nodata
                    # Not warning about missing nodata - it's optional for many image types
                    # if nodata_value is None and bands > 3:
                    #     warnings.append(
                    #         "No nodata value defined. Segmentation may include invalid pixels."
                    #     )
                    
                    # Check geotransform
                    if not has_geotransform:
                        warnings.append(
                            "No geotransform information found. "
                            "Geographic bounds will not be accurate."
                        )
                    
                    # Check compression
                    if compression and compression not in ['lzw', 'deflate', 'zstd', 'none']:
                        warnings.append(
                            f"Compression type '{compression}' may not be widely supported"
                        )
                    
                    # Calculate approximate file size stats
                    size_mb = len(tiff_bytes) / (1024 * 1024)
                    if size_mb > 500:
                        warnings.append(
                            f"Large file ({size_mb:.1f} MB). Processing may take longer."
                        )
                    
                    logger.info(
                        f"TIFF validation successful: {width}x{height}, "
                        f"{bands} bands, EPSG:{epsg}, {size_mb:.2f}MB"
                    )
                    
                    return TiffMetadata(
                        width=width,
                        height=height,
                        bands=bands,
                        dtype=dtype,
                        crs=crs,
                        epsg=epsg,
                        nodata_value=nodata_value,
                        bounds=bounds,
                        pixel_width=pixel_width,
                        pixel_height=pixel_height,
                        compression=compression,
                        photometric=photometric,
                        has_geotransform=has_geotransform,
                        is_valid=is_valid,
                        warnings=warnings,
                        info=str(info)
                    )
        
        except TiffValidationError:
            raise
        except rasterio.errors.RasterioIOError as e:
            logger.error(f"Failed to read TIFF file: {e}")
            raise InvalidTiffFileError(f"Invalid or corrupted TIFF file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error validating TIFF: {e}")
            raise InvalidTiffFileError(f"Error validating TIFF: {e}") from e
    
    @staticmethod
    def validate_epsg_match(tiff_bytes: bytes, expected_epsg: int) -> bool:
        """
        Validates if TIFF's EPSG matches expected EPSG.
        
        Args:
            tiff_bytes: Raw TIFF file content
            expected_epsg: Expected EPSG code
            
        Returns:
            bool: True if EPSG matches, False otherwise
            
        Raises:
            UnsupportedCRSError: If TIFF doesn't have EPSG info
        """
        try:
            with MemoryFile(tiff_bytes) as memfile:
                with memfile.open() as src:
                    if not src.crs:
                        raise UnsupportedCRSError(
                            "TIFF file has no CRS information"
                        )
                    
                    tiff_epsg = src.crs.to_epsg()
                    if tiff_epsg is None:
                        logger.warning(
                            f"Could not convert TIFF CRS to EPSG: {src.crs}"
                        )
                        return False
                    
                    return tiff_epsg == expected_epsg
        
        except rasterio.errors.RasterioIOError as e:
            raise InvalidTiffFileError(f"Cannot read TIFF file: {e}") from e
