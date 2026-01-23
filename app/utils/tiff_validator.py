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
import warnings
from dataclasses import dataclass
from typing import Optional

import rasterio
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)

# Suprimir warnings de rasterio/GDAL a nivel de módulo
warnings.filterwarnings('ignore', message='.*PROJ.*')
warnings.filterwarnings('ignore', message='.*CPLE_AppDefined.*')
warnings.filterwarnings('ignore', message='.*GTIFF_SRS_SOURCE.*')


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
    def _validate_dimensions(width: int, height: int) -> None:
        """Valida que las dimensiones sean suficientemente grandes."""
        if width < TiffValidator.MIN_WIDTH or height < TiffValidator.MIN_HEIGHT:
            raise ImageSizeTooSmallError(
                f"Image dimensions {width}x{height} are too small. "
                f"Minimum required: {TiffValidator.MIN_WIDTH}x{TiffValidator.MIN_HEIGHT}"
            )
    
    @staticmethod
    def _validate_bands(bands: int) -> None:
        """Valida que el número de bandas sea soportado."""
        if bands not in TiffValidator.SUPPORTED_BANDS:
            raise InsufficientBandsError(
                f"Unsupported band count: {bands}. "
                f"Supported: {TiffValidator.SUPPORTED_BANDS}"
            )
    
    @staticmethod
    def _check_dtype_warning(dtype: str, warnings_list: list) -> None:
        """Verifica si el tipo de dato es inusual."""
        if dtype not in TiffValidator.SUPPORTED_DTYPES:
            warnings_list.append(
                f"Unusual data type: {dtype}. "
                f"Supported: {TiffValidator.SUPPORTED_DTYPES}"
            )
    
    @staticmethod
    def _check_crs_warnings(crs: str | None, epsg: int | None, warnings_list: list) -> None:
        """Verifica y agrega warnings relacionados a CRS."""
        if not crs:
            warnings_list.append("No CRS information found in TIFF")
        elif epsg is None and crs:
            warnings_list.append(f"CRS found ({crs}) but cannot convert to EPSG code")
    
    @staticmethod
    def _check_geotransform_warning(has_geotransform: bool, warnings_list: list) -> None:
        """Verifica si hay geotransform."""
        if not has_geotransform:
            warnings_list.append(
                "No geotransform information found. "
                "Geographic bounds will not be accurate."
            )
    
    @staticmethod
    def _check_compression_warning(compression: str | None, warnings_list: list) -> None:
        """Verifica si la compresión es soportada."""
        if compression and compression not in ['lzw', 'deflate', 'zstd', 'none']:
            warnings_list.append(
                f"Compression type '{compression}' may not be widely supported"
            )
    
    @staticmethod
    def _check_file_size_warning(tiff_bytes: bytes, warnings_list: list) -> float:
        """Verifica si el tamaño del archivo es demasiado grande."""
        size_mb = len(tiff_bytes) / (1024 * 1024)
        if size_mb > 500:
            warnings_list.append(
                f"Large file ({size_mb:.1f} MB). Processing may take longer."
            )
        return size_mb
    
    @staticmethod
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
        warnings_list = []
        
        try:
            with MemoryFile(tiff_bytes) as memfile:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    
                    with memfile.open() as src:
                        # Extract basic info
                        width = src.width
                        height = src.height
                        bands = src.count
                        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
                        
                        # Extract CRS info
                        crs = None
                        epsg = None
                        try:
                            if src.crs:
                                crs = str(src.crs)
                                epsg = src.crs.to_epsg()
                        except Exception:
                            crs = None
                            epsg = None
                        
                        # Extract metadata
                        nodata_value = src.nodata
                        bounds = src.bounds
                        pixel_width = abs(src.transform.a)
                        pixel_height = abs(src.transform.e)
                        compression = src.compression
                        photometric = src.photometric
                        has_geotransform = src.transform is not None
                        info = src.profile
                        
                        # Validations
                        TiffValidator._validate_dimensions(width, height)
                        TiffValidator._validate_bands(bands)
                        TiffValidator._check_dtype_warning(dtype, warnings_list)
                        TiffValidator._check_crs_warnings(crs, epsg, warnings_list)
                        TiffValidator._check_geotransform_warning(has_geotransform, warnings_list)
                        TiffValidator._check_compression_warning(compression, warnings_list)
                        size_mb = TiffValidator._check_file_size_warning(tiff_bytes, warnings_list)
                        
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
                            is_valid=True,
                            warnings=warnings_list,
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
