"""
API endpoints for TIFF validation and metadata extraction.

Provides endpoints to:
- Validate TIFF files before full upload
- Extract metadata from TIFF files
- Check CRS compatibility
"""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.schemas.schemas import (
    TiffBandInfo,
    TiffBoundsInfo,
    TiffMetadataResponse,
    TiffPixelSize,
    TiffValidationResponse,
)
from app.utils.tiff_metadata_extractor import TiffMetadataExtractor
from app.utils.tiff_validator import (
    ImageSizeTooSmallError,
    InsufficientBandsError,
    InvalidTiffFileError,
    TiffValidator,
    UnsupportedCRSError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/imports", tags=["imports-validation"])
settings = get_settings()

# Model input size (should match the segmentation model)
MODEL_HEIGHT = 256
MODEL_WIDTH = 256
ESTIMATED_PROCESSING_TIME_PER_MEGAPIXEL = 0.01  # seconds


@router.post("/validate-tiff", response_model=TiffValidationResponse)
async def validate_tiff(
    file: UploadFile = File(...),
    epsg_code: int | None = Query(None),
) -> TiffValidationResponse:
    """
    Validates a TIFF file before uploading for segmentation.
    
    This endpoint performs quick validation without storing the file.
    
    Args:
        file: TIFF file to validate
        epsg_code: Optional EPSG code to validate against TIFF's CRS
    
    Returns:
        TiffValidationResponse: Validation result with metadata
        
    Raises:
        HTTPException 400: If TIFF is invalid
    """
    try:
        # Read file into bytes
        tiff_bytes = await file.read()
        
        if not tiff_bytes:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Validate TIFF structure and metadata
        metadata = TiffValidator.validate(tiff_bytes)
        
        # Check EPSG match if provided
        if epsg_code is not None:
            if metadata.epsg != epsg_code:
                logger.warning(
                    f"EPSG mismatch: File has {metadata.epsg}, "
                    f"but {epsg_code} was specified"
                )
        
        # Calculate estimated processing time
        megapixels = (metadata.width * metadata.height) / (1024 * 1024)
        estimated_time = megapixels * ESTIMATED_PROCESSING_TIME_PER_MEGAPIXEL
        
        # Convert metadata to response format
        bands = [
            TiffBandInfo(
                index=i+1,
                dtype=metadata.dtype,
            )
            for i in range(metadata.bands)
        ]
        
        bounds = TiffBoundsInfo(
            minx=metadata.bounds[0],
            miny=metadata.bounds[1],
            maxx=metadata.bounds[2],
            maxy=metadata.bounds[3],
        )
        
        bounds_wgs84 = None
        
        pixel_size = TiffPixelSize(
            width=metadata.pixel_width,
            height=metadata.pixel_height,
        )
        
        # Calculate file size in MB
        file_size_mb = len(tiff_bytes) / (1024 * 1024)
        
        logger.info(
            f"TIFF validation successful for {file.filename}: "
            f"{metadata.width}x{metadata.height}, {metadata.bands} bands, "
            f"EPSG:{metadata.epsg}"
        )
        
        return TiffValidationResponse(
            valid=metadata.is_valid,
            width=metadata.width,
            height=metadata.height,
            bandCount=metadata.bands,
            dtype=metadata.dtype,
            epsgCode=metadata.epsg,
            crsWkt=metadata.crs,
            bounds=bounds,
            boundsWgs84=bounds_wgs84,
            pixelSize=pixel_size,
            compression=metadata.compression,
            photometric=metadata.photometric,
            bands=bands,
            fileSizeMb=file_size_mb,
            warnings=metadata.warnings,
            estimatedProcessingTimeSec=estimated_time,
        )
    
    except InvalidTiffFileError as e:
        logger.warning(f"Invalid TIFF file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    except InsufficientBandsError as e:
        logger.warning(f"Insufficient bands in {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    except ImageSizeTooSmallError as e:
        logger.warning(f"Image too small {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    except UnsupportedCRSError as e:
        logger.warning(f"Unsupported CRS in {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    except Exception as e:
        logger.error(f"Unexpected error validating TIFF {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error validating TIFF file"
        ) from e


@router.post("/tiff-metadata", response_model=TiffMetadataResponse)
async def extract_tiff_metadata(
    file: UploadFile = File(...),
) -> TiffMetadataResponse:
    """
    Extracts comprehensive metadata from a TIFF file.
    
    This endpoint provides detailed information including:
    - Band statistics (min, max values per band)
    - Complete TIFF tags
    - Estimated sensor information
    - Capture date if available in metadata
    
    Args:
        file: TIFF file to extract metadata from
    
    Returns:
        TiffMetadataResponse: Comprehensive TIFF metadata
        
    Raises:
        HTTPException 400: If TIFF is invalid
    """
    try:
        tiff_bytes = await file.read()
        
        if not tiff_bytes:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Extract metadata
        info = TiffMetadataExtractor.extract(tiff_bytes)
        
        # Convert bands to response format
        bands = [
            TiffBandInfo(
                index=b.index,
                dtype=b.dtype,
                minValue=b.min_value,
                maxValue=b.max_value,
                nodata=b.nodata,
                colorInterp=b.colorinterp,
            )
            for b in info.bands
        ]
        
        # Convert bounds
        bounds = TiffBoundsInfo(
            minx=info.bounds["minx"],
            miny=info.bounds["miny"],
            maxx=info.bounds["maxx"],
            maxy=info.bounds["maxy"],
        )
        
        bounds_wgs84 = None
        if info.bounds_wgs84:
            bounds_wgs84 = TiffBoundsInfo(
                minx=info.bounds_wgs84["minx"],
                miny=info.bounds_wgs84["miny"],
                maxx=info.bounds_wgs84["maxx"],
                maxy=info.bounds_wgs84["maxy"],
            )
        
        pixel_size = TiffPixelSize(
            width=info.pixel_size["width"],
            height=info.pixel_size["height"],
        )
        
        # Extract capture date and sensor hints
        capture_date = TiffMetadataExtractor.extract_capture_date_from_tags(info.tags)
        sensor_hints = TiffMetadataExtractor.get_sensor_hints(info.tags)
        
        logger.info(
            f"Extracted metadata from {file.filename}: "
            f"{info.width}x{info.height}, {info.band_count} bands"
        )
        
        return TiffMetadataResponse(
            width=info.width,
            height=info.height,
            bandCount=info.band_count,
            epsgCode=info.epsg_code,
            crsWkt=info.crs_wkt,
            bounds=bounds,
            boundsWgs84=bounds_wgs84,
            pixelSize=pixel_size,
            compression=info.compression,
            photometric=info.photometric,
            bands=bands,
            tags=info.tags,
            fileSizeBytes=info.file_size_bytes,
            fileSizeMb=info.file_size_mb,
            warnings=info.warnings,
            captureDate=capture_date,
            sensorHints=sensor_hints,
        )
    
    except InvalidTiffFileError as e:
        logger.warning(f"Invalid TIFF file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    except Exception as e:
        logger.error(f"Unexpected error extracting TIFF metadata {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error extracting TIFF metadata"
        ) from e


@router.get("/check-epsg-support")
async def check_epsg_support(
    epsg_code: int = Query(..., description="EPSG code to check"),
    db: Session = Depends(get_db),
) -> dict[str, bool | str]:
    """
    Checks if a specific EPSG code is supported for uploads in a given region.
    
    Args:
        epsg_code: EPSG code to check
        db: Database session
    
    Returns:
        dict: Support status and description
    """
    # This is a placeholder - in a real implementation,
    # you would check against a database of supported EPSG codes per region
    
    SUPPORTED_EPSG = {
        4326: "WGS 84 (Geographic)",
        3857: "Web Mercator",
        32717: "WGS 84 / UTM zone 17S (Ecuador)",
        32718: "WGS 84 / UTM zone 18S",
    }
    
    is_supported = epsg_code in SUPPORTED_EPSG
    description = SUPPORTED_EPSG.get(epsg_code, "Unknown EPSG code")
    
    return {
        "supported": is_supported,
        "epsgCode": epsg_code,
        "description": description,
    }
