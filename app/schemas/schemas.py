from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class RegionBase(BaseModel):
    id: str
    name: str


class RegionCreate(BaseModel):
    id: str
    name: str


class RegionRead(RegionBase):
    geometry: dict | None = None


class SceneUploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    scene_id: str = Field(..., alias="sceneId")
    region_id: str = Field(..., alias="regionId")
    capture_date: date = Field(..., alias="captureDate")
    epsg: int
    sensor: str
    raster_path: str = Field(..., alias="rasterPath")


class SegmentImportProperties(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    scene_id: str = Field(..., alias="sceneId")
    class_id: str = Field(..., alias="classId")
    confidence: float
    area_m2: float = Field(..., alias="areaM2")
    periodo: str
    source: str | None = "import"


class FeatureGeometry(BaseModel):
    type: str
    coordinates: Any


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: FeatureGeometry
    properties: SegmentImportProperties


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature]

    @field_validator("features")
    @classmethod
    def ensure_features(
        cls, v: List[GeoJSONFeature], info: ValidationInfo
    ) -> List[GeoJSONFeature]:
        if not v:
            raise ValueError("FeatureCollection must contain at least one feature")
        return v


class SegmentsImportResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    inserted: int
    segment_ids: List[str] = Field(..., alias="segmentIds")


class SegmentUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    class_id: Optional[str] = Field(None, alias="classId")
    confidence: Optional[float] = None
    notes: Optional[str] = None


class SegmentProperties(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    segment_id: str = Field(..., alias="segmentId")
    scene_id: str = Field(..., alias="sceneId")
    region_id: str = Field(..., alias="regionId")
    class_id: str = Field(..., alias="classId")
    class_name: str = Field(..., alias="className")
    area_m2: float = Field(..., alias="areaM2")
    periodo: str
    confidence: float
    source: str


class SegmentFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict
    properties: SegmentProperties


class SegmentFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[SegmentFeature]


class AggregationRebuildRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    periodo: str
    region_id: str = Field(..., alias="regionId")


class DistributionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    class_id: str = Field(..., alias="classId")
    class_name: str = Field(..., alias="className")
    percentage: float
    area_m2: float = Field(..., alias="areaM2")


class TrendItem(BaseModel):
    periodo: str
    value: float


class RegionSummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    region_id: str = Field(..., alias="regionId")
    periodo: str
    segments_visible: int = Field(..., alias="segmentsVisible")
    cobertura_verde: float = Field(..., alias="coberturaVerde")
    distribution: List[DistributionItem]
    trend: List[TrendItem]
    messages: List[str]


class AggregationSummaryRead(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    region_id: str = Field(..., alias="regionId")
    periodo: str
    total_area_m2: float = Field(..., alias="totalAreaM2")
    green_coverage: float = Field(..., alias="greenCoverage")
    distribution: List[DistributionItem]
    trend: List[TrendItem]


class CatalogClassRead(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    class_id: str = Field(..., alias="classId")
    nombre: str
    color_hex: str = Field(..., alias="colorHex")
    icono_prime_ng: str | None = Field(None, alias="iconoPrimeNg")
    description: str | None = None


class RegionPeriodItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    periodo: str
    region_id: str = Field(..., alias="regionId")
    segment_count: int = Field(..., alias="segmentCount")
    last_updated: datetime | None = Field(None, alias="lastUpdated")


class SubregionHistoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    periodo: str
    area_m2: float = Field(..., alias="areaM2")
    dominant_class_id: str = Field(..., alias="dominantClassId")
    dominant_class_name: str = Field(..., alias="dominantClassName")
    delta_vs_prev: float | None = Field(None, alias="deltaVsPrev")


class SubregionHistoryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    subregion_id: str = Field(..., alias="subregionId")
    region_id: str = Field(..., alias="regionId")
    history: List[SubregionHistoryItem]


class ReportDownloadRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    region_id: str = Field(..., alias="regionId")
    periodos: List[str]
    class_filters: List[str] | None = Field(None, alias="classFilters")
    segments: List[str] | None = None


class ReportJobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    report_id: str = Field(..., alias="reportId")
    download_url: str = Field(..., alias="downloadUrl")
    expires_at: datetime = Field(..., alias="expiresAt")


class ReportDownloadLink(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    report_id: str = Field(..., alias="reportId")
    region_id: str = Field(..., alias="regionId")
    filters: Dict[str, Any]
    download_url: str = Field(..., alias="downloadUrl")
    expires_at: datetime = Field(..., alias="expiresAt")


class PeriodRangeQuery(BaseModel):
    from_period: str | None = Field(default=None, alias="from")
    to_period: str | None = Field(default=None, alias="to")


class SegmentsTilesQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    region_id: str = Field(..., alias="regionId")
    periodo: str
    class_id: List[str] | None = Field(None, alias="classId")
    bbox: str | None = None


# TIFF Validation and Metadata Schemas

class TiffBandInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    index: int
    dtype: str
    min_value: Optional[float] = Field(None, alias="minValue")
    max_value: Optional[float] = Field(None, alias="maxValue")
    nodata: Optional[Any] = None
    color_interp: Optional[str] = Field(None, alias="colorInterp")


class TiffBoundsInfo(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float


class TiffPixelSize(BaseModel):
    width: float
    height: float


class TiffValidationResponse(BaseModel):
    """Response for TIFF validation endpoint"""
    model_config = ConfigDict(populate_by_name=True)
    
    valid: bool
    width: int
    height: int
    band_count: int = Field(..., alias="bandCount")
    dtype: str
    epsg_code: Optional[int] = Field(None, alias="epsgCode")
    crs_wkt: Optional[str] = Field(None, alias="crsWkt")
    bounds: TiffBoundsInfo
    bounds_wgs84: Optional[TiffBoundsInfo] = Field(None, alias="boundsWgs84")
    pixel_size: TiffPixelSize = Field(..., alias="pixelSize")
    compression: Optional[str] = None
    photometric: Optional[str] = None
    bands: List[TiffBandInfo]
    file_size_mb: float = Field(..., alias="fileSizeMb")
    warnings: List[str]
    estimated_processing_time_sec: Optional[float] = Field(None, alias="estimatedProcessingTimeSec")


class TiffMetadataResponse(BaseModel):
    """Comprehensive TIFF metadata response"""
    model_config = ConfigDict(populate_by_name=True)
    
    width: int
    height: int
    band_count: int = Field(..., alias="bandCount")
    epsg_code: Optional[int] = Field(None, alias="epsgCode")
    crs_wkt: Optional[str] = Field(None, alias="crsWkt")
    bounds: TiffBoundsInfo
    bounds_wgs84: Optional[TiffBoundsInfo] = Field(None, alias="boundsWgs84")
    pixel_size: TiffPixelSize = Field(..., alias="pixelSize")
    compression: Optional[str] = None
    photometric: Optional[str] = None
    bands: List[TiffBandInfo]
    tags: Dict[str, Any] = Field(default_factory=dict)
    file_size_bytes: int = Field(..., alias="fileSizeBytes")
    file_size_mb: float = Field(..., alias="fileSizeMb")
    warnings: List[str]
    capture_date: Optional[str] = Field(None, alias="captureDate")
    sensor_hints: Dict[str, str] = Field(default_factory=dict, alias="sensorHints")


class TiffValidationRequest(BaseModel):
    """Request to validate TIFF before upload"""
    model_config = ConfigDict(populate_by_name=True)
    
    epsg_code: Optional[int] = Field(None, alias="epsgCode")
    require_geotransform: bool = Field(True, alias="requireGeotransform")


# Segmentation and Pixel Coverage Analysis Schemas

class PixelCoverageItem(BaseModel):
    """Cobertura de una clase en píxeles y área en metros cuadrados"""
    model_config = ConfigDict(populate_by_name=True)
    
    class_id: int = Field(..., alias="classId")
    class_name: str = Field(..., alias="className")
    pixel_count: int = Field(..., alias="pixelCount")
    coverage_percentage: float = Field(..., alias="coveragePercentage")
    area_m2: float | None = Field(None, alias="areaM2")


class SegmentationResponseDTO(BaseModel):
    """Respuesta de predicción con análisis de cobertura por píxeles y área en m²"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "scene_id": "550e8400-e29b-41d4-a716-446655440000",
                "total_pixels": 262144,
                "total_area_m2": 262144.0,
                "pixel_area_m2": 1.0,
                "coverage_by_class": [
                    {
                        "class_id": 0,
                        "class_name": "Vegetación",
                        "pixel_count": 131072,
                        "coverage_percentage": 50.0,
                        "area_m2": 131072.0
                    },
                    {
                        "class_id": 1,
                        "class_name": "Agua",
                        "pixel_count": 65536,
                        "coverage_percentage": 25.0,
                        "area_m2": 65536.0
                    }
                ]
            }
        }
    )
    
    scene_id: str = Field(..., alias="sceneId")
    total_pixels: int = Field(..., alias="totalPixels")
    total_area_m2: float | None = Field(None, alias="totalAreaM2")
    pixel_area_m2: float | None = Field(None, alias="pixelAreaM2")
    coverage_by_class: List[PixelCoverageItem] = Field(..., alias="coverageByClass")


class SegmentationCoverageRead(BaseModel):
    """Lectura de cobertura guardada en BD"""
    model_config = ConfigDict(populate_by_name=True)
    
    scene_id: str = Field(..., alias="sceneId")
    total_pixels: int = Field(..., alias="totalPixels")
    total_area_m2: float | None = Field(None, alias="totalAreaM2")
    pixel_area_m2: float | None = Field(None, alias="pixelAreaM2")
    image_resolution: str = Field(..., alias="imageResolution")
    coverage_by_class: List[PixelCoverageItem] = Field(..., alias="coverageByClass")
    created_at: datetime = Field(..., alias="createdAt")


class SegmentationCoverageSummary(BaseModel):
    """Resumen rápido de cobertura dominante"""
    scene_id: str
    dominant_class: str
    dominant_percentage: float
    secondary_class: Optional[str] = None
    secondary_percentage: Optional[float] = None
