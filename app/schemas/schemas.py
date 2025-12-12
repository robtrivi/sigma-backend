from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class RegionBase(BaseModel):
    id: str
    name: str


class RegionCreate(BaseModel):
    id: str
    name: str


class RegionRead(RegionBase):
    geometry: dict | None = None


class SceneUploadResponse(BaseModel):
    sceneId: str
    regionId: str
    captureDate: date
    epsg: int
    sensor: str
    rasterPath: str


class SegmentImportProperties(BaseModel):
    sceneId: str
    classId: str
    confidence: float
    areaM2: float
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
    inserted: int
    segmentIds: List[str]


class SegmentUpdateRequest(BaseModel):
    classId: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None


class SegmentProperties(BaseModel):
    segmentId: str
    sceneId: str
    regionId: str
    classId: str
    className: str
    areaM2: float
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
    periodo: str
    regionId: str


class DistributionItem(BaseModel):
    classId: str
    className: str
    percentage: float
    areaM2: float


class TrendItem(BaseModel):
    periodo: str
    value: float


class RegionSummaryResponse(BaseModel):
    regionId: str
    periodo: str
    segmentsVisible: int
    coberturaVerde: float
    distribution: List[DistributionItem]
    trend: List[TrendItem]
    messages: List[str]


class AggregationSummaryRead(BaseModel):
    regionId: str
    periodo: str
    totalAreaM2: float
    greenCoverage: float
    distribution: List[DistributionItem]
    trend: List[TrendItem]


class CatalogClassRead(BaseModel):
    classId: str
    nombre: str
    colorHex: str
    iconoPrimeNg: str | None = None
    description: str | None = None


class RegionPeriodItem(BaseModel):
    periodo: str
    regionId: str
    segmentCount: int
    lastUpdated: datetime | None = None


class SubregionHistoryItem(BaseModel):
    periodo: str
    areaM2: float
    dominantClassId: str
    dominantClassName: str
    deltaVsPrev: float | None = None


class SubregionHistoryResponse(BaseModel):
    subregionId: str
    regionId: str
    history: List[SubregionHistoryItem]


class ReportDownloadRequest(BaseModel):
    regionId: str
    periodos: List[str]
    classFilters: List[str] | None = None
    segments: List[str] | None = None


class ReportJobResponse(BaseModel):
    reportId: str
    downloadUrl: str
    expiresAt: datetime


class ReportDownloadLink(BaseModel):
    reportId: str
    regionId: str
    filters: Dict[str, Any]
    downloadUrl: str
    expiresAt: datetime


class PeriodRangeQuery(BaseModel):
    from_period: str | None = Field(default=None, alias="from")
    to_period: str | None = Field(default=None, alias="to")


class SegmentsTilesQuery(BaseModel):
    regionId: str
    periodo: str
    classId: List[str] | None = None
    bbox: str | None = None


# TIFF Validation and Metadata Schemas

class TiffBandInfo(BaseModel):
    index: int
    dtype: str
    minValue: Optional[float] = None
    maxValue: Optional[float] = None
    nodata: Optional[Any] = None
    colorInterp: Optional[str] = None


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
    valid: bool
    width: int
    height: int
    bandCount: int
    dtype: str
    epsgCode: Optional[int] = None
    crsWkt: Optional[str] = None
    bounds: TiffBoundsInfo
    boundsWgs84: Optional[TiffBoundsInfo] = None
    pixelSize: TiffPixelSize
    compression: Optional[str] = None
    photometric: Optional[str] = None
    bands: List[TiffBandInfo]
    fileSizeMb: float
    warnings: List[str]
    estimatedProcessingTimeSec: Optional[float] = None


class TiffMetadataResponse(BaseModel):
    """Comprehensive TIFF metadata response"""
    width: int
    height: int
    bandCount: int
    epsgCode: Optional[int] = None
    crsWkt: Optional[str] = None
    bounds: TiffBoundsInfo
    boundsWgs84: Optional[TiffBoundsInfo] = None
    pixelSize: TiffPixelSize
    compression: Optional[str] = None
    photometric: Optional[str] = None
    bands: List[TiffBandInfo]
    tags: Dict[str, Any] = Field(default_factory=dict)
    fileSizeBytes: int
    fileSizeMb: float
    warnings: List[str]
    captureDate: Optional[str] = None
    sensorHints: Dict[str, str] = Field(default_factory=dict)


class TiffValidationRequest(BaseModel):
    """Request to validate TIFF before upload"""
    epsgCode: Optional[int] = None
    requireGeotransform: bool = True


# Segmentation and Pixel Coverage Analysis Schemas

class PixelCoverageItem(BaseModel):
    """Cobertura de una clase en píxeles"""
    class_id: int
    class_name: str
    pixel_count: int
    coverage_percentage: float


class SegmentationResponseDTO(BaseModel):
    """Respuesta de predicción con análisis de cobertura por píxeles"""
    scene_id: str
    total_pixels: int
    coverage_by_class: List[PixelCoverageItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "scene_id": "550e8400-e29b-41d4-a716-446655440000",
                "total_pixels": 262144,
                "coverage_by_class": [
                    {
                        "class_id": 0,
                        "class_name": "Vegetación",
                        "pixel_count": 131072,
                        "coverage_percentage": 50.0
                    },
                    {
                        "class_id": 1,
                        "class_name": "Agua",
                        "pixel_count": 65536,
                        "coverage_percentage": 25.0
                    }
                ]
            }
        }


class SegmentationCoverageRead(BaseModel):
    """Lectura de cobertura guardada en BD"""
    scene_id: str
    total_pixels: int
    image_resolution: str
    coverage_by_class: List[PixelCoverageItem]
    created_at: datetime


class SegmentationCoverageSummary(BaseModel):
    """Resumen rápido de cobertura dominante"""
    scene_id: str
    dominant_class: str
    dominant_percentage: float
    secondary_class: Optional[str] = None
    secondary_percentage: Optional[float] = None
