from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class RegionBase(BaseModel):
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
    def ensure_features(cls, v: List[GeoJSONFeature], info: ValidationInfo) -> List[GeoJSONFeature]:
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


class SceneSegmentRequest(BaseModel):
    method: str = "kmeans"
    n_classes: int = Field(ge=2, le=15)
    bands: List[int] | None = None
    downscale_factor: float | None = Field(default=None, gt=0)
