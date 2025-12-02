from __future__ import annotations

import uuid
from datetime import date, datetime

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import JSON, Column, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class Region(Base):
    __tablename__ = "regions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    geometry: Mapped[WKBElement | None] = mapped_column(Geometry("MULTIPOLYGON", 4326, spatial_index=True), nullable=True)

    scenes: Mapped[list["Scene"]] = relationship(back_populates="region", cascade="all,delete")
    segments: Mapped[list["Segment"]] = relationship(back_populates="region")
    subregions: Mapped[list["Subregion"]] = relationship(back_populates="region")


class Scene(Base, TimestampMixin):
    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id: Mapped[str] = mapped_column(ForeignKey("regions.id", ondelete="RESTRICT"), nullable=False)
    capture_date: Mapped[date] = mapped_column(Date, nullable=False)
    epsg: Mapped[int] = mapped_column(Integer, nullable=False)
    sensor: Mapped[str] = mapped_column(String(128), nullable=False)
    raster_path: Mapped[str] = mapped_column(String(512), nullable=False)

    region: Mapped[Region] = relationship(back_populates="scenes")
    segments: Mapped[list["Segment"]] = relationship(back_populates="scene")


class ClassCatalog(Base):
    __tablename__ = "class_catalog"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    color_hex: Mapped[str] = mapped_column(String(7), nullable=False)
    icono_primeng: Mapped[str] = mapped_column(String(128), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    segments: Mapped[list["Segment"]] = relationship(back_populates="class_catalog")


class Segment(Base, TimestampMixin):
    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False)
    region_id: Mapped[str] = mapped_column(ForeignKey("regions.id", ondelete="RESTRICT"), nullable=False)
    class_id: Mapped[str] = mapped_column(ForeignKey("class_catalog.id", ondelete="RESTRICT"), nullable=False)
    class_name: Mapped[str] = mapped_column(String(255), nullable=False)
    periodo: Mapped[str] = mapped_column(String(7), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    area_m2: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="import")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    geometry: Mapped[WKBElement] = mapped_column(Geometry("MULTIPOLYGON", 4326, spatial_index=True), nullable=False)
    is_manual_edited: Mapped[bool] = mapped_column(default=False)

    scene: Mapped[Scene] = relationship(back_populates="segments")
    region: Mapped[Region] = relationship(back_populates="segments")
    class_catalog: Mapped[ClassCatalog] = relationship(back_populates="segments")

    __table_args__ = (
        Index("idx_segments_region_period_class", "region_id", "periodo", "class_id"),
    )


class AggregationSummary(Base, TimestampMixin):
    __tablename__ = "aggregation_summary"
    __table_args__ = (
        UniqueConstraint("region_id", "periodo", name="uq_region_periodo"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id: Mapped[str] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"), nullable=False)
    periodo: Mapped[str] = mapped_column(String(7), nullable=False)
    total_area_m2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    green_coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    distribution_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    trend_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    region: Mapped[Region] = relationship()


class Subregion(Base, TimestampMixin):
    __tablename__ = "subregions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id: Mapped[str] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    geometry: Mapped[WKBElement] = mapped_column(Geometry("MULTIPOLYGON", 4326, spatial_index=True), nullable=False)

    region: Mapped[Region] = relationship(back_populates="subregions")


class ReportRequest(Base, TimestampMixin):
    __tablename__ = "report_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id: Mapped[str] = mapped_column(String(64), nullable=False)
    periodos: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    class_filters: Mapped[list[str]] = mapped_column(JSON, nullable=True)
    segment_ids: Mapped[list[str]] = mapped_column(JSON, nullable=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
