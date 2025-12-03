from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import ReportRequest, Segment
from app.schemas.schemas import ReportDownloadRequest, ReportJobResponse
from app.services.aggregations_service import AggregationsService

logger = logging.getLogger(__name__)


class ReportsService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.aggregations = AggregationsService(self.settings)

    def generate_report(
        self, db: Session, request: ReportDownloadRequest
    ) -> ReportJobResponse:
        segments = self._fetch_segments(db, request)
        if not segments:
            raise ValueError("No hay segmentos para los filtros seleccionados")
        filename = f"{uuid.uuid4()}.csv"
        file_path = Path(self.settings.reports_dir) / filename
        summaries = self._collect_summaries(db, request)
        self._write_csv(file_path, segments, summaries)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        record = ReportRequest(
            region_id=request.regionId,
            periodos=request.periodos,
            class_filters=request.classFilters,
            segment_ids=request.segments,
            file_path=str(file_path),
            expires_at=expires_at,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        download_url = f"{self.settings.api_prefix}/reports/{record.id}/download"
        return ReportJobResponse(
            reportId=str(record.id), downloadUrl=download_url, expiresAt=expires_at
        )

    def get_report_file(self, db: Session, report_id: str) -> Path:
        try:
            report_uuid = uuid.UUID(report_id)
        except ValueError as exc:
            raise ValueError("reportId inválido") from exc
        report = db.get(ReportRequest, report_uuid)
        if not report:
            raise ValueError("Reporte no encontrado")
        path = Path(report.file_path)
        if not path.exists():
            raise ValueError("Archivo de reporte no disponible")
        if datetime.utcnow() > report.expires_at:
            raise ValueError("El enlace del reporte expiró")
        return path

    def _fetch_segments(
        self, db: Session, request: ReportDownloadRequest
    ) -> List[Segment]:
        stmt: Select[tuple[Segment]] = select(Segment).where(
            Segment.region_id == request.regionId,
            Segment.periodo.in_(request.periodos),
        )
        if request.classFilters:
            stmt = stmt.where(Segment.class_id.in_(request.classFilters))
        if request.segments:
            parsed_ids = [uuid.UUID(seg_id) for seg_id in request.segments]
            stmt = stmt.where(Segment.id.in_(parsed_ids))
        return db.execute(stmt).scalars().all()

    def _collect_summaries(
        self, db: Session, request: ReportDownloadRequest
    ) -> dict[str, dict]:
        summaries: dict[str, dict] = {}
        for periodo in request.periodos:
            summary = self.aggregations.snapshot_period(db, request.regionId, periodo)
            summaries[periodo] = summary
        return summaries

    def _write_csv(
        self, path: Path, segments: Iterable[Segment], summaries: dict[str, dict]
    ) -> None:
        headers = [
            "periodo",
            "segment_id",
            "region_id",
            "class_id",
            "class_name",
            "area_m2",
            "confidence",
            "source",
            "notes",
        ]
        with path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["SIGMA Reporte de Cobertura"])
            writer.writerow([f"Generado: {datetime.utcnow().isoformat()}Z"])
            writer.writerow([])
            writer.writerow(["Resumen por periodo"])
            writer.writerow(["Periodo", "Cobertura verde (%)", "Área total (m2)"])
            for periodo, summary in summaries.items():
                writer.writerow(
                    [
                        periodo,
                        f"{summary['green_coverage']:.2f}",
                        f"{summary['total_area']:.2f}",
                    ]
                )
            writer.writerow([])
            writer.writerow(headers)
            for segment in segments:
                writer.writerow(
                    [
                        segment.periodo,
                        segment.id,
                        segment.region_id,
                        segment.class_id,
                        segment.class_name,
                        segment.area_m2,
                        segment.confidence,
                        segment.source,
                        segment.notes or "",
                    ]
                )
        logger.info("Reporte generado en %s", path)
