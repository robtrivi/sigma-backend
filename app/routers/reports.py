from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.schemas import ReportDownloadRequest, ReportJobResponse
from app.services.reports_service import ReportsService

router = APIRouter(prefix="/reports", tags=["reports"])
reports_service = ReportsService()


@router.post("/download", response_model=ReportJobResponse)
def trigger_report(
    payload: ReportDownloadRequest,
    db: Session = Depends(get_db),
) -> ReportJobResponse:
    try:
        return reports_service.generate_report(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{report_id}/download")
def download_report(report_id: str, db: Session = Depends(get_db)):
    try:
        path = reports_service.get_report_file(db, report_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name)
