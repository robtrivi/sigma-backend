from __future__ import annotations

import logging

from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import aggregations, catalogs, imports, regions, reports, segments, subregions

settings = get_settings()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.project_name, version="1.0.0")

app.include_router(imports.router, prefix=settings.api_prefix)
app.include_router(segments.router, prefix=settings.api_prefix)
app.include_router(regions.router, prefix=settings.api_prefix)
app.include_router(catalogs.router, prefix=settings.api_prefix)
app.include_router(reports.router, prefix=settings.api_prefix)
app.include_router(subregions.router, prefix=settings.api_prefix)
app.include_router(aggregations.router, prefix=settings.api_prefix)


@app.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
