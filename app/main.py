from __future__ import annotations

import logging
import os
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ===== CONFIGURACIÓN GDAL/PROJ =====
# Configurar GDAL para evitar warnings de PROJ
os.environ['PROJ_SKIP_NETWORK'] = 'YES'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'YES'

# Supprimir warnings no críticos de rasterio
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
logging.getLogger('rasterio._env').setLevel(logging.ERROR)
logging.getLogger('rasterio._gdal').setLevel(logging.ERROR)
# ===================================

from app.core.config import get_settings
from app.routers import aggregations, catalogs, imports, regions, reports, scenes, segments, subregions, tiff_validation

settings = get_settings()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.project_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200","https://67fg29kg-4200.use2.devtunnels.ms"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(imports.router, prefix=settings.api_prefix)
app.include_router(segments.router, prefix=settings.api_prefix)
app.include_router(scenes.router, prefix=settings.api_prefix)
app.include_router(regions.router, prefix=settings.api_prefix)
app.include_router(catalogs.router, prefix=settings.api_prefix)
app.include_router(reports.router, prefix=settings.api_prefix)
app.include_router(subregions.router, prefix=settings.api_prefix)
app.include_router(aggregations.router, prefix=settings.api_prefix)
app.include_router(tiff_validation.router, prefix=settings.api_prefix)


@app.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
