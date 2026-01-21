from __future__ import annotations

import base64
from pathlib import Path
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
import rasterio
from rasterio.io import MemoryFile
import numpy as np
from PIL import Image

from app.core.db import get_db
from app.core.config import get_settings
from app.models import Scene

router = APIRouter(prefix="/scenes", tags=["scenes"])
settings = get_settings()


@router.get("/{scene_id}/original-image")
async def get_original_image(
    scene_id: str,
    db: Session = Depends(get_db),
):
    """
    Sirve la imagen original (scene.tif) convertida a PNG con transparencia.
    Los píxeles negros se hacen transparentes para que se vea el fondo del mapa.
    """
    scene = db.get(Scene, scene_id)
    if not scene or not scene.raster_path:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    
    raster_path = Path(scene.raster_path)
    if not raster_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene file not found: {raster_path}")
    
    try:
        with rasterio.open(raster_path) as src:
            # Leer datos de la imagen
            data = src.read()
            
            # Convertir a formato que PIL pueda leer (RGB)
            if len(data.shape) == 3:
                # Multibanda
                if data.shape[0] >= 3:
                    # RGB: usar primeras 3 bandas
                    rgb_data = np.dstack([data[0], data[1], data[2]])
                elif data.shape[0] == 1:
                    # Monoband: repetir en 3 canales
                    band = data[0]
                    rgb_data = np.dstack([band, band, band])
                else:
                    rgb_data = np.dstack([data[i] for i in range(data.shape[0])])
            else:
                # Datos simples 2D
                band = data
                rgb_data = np.dstack([band, band, band])
            
            # Normalizar a 0-255 si es necesario
            if rgb_data.max() > 255:
                rgb_data = ((rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min()) * 255).astype(np.uint8)
            else:
                rgb_data = rgb_data.astype(np.uint8)
            
            # Crear canal alfa: los píxeles negros (R,G,B < 10) serán transparentes
            # Calcular luminancia: si es muy oscuro, hacer transparente
            luminance = (rgb_data[:,:,0].astype(float) * 0.299 + 
                        rgb_data[:,:,1].astype(float) * 0.587 + 
                        rgb_data[:,:,2].astype(float) * 0.114)
            
            # Píxeles muy oscuros = transparentes (alpha = 0)
            # Píxeles claros = opacos (alpha = 255)
            alpha = np.where(luminance < 20, 0, 255).astype(np.uint8)
            
            # Crear imagen RGBA
            rgba_data = np.dstack([rgb_data, alpha])
            img = Image.fromarray(rgba_data, mode='RGBA')
            
            # Guardar en memoria como PNG
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return StreamingResponse(
                iter([img_buffer.getvalue()]),
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename=scene_{scene_id}.png"}
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error converting image: {str(e)}"
        )


@router.get("/{scene_id}/image-info")
async def get_image_info(
    scene_id: str,
    db: Session = Depends(get_db),
):
    """
    Retorna información de georreferenciación de la imagen original.
    Incluye bounds y CRS.
    """
    scene = db.get(Scene, scene_id)
    if not scene or not scene.raster_path:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    
    raster_path = Path(scene.raster_path)
    if not raster_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene file not found: {raster_path}")
    
    try:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds  # (left, bottom, right, top)
            crs = src.crs.to_epsg() if src.crs else 4326
            
            return {
                "sceneId": scene_id,
                "bounds": {
                    "minX": bounds.left,
                    "minY": bounds.bottom,
                    "maxX": bounds.right,
                    "maxY": bounds.top
                },
                "crs": crs,
                "width": src.width,
                "height": src.height,
                "count": src.count,  # número de bandas
                "dtype": src.dtypes[0] if src.dtypes else "uint8"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading image metadata: {str(e)}"
        )


@router.get("/{scene_id}/original-image-base64")
async def get_original_image_base64(
    scene_id: str,
    db: Session = Depends(get_db),
):
    """
    Retorna la imagen original codificada como base64.
    Útil para mostrar directamente en mapas web.
    """
    scene = db.get(Scene, scene_id)
    if not scene or not scene.raster_path:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    
    raster_path = Path(scene.raster_path)
    if not raster_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene file not found: {raster_path}")
    
    try:
        with open(raster_path, 'rb') as f:
            image_data = f.read()
            base64_str = base64.b64encode(image_data).decode('utf-8')
            
            return {
                "sceneId": scene_id,
                "image": f"data:image/tiff;base64,{base64_str}",
                "mimeType": "image/tiff"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading image: {str(e)}"
        )
