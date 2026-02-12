# SIGMA Backend

Backend geoespacial de SIGMA construido con FastAPI, SQLAlchemy 2 y PostGIS.

## Documentacion

- Guia de despliegue en VPS: [`DEPLOYMENT.md`](DEPLOYMENT.md)

## Objetivo del servicio

El backend expone APIs para:

- ingesta de escenas raster y segmentos
- consulta geoespacial para mapa (GeoJSON)
- agregaciones por region/periodo
- generacion de reportes CSV
- segmentacion automatica sobre raster

## Stack tecnico

- Python 3.11+
- FastAPI + Uvicorn
- SQLAlchemy 2 + Alembic
- PostgreSQL 14+ + PostGIS
- Rasterio / GDAL / PyProj / Shapely
- TensorFlow (segmentacion)

## Requisitos de sistema

Antes de instalar dependencias Python, asegure estas librerias del sistema (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y \
  gdal-bin libgdal-dev \
  proj-bin proj-data libproj-dev \
  libgeos-dev libspatialindex-dev \
  libgl1 libglib2.0-0
```

## Configuracion de base de datos

Crear base y extension:

```sql
CREATE DATABASE sigma;
\c sigma
CREATE EXTENSION IF NOT EXISTS postgis;
```

## Variables de entorno

Crear un archivo `.env` en la raiz del repositorio.

Ejemplo minimo:

```env
DATABASE_URL=postgresql+psycopg://sigma:sigma@localhost:5432/sigma
API_PREFIX=/api/v1
GREEN_CLASS_IDS=green,tree_canopy,park
SEGMENTATION_MODEL_PATH=model.keras
```

Variables relevantes:

- `DATABASE_URL` (obligatoria): conexion SQLAlchemy a PostgreSQL.
- `API_PREFIX` (opcional, default `/api/v1`): prefijo global de rutas.
- `GREEN_CLASS_IDS` (opcional): ids de clases consideradas cobertura verde.
- `SEGMENTATION_MODEL_PATH` (opcional, default `model.keras`): ruta del modelo de segmentacion.
- `DATA_DIR` (opcional, default `data`): base para `scenes/` y `reports/`.

## Instalacion para desarrollo local

1. Clonar repositorio.

```bash
git clone <URL_DEL_REPOSITORIO>
cd sigma-backend
```

2. Instalar `uv` (si no esta instalado).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Sincronizar entorno y dependencias.

```bash
uv sync
```

Esto crea `.venv` automaticamente con dependencias del proyecto.

## Migraciones

Aplicar migraciones:

```bash
uv run alembic upgrade head
```

Crear una nueva migracion:

```bash
uv run alembic revision -m "descripcion" --autogenerate
```

## Ejecucion en desarrollo

Levantar API:

```bash
uv run uvicorn app.main:app --reload --env-file .env
```

Endpoints de verificacion:

- Health: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`

## CORS en desarrollo y produccion

Estado actual del codigo:

- Los origenes CORS estan definidos de forma explicita en `app/main.py`.

Implicacion:

- Si tu frontend corre en otro dominio/puerto, debes actualizar la lista `allow_origins` en `app/main.py`.

## Flujo recomendado de desarrollo

1. Levantar PostgreSQL + PostGIS local.
2. Configurar `.env`.
3. Ejecutar `uv sync`.
4. Ejecutar `uv run alembic upgrade head`.
5. Levantar API con `uvicorn`.
6. Validar `/health` y `/docs`.

## Calidad de codigo

Lint:

```bash
uv run ruff check app
```

Formato:

```bash
uv run black app
```

## Estructura del proyecto

```text
app/
  core/          # Configuracion, settings y sesion de BD
  models/        # Modelos ORM
  schemas/       # Esquemas Pydantic
  services/      # Logica de negocio
  routers/       # Endpoints FastAPI
  utils/         # Utilidades GIS y raster
alembic/         # Migraciones de BD
pyproject.toml   # Dependencias y metadatos
```

## Endpoints principales

- `POST /api/v1/imports/scenes`
- `POST /api/v1/imports/segments`
- `GET /api/v1/segments/tiles`
- `POST /api/v1/aggregations/rebuild`
- `GET /api/v1/regions/{regionId}/summary`
- `POST /api/v1/reports/download`

## Notas operativas

- El sistema crea y usa directorios para datos (`scenes` y `reports`) bajo `DATA_DIR`.
- Para despliegue productivo, usar la guia detallada en [`DEPLOYMENT.md`](DEPLOYMENT.md).
