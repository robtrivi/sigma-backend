# SIGMA Backend

Backend geoespacial para el proyecto SIGMA, construido con FastAPI, SQLAlchemy 2 y PostGIS. Expone endpoints listos para ser consumidos por Angular 20 + Leaflet utilizando GeoJSON en EPSG:4326.

## Requisitos

- Python 3.11+
- [uv](https://astral.sh/uv) instalado (gestor de entornos y dependencias)
- PostgreSQL 14+ con extensión PostGIS
- Dependencias de sistema para gdal/rasterio (libgdal, libspatialindex, etc.) instaladas en el sistema operativo

## Configuración de base de datos

```sql
CREATE DATABASE sigma;
\c sigma
CREATE EXTENSION IF NOT EXISTS postgis;
```

Crea un usuario con permisos sobre el esquema si es necesario y actualiza `DATABASE_URL` en `.env`.

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
DATABASE_URL=postgresql+psycopg://sigma:sigma@localhost:5432/sigma
API_PREFIX=/api/v1
GREEN_CLASS_IDS=green,tree_canopy,park
```

`GREEN_CLASS_IDS` permite ajustar qué clases cuentan como cobertura verde para los reportes.

## Instalación (con uv)

1. Clonar el repositorio:

   ```bash
   git clone https://ruta/a/sigma-backend.git
   cd sigma-backend
   ```

2. Sincronizar dependencias y entorno virtual con `uv`:

   ```bash
   uv sync
   ```

   Esto creará automáticamente:

   - El entorno virtual `.venv/`
   - El archivo de bloqueo `uv.lock`
   - Todas las dependencias declaradas en `pyproject.toml`

Opcionalmente, si deseas trabajar con el entorno activado de forma clásica:

```bash
source .venv/bin/activate      # Linux / macOS
# o .venv\Scripts\activate     # Windows
```

## Migraciones (Alembic)

1. Asegúrate de tener configurado `.env` con `DATABASE_URL` apuntando a la base de datos SIGMA.

2. Ejecutar migraciones:

   ```bash
   uv run alembic upgrade head
   ```

3. Para crear nuevas migraciones:

   ```bash
   uv run alembic revision -m "descripcion" --autogenerate
   ```

`uv run` garantiza que el comando se ejecute con el entorno virtual y dependencias correctas.

## Ejecución del servidor

Para levantar el servidor de desarrollo:

```bash
uv run uvicorn app.main:app --reload
```

Por defecto expone la API en `http://localhost:8000`.  
La documentación interactiva está disponible en `http://localhost:8000/docs`.

Si necesitas que uvicorn cargue variables desde `.env` directamente:

```bash
uv run uvicorn app.main:app --reload --env-file .env
```

## Estructura principal

```text
app/
  core/          # Configuración de app y sesión de BD
  models/        # ORM SQLAlchemy + geoalchemy2
  schemas/       # Modelos Pydantic para requests/responses
  services/      # Lógica de negocio (segmentos, agregaciones, reportes, segmentación raster)
  routers/       # Rutas FastAPI agrupadas por dominio
  utils/         # Utilidades GIS (GeoJSON, reproyecciones)
alembic/         # Configuración y migraciones
pyproject.toml   # Configuración del proyecto y dependencias gestionadas por uv
```

## Resumen de endpoints

- `POST /api/v1/imports/scenes`  
  Ingesta de escenas satelitales (multipart + subida de TIFF/JP2).

- `POST /api/v1/imports/segments`  
  Ingesta masiva de segmentos vía GeoJSON FeatureCollection.

- `PUT /api/v1/segments/{segmentId}`  
  Edición de clase/confianza/notas de un segmento.

- `GET /api/v1/segments/tiles`  
  GeoJSON filtrado por región, periodo, clases y bbox para Leaflet.

- `POST /api/v1/segments/scenes/{sceneId}/segment`  
  Segmentación automática (KMeans) desde un raster.

- `POST /api/v1/aggregations/rebuild`  
  Recalcula estadísticas agregadas de un periodo/región.

- `GET /api/v1/catalogs/classes`  
  Catálogo de clases disponibles para filtros del frontend.

- `GET /api/v1/regions/{regionId}/periods`  
  Lista de periodos disponibles con recuentos.

- `GET /api/v1/regions/{regionId}/summary`  
  KPIs y distribución por clase para DashboardPanel.

- `GET /api/v1/subregions/{subregionId}/history`  
  Serie temporal por subregión.

- `POST /api/v1/reports/download`  
  Genera reporte CSV con métricas agregadas y detalle.

- `GET /api/v1/reports/{reportId}/download`  
  Descarga del reporte generado.

## Flujo típico

1. Registrar regiones y catálogo básico vía migraciones o scripts.
2. Ingresar escenas (`/imports/scenes`) y segmentos (`/imports/segments`).
3. Ejecutar `POST /aggregations/rebuild` para actualizar KPIs del periodo.
4. Consumir `/segments/tiles`, `/regions/{id}/summary` y `/regions/{id}/periods` desde Angular/Leaflet.
5. Descargar reportes o generar segmentaciones automáticas según sea necesario.

## Comandos útiles (resumen)

- Instalar dependencias y preparar el entorno:

  ```bash
  uv sync
  ```

- Levantar el API:

  ```bash
  uv run uvicorn app.main:app --reload
  ```

- Aplicar migraciones:

  ```bash
  uv run alembic upgrade head
  ```

- Crear una nueva migración:

  ```bash
  uv run alembic revision -m "mensaje" --autogenerate
  ```

- Ejecutar Ruff:

  ```bash
  uv run ruff check app
  ```

- Formatear con Black:

  ```bash
  uv run black app
  ```

Mantén la carpeta `data/` accesible para guardar escenas y reportes generados si tu implementación la utiliza.