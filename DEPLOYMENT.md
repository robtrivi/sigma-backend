# Guia de Despliegue en VPS - SIGMA Backend

Documento tecnico de despliegue para estudiantes de ingenieria y equipos de operaciones.

## Indice

1. [Objetivo y alcance](#1-objetivo-y-alcance)
2. [Arquitectura objetivo](#2-arquitectura-objetivo)
3. [Prerequisitos](#3-prerequisitos)
4. [Provisionamiento base del servidor](#4-provisionamiento-base-del-servidor)
5. [Instalacion de dependencias de sistema](#5-instalacion-de-dependencias-de-sistema)
6. [Base de datos PostgreSQL + PostGIS](#6-base-de-datos-postgresql--postgis)
7. [Despliegue de la aplicacion](#7-despliegue-de-la-aplicacion)
8. [Configuracion de entorno](#8-configuracion-de-entorno)
9. [Migraciones y prueba funcional inicial](#9-migraciones-y-prueba-funcional-inicial)
10. [Servicio systemd](#10-servicio-systemd)
11. [Reverse proxy con Nginx](#11-reverse-proxy-con-nginx)
12. [HTTPS con Let's Encrypt](#12-https-con-lets-encrypt)
13. [Politica CORS (estado actual y recomendacion)](#13-politica-cors-estado-actual-y-recomendacion)
14. [Backups y recuperacion](#14-backups-y-recuperacion)
15. [Monitoreo y operacion](#15-monitoreo-y-operacion)
16. [Actualizaciones y rollback](#16-actualizaciones-y-rollback)
17. [Troubleshooting](#17-troubleshooting)
18. [Checklist de salida a produccion](#18-checklist-de-salida-a-produccion)
19. [Referencias al codigo del repositorio](#19-referencias-al-codigo-del-repositorio)

## 1. Objetivo y alcance

Este documento describe un despliegue de produccion para `sigma-backend` en un VPS Linux.

Resultado esperado:

- API FastAPI ejecutandose con `systemd`
- Nginx como reverse proxy publico
- HTTPS habilitado con certificados validos
- PostgreSQL con extension PostGIS
- Persistencia de datos de escenas y reportes
- Procedimiento operativo de actualizacion, respaldo y recuperacion

## 2. Arquitectura objetivo

Topologia recomendada para una instancia unica:

- Aplicacion: `uvicorn` en `127.0.0.1:8000`
- Proxy: Nginx en `80/443`
- Base de datos: PostgreSQL + PostGIS en `localhost:5432` (o servicio externo)
- Codigo: `/opt/sigma-backend`
- Datos: `/var/lib/sigma-backend/data`

## 3. Prerequisitos

- VPS Ubuntu 22.04 o 24.04
- Acceso SSH con usuario sudo
- Dominio apuntando al VPS (ejemplo: `api.tudominio.com`)
- Python 3.11 o superior

Verificacion minima:

```bash
python3 --version
uname -a
```

## 4. Provisionamiento base del servidor

### 4.1 Actualizacion de paquetes

```bash
sudo apt update && sudo apt upgrade -y
```

### 4.2 Paquetes base

```bash
sudo apt install -y \
  git curl ca-certificates unzip \
  build-essential pkg-config \
  nginx ufw
```

### 4.3 Usuario de despliegue

```bash
sudo adduser --disabled-password --gecos "" sigma
sudo usermod -aG sudo sigma
```

### 4.4 Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

## 5. Instalacion de dependencias de sistema

El proyecto utiliza procesamiento geoespacial y de imagen (`rasterio`, `pyproj`, `shapely`, OpenCV, TensorFlow).

```bash
sudo apt install -y \
  gdal-bin libgdal-dev \
  proj-bin proj-data libproj-dev \
  libgeos-dev libspatialindex-dev \
  libgl1 libglib2.0-0
```

## 6. Base de datos PostgreSQL + PostGIS

### 6.1 Instalacion

```bash
sudo apt install -y postgresql postgresql-contrib postgis
```

### 6.2 Creacion de usuario y base

```bash
sudo -u postgres psql
```

```sql
CREATE USER sigma WITH PASSWORD 'CAMBIA_ESTA_PASSWORD';
CREATE DATABASE sigma OWNER sigma;
\c sigma
CREATE EXTENSION IF NOT EXISTS postgis;
GRANT ALL PRIVILEGES ON DATABASE sigma TO sigma;
```

Validacion:

```bash
sudo -u postgres psql -d sigma -c "SELECT PostGIS_Full_Version();"
```

## 7. Despliegue de la aplicacion

### 7.1 Instalar `uv`

Como usuario `sigma`:

```bash
su - sigma
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 7.2 Clonar repositorio

```bash
sudo mkdir -p /opt/sigma-backend
sudo chown -R sigma:sigma /opt/sigma-backend
cd /opt/sigma-backend
git clone <URL_DEL_REPOSITORIO> .
```

### 7.3 Instalar dependencias Python

```bash
cd /opt/sigma-backend
uv sync
```

## 8. Configuracion de entorno

Crear archivo `.env` en `/opt/sigma-backend/.env`:

```env
DATABASE_URL=postgresql+psycopg://sigma:CAMBIA_ESTA_PASSWORD@localhost:5432/sigma
API_PREFIX=/api/v1
GREEN_CLASS_IDS=green,tree_canopy,park
SEGMENTATION_MODEL_PATH=/var/lib/sigma-backend/model.keras
DATA_DIR=/var/lib/sigma-backend/data
POSTGIS_ENABLED=true
PROJECT_NAME=SIGMA Backend API
DEFAULT_EPSG=4326
```

Crear rutas persistentes:

```bash
sudo mkdir -p /var/lib/sigma-backend/data/scenes
sudo mkdir -p /var/lib/sigma-backend/data/reports
sudo chown -R sigma:sigma /var/lib/sigma-backend
```

## 9. Migraciones y prueba funcional inicial

### 9.1 Aplicar migraciones

```bash
cd /opt/sigma-backend
uv run alembic upgrade head
```

### 9.2 Prueba local de arranque

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

En otra sesion:

```bash
curl -sS http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{"status":"ok"}
```

## 10. Servicio systemd

Crear `/etc/systemd/system/sigma-backend.service`:

```ini
[Unit]
Description=SIGMA Backend FastAPI
After=network.target postgresql.service

[Service]
User=sigma
Group=sigma
WorkingDirectory=/opt/sigma-backend
Environment=PATH=/home/sigma/.local/bin:/opt/sigma-backend/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
EnvironmentFile=/opt/sigma-backend/.env
ExecStart=/home/sigma/.local/bin/uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
TimeoutStopSec=20
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=false
ReadWritePaths=/opt/sigma-backend /var/lib/sigma-backend

[Install]
WantedBy=multi-user.target
```

Activacion:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sigma-backend
sudo systemctl start sigma-backend
sudo systemctl status sigma-backend
```

Logs:

```bash
journalctl -u sigma-backend -f
```

## 11. Reverse proxy con Nginx

Crear `/etc/nginx/sites-available/sigma-backend`:

```nginx
server {
    listen 80;
    server_name api.tudominio.com;

    client_max_body_size 200M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 60s;
    }
}
```

Habilitar y validar:

```bash
sudo ln -s /etc/nginx/sites-available/sigma-backend /etc/nginx/sites-enabled/sigma-backend
sudo nginx -t
sudo systemctl reload nginx
curl -I http://api.tudominio.com/health
```

## 12. HTTPS con Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.tudominio.com
sudo certbot renew --dry-run
```

## 13. Politica CORS (estado actual y recomendacion)

Estado actual del codigo:

- CORS esta hardcodeado en `app/main.py`.
- Origenes permitidos: `http://localhost:4200` y un dominio de devtunnel.

Implicacion de despliegue:

- Si el frontend de produccion usa otro dominio, el navegador bloqueara las llamadas.

Accion recomendada:

1. Parametrizar CORS en configuracion (ejemplo: variable `CORS_ORIGINS`).
2. Usar `allow_origins=settings.cors_origins` en `CORSMiddleware`.
3. Definir valores por ambiente en `.env`.

Mientras no se aplique ese cambio de codigo, actualiza manualmente la lista en `app/main.py` antes de desplegar.

## 14. Backups y recuperacion

### 14.1 Backup de base de datos

Script sugerido `/usr/local/bin/backup_sigma_db.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR=/var/backups/sigma
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

pg_dump "postgresql://sigma:CAMBIA_ESTA_PASSWORD@localhost:5432/sigma" \
  -Fc -f "$BACKUP_DIR/sigma_$TS.dump"

find "$BACKUP_DIR" -type f -name "sigma_*.dump" -mtime +14 -delete
```

### 14.2 Backup de archivos

Respaldar periodicamente:

- `/var/lib/sigma-backend/data/scenes`
- `/var/lib/sigma-backend/data/reports`
- `/var/lib/sigma-backend/model.keras` (si aplica)

### 14.3 Restore de base

```bash
createdb -U postgres sigma_restored
pg_restore -d sigma_restored /var/backups/sigma/sigma_YYYYMMDD_HHMMSS.dump
```

## 15. Monitoreo y operacion

Comandos base:

```bash
sudo systemctl status sigma-backend
journalctl -u sigma-backend -n 200 --no-pager
journalctl -u sigma-backend -f
sudo systemctl status nginx
sudo nginx -t
curl -fsS https://api.tudominio.com/health
```

Indicadores minimos recomendados:

- Disponibilidad del endpoint `/health`
- Reinicios del servicio `sigma-backend`
- Espacio libre en disco
- Crecimiento de base de datos

## 16. Actualizaciones y rollback

### 16.1 Proceso de actualizacion

```bash
su - sigma
cd /opt/sigma-backend
git pull
uv sync
uv run alembic upgrade head
sudo systemctl restart sigma-backend
sudo systemctl status sigma-backend
```

### 16.2 Validacion posterior

```bash
curl -fsS https://api.tudominio.com/health
```

### 16.3 Rollback

- Volver al commit estable anterior.
- Reinstalar dependencias con `uv sync`.
- Restaurar base si hubo cambios incompatibles.
- Reiniciar servicio.

## 17. Troubleshooting

### 17.1 Falla `rasterio`/`gdal`

- Confirmar paquetes del sistema de la seccion 5.
- Reinstalar entorno con `uv sync`.

### 17.2 Falla de conexion a PostgreSQL

- Validar `DATABASE_URL`.
- Revisar `systemctl status postgresql`.
- Confirmar PostGIS con `SELECT PostGIS_Full_Version();`.

### 17.3 Error CORS en frontend

- Revisar dominios permitidos en `app/main.py`.
- Confirmar que el dominio frontend real este incluido.

### 17.4 Error `413 Request Entity Too Large`

- Aumentar `client_max_body_size` en Nginx.

### 17.5 Falla de segmentacion

- Validar `SEGMENTATION_MODEL_PATH`.
- Confirmar permisos de lectura del usuario `sigma`.
- Revisar logs en `journalctl -u sigma-backend -f`.

## 18. Checklist de salida a produccion

- [ ] DNS resuelto a la IP publica del VPS
- [ ] Firewall activo y puertos minimos abiertos
- [ ] PostgreSQL + PostGIS operativo
- [ ] Variables de entorno validadas
- [ ] Migraciones aplicadas
- [ ] Servicio `sigma-backend` habilitado y estable
- [ ] Nginx operando como proxy
- [ ] Certificado TLS activo y renovacion verificada
- [ ] Politica CORS alineada con el frontend real
- [ ] Backups de base y archivos programados
- [ ] Monitoreo de disponibilidad habilitado

## 19. Referencias al codigo del repositorio

- Configuracion de aplicacion: `app/core/config.py`
- Inicializacion API y CORS: `app/main.py`
- Migraciones: `alembic/env.py`
- Dependencias Python: `pyproject.toml`
- Guia de desarrollo base: `README.md`
