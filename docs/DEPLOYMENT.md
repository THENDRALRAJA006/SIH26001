# LAND-JEPA — Deployment Architecture

## Overview

LAND-JEPA uses Docker and Docker Compose for containerized deployment.
Development and demo environments use a single `docker-compose.yml`.
Production deployment targets are documented but not deployed during the hackathon.

---

## Development / Demo Environment

### Services

```
docker-compose.yml
├── db          PostgreSQL 15 + PostGIS 3
├── backend     FastAPI + Uvicorn
├── frontend    Vite dev server (or Nginx for built assets)
└── worker      APScheduler / Celery worker (optional)
```

### Startup Order

```
db (health check: pg_isready)
  └── backend (depends_on: db)
       └── frontend (depends_on: backend)
            └── worker (depends_on: backend)
```

### Port Mapping

| Service    | Internal port | External port |
|------------|---------------|---------------|
| db         | 5432          | 5432          |
| backend    | 8000          | 8000          |
| frontend   | 3000          | 3000          |
| worker     | —             | —             |

---

## Environment Configuration

### .env.example

```bash
# ── Database ──────────────────────────────────────────
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=landjepa
POSTGRES_USER=landjepa_user
POSTGRES_PASSWORD=CHANGE_ME

# ── Backend ───────────────────────────────────────────
SECRET_KEY=CHANGE_ME_TO_RANDOM_64_CHAR_STRING
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
ALLOWED_ORIGINS=http://localhost:3000

# ── Demo Mode ─────────────────────────────────────────
DEMO_MODE=true

# ── ML ────────────────────────────────────────────────
ML_DEVICE=cpu          # 'cpu' or 'cuda'
ML_CHECKPOINTS_DIR=ml/checkpoints

# ── Data ingestion ────────────────────────────────────
RAINFALL_PROVIDER=demo   # 'demo' | 'open_meteo' | 'imد' | ...
WEATHER_PROVIDER=open_meteo
SOIL_MOISTURE_PROVIDER=demo
INSAR_ENABLED=false

# ── Alerts ────────────────────────────────────────────
ALERT_DEMO_ONLY=true     # Never send real alerts when true

# ── Scheduling ────────────────────────────────────────
RAINFALL_REFRESH_INTERVAL_MINUTES=60
WEATHER_REFRESH_INTERVAL_MINUTES=60
SOIL_MOISTURE_REFRESH_INTERVAL_MINUTES=1440
RISK_RECOMPUTE_INTERVAL_MINUTES=60

# ── Uploads ───────────────────────────────────────────
UPLOAD_DIR=data/uploads
MAX_PHOTO_SIZE_MB=10
MAX_VIDEO_SIZE_MB=50
```

---

## docker-compose.yml Structure

```yaml
version: "3.9"

services:

  db:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: ../docker/backend/Dockerfile
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./ml/checkpoints:/app/ml/checkpoints
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: ../docker/frontend/Dockerfile
    env_file: .env
    ports:
      - "3000:3000"
    depends_on:
      - backend

  worker:
    build:
      context: ./backend
      dockerfile: ../docker/backend/Dockerfile
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./ml/checkpoints:/app/ml/checkpoints
    depends_on:
      db:
        condition: service_healthy
    command: python -m app.worker
    profiles:
      - worker

volumes:
  pgdata:
```

---

## Dockerfiles

### Backend Dockerfile (docker/backend/Dockerfile)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
```

### Frontend Dockerfile (docker/frontend/Dockerfile)

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

---

## Database Initialization

On first startup, `docker/db/init.sql` runs:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

Migrations are then applied via:
```bash
docker compose exec backend alembic upgrade head
```

Demo seed data:
```bash
docker compose exec backend python scripts/seed_demo.py
```

---

## Production Deployment Notes

The following production configurations are documented but NOT implemented during hackathon:

| Concern                  | Recommended solution                              |
|--------------------------|---------------------------------------------------|
| PostgreSQL HA            | AWS RDS Multi-AZ or Cloud SQL HA                  |
| Application scaling      | Kubernetes Deployment (min 2 replicas)            |
| File storage             | AWS S3 / GCP GCS (not local disk)                 |
| Secrets                  | AWS Secrets Manager / GCP Secret Manager          |
| TLS termination          | Nginx / Load balancer with ACM/Let's Encrypt      |
| Token denylist           | Redis                                             |
| Background tasks         | Celery + Redis (not in-process APScheduler)       |
| GIS tile serving         | pg_tileserv or Martin                             |
| Log aggregation          | ELK Stack / Cloud Logging                         |
| Monitoring               | Prometheus + Grafana                              |
| CI/CD                    | GitHub Actions                                    |

---

## Quick Start Commands

```bash
# Clone and configure
git clone <repo>
cd LAND-JEPA
cp .env.example .env
# Edit .env as needed

# Start services
docker compose up -d db
docker compose up -d backend
docker compose up -d frontend

# Apply migrations
docker compose exec backend alembic upgrade head

# Seed demo data
docker compose exec backend python scripts/seed_demo.py

# Run tests
docker compose exec backend pytest tests/
docker compose exec frontend npm run test
```
