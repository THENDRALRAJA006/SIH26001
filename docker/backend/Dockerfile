# =====================================================================
# LAND-JEPA Backend Dockerfile
# Multi-stage Python 3.11 image for FastAPI risk & alert service
# =====================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for C-extensions (geospatial, psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# ── Production Runner Stage ───────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system libraries (libpq for postgresql, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/backend \
    PORT=8000

# Copy application code
COPY backend/ /app/backend/
COPY ml/ /app/ml/
COPY gis/ /app/gis/

# Create non-root user for security
RUN groupadd -r landjepa && useradd -r -g landjepa -d /app -s /sbin/nologin landjepa \
    && chown -R landjepa:landjepa /app

USER landjepa

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
