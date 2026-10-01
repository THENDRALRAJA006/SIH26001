# =====================================================================
# LAND-JEPA Backend Production Dockerfile for Render & Container Cloud
# Multi-stage Python 3.11 image for FastAPI early warning & risk engine
# =====================================================================

# ── 1. Build Stage (compile wheels, install C-extensions) ────────────
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

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Pre-install CPU-only PyTorch and torchvision to avoid 2GB CUDA bloat on CPU instances
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt


# ── 2. Production Runner Stage (minimal runtime footprint) ───────────
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system libraries:
# - libpq5: PostgreSQL runtime client
# - libgomp1: GNU OpenMP for PyTorch / XGBoost / scikit-learn
# - curl: Container healthcheck probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and entrypoints from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/backend \
    PORT=8000

# Copy application source code and models
COPY backend/ /app/backend/
COPY ml/ /app/ml/
COPY gis/ /app/gis/
COPY data/ /app/data/
COPY yolov8n.pt /app/yolov8n.pt

# Ensure persistent directories exist and configure non-root user for security
RUN mkdir -p /app/results /app/data/uploads /app/ml/checkpoints \
    && groupadd -r landjepa && useradd -r -g landjepa -d /app -s /sbin/nologin landjepa \
    && chown -R landjepa:landjepa /app

USER landjepa

WORKDIR /app

EXPOSE 8000

# Health check probing the dynamic port
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start uvicorn with shell parameter expansion to bind cleanly to Render's $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
