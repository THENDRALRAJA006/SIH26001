FROM python:3.11-slim

LABEL maintainer="ZAIX Team"
LABEL project="LAND-JEPA"

# System dependencies (GDAL for geospatial processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

EXPOSE 8000

# Default: run FastAPI dev server
# Override command in docker-compose for worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
