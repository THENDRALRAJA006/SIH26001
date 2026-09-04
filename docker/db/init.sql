-- =====================================================================
-- LAND-JEPA: Database Initialization Script
-- SIH26001 — Northeast India Landslide Early Warning System
-- =====================================================================

-- Enable PostGIS and spatial extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Set default timezone to UTC
ALTER DATABASE landjepa SET timezone TO 'UTC';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'LAND-JEPA PostGIS database extensions successfully initialized.';
END $$;
