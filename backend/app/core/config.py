"""
LAND-JEPA — Application Settings

Loaded from environment variables (via .env file in development).
All secrets must be environment variables — never hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────
    APP_NAME: str = "LAND-JEPA"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "landjepa"
    POSTGRES_USER: str = "landjepa_user"
    POSTGRES_PASSWORD: str = "changeme"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Auth ──────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE_ME_THIS_IS_NOT_SAFE"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── CORS ──────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # ── Demo / Safety ─────────────────────────────────────────────────
    DEMO_MODE: bool = True
    ALERT_DEMO_ONLY: bool = True  # NEVER send real alerts when True

    # ── ML ────────────────────────────────────────────────────────────
    ML_DEVICE: str = "cpu"
    ML_CHECKPOINTS_DIR: str = "ml/checkpoints"

    # ── Data Providers ────────────────────────────────────────────────
    RAINFALL_PROVIDER: str = "demo"
    WEATHER_PROVIDER: str = "open_meteo"
    SOIL_MOISTURE_PROVIDER: str = "demo"
    TERRAIN_PROVIDER: str = "demo"
    INSAR_ENABLED: bool = False

    # ── Scheduling ────────────────────────────────────────────────────
    RAINFALL_REFRESH_INTERVAL_MINUTES: int = 60
    WEATHER_REFRESH_INTERVAL_MINUTES: int = 60
    SOIL_MOISTURE_REFRESH_INTERVAL_MINUTES: int = 1440
    RISK_RECOMPUTE_INTERVAL_MINUTES: int = 60

    # ── Uploads ───────────────────────────────────────────────────────
    UPLOAD_DIR: str = "data/uploads"
    MAX_PHOTO_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 50

    @field_validator("ML_DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "mps"):
            raise ValueError(f"ML_DEVICE must be 'cpu', 'cuda', or 'mps', got '{v}'")
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
