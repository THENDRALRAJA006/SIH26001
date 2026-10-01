"""
LAND-JEPA — Application Settings

Loaded from environment variables (via .env file in development).
All secrets must be environment variables — never hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
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
    DATABASE_URL_DIRECT: Optional[str] = Field(default=None, validation_alias="DATABASE_URL")

    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_URL_DIRECT:
            url = self.DATABASE_URL_DIRECT
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Synchronous URL for Alembic migrations."""
        if self.DATABASE_URL_DIRECT:
            url = self.DATABASE_URL_DIRECT
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+psycopg2://", 1)
            elif url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+psycopg2://"):
                url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
            return url
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

    # ── External API Keys ─────────────────────────────────────────────
    OPEN_METEO_API_KEY: str = ""
    OPENWEATHER_API_KEY: str = ""
    WEATHER_PRIMARY_PROVIDER: str = "openweather"  # 'openweather' | 'open_meteo' | 'existing'
    USGS_API_KEY: str = ""
    NASA_EARTHDATA_TOKEN: str = ""
    MAPTILER_API_KEY: str = ""
    ARCGIS_API_KEY: str = ""
    ARCGIS_ORIGIN: str = "http://localhost:5173"
    ULTRALYTICS_API_KEY: str = ""

    # ── Citizen Vision Verification (Ultralytics YOLO) ─────────────────
    CITIZEN_VISION_MODEL_PATH: str = "ml/checkpoints/citizen_vision/best.pt"
    CITIZEN_VISION_CONFIDENCE_THRESHOLD: float = 0.25
    CITIZEN_VISION_IOU_THRESHOLD: float = 0.45
    CITIZEN_VISION_DATASET_DIR: str = "data/datasets"
    CITIZEN_VISION_MODEL_VERSION: str = "citizen-vision-v1"

    # ── Scheduling ────────────────────────────────────────────────────
    RAINFALL_REFRESH_INTERVAL_MINUTES: int = 60
    WEATHER_REFRESH_INTERVAL_MINUTES: int = 60
    SOIL_MOISTURE_REFRESH_INTERVAL_MINUTES: int = 1440
    RISK_RECOMPUTE_INTERVAL_MINUTES: int = 60

    # ── Uploads ───────────────────────────────────────────────────────
    UPLOAD_DIR: str = "data/uploads"
    MAX_PHOTO_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 50

    # ── Notifications & SMS / Push Early Warning System ───────────────
    SMS_PROVIDER: str = ""  # 'twilio' | 'msg91' | 'aws_sns' | 'fast2sms' | 'mock'
    SMS_API_KEY: str = ""
    SMS_API_SECRET: str = ""
    SMS_SENDER_ID: str = "LNDJPA"  # TRAI DLT 6-char Alpha Sender ID
    SMS_TEMPLATE_ID: str = ""       # TRAI DLT Registered Template ID
    PUSH_PROVIDER: str = "webpush"  # 'webpush' | 'fcm' | 'mock'
    PUSH_VAPID_PUBLIC_KEY: str = ""
    PUSH_VAPID_PRIVATE_KEY: str = ""
    PUSH_VAPID_SUBJECT: str = "mailto:ops@landjepa.gov.in"
    FCM_SERVER_KEY: str = ""
    NOTIFICATION_TEST_MODE: bool = True
    NOTIFICATION_TEST_PHONE: str = "+919876543210"
    NOTIFICATION_WEBHOOK_SECRET: str = "landjepa-webhook-secret-2026"
    NOTIFICATION_MAX_RETRIES: int = 3
    NOTIFICATION_RATE_LIMIT_PER_MINUTE: int = 30

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
