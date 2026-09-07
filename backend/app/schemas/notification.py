"""LAND-JEPA — Notification Schemas
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DeliveryState(str, Enum):
    QUEUED = "QUEUED"
    SUBMITTED = "SUBMITTED"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    SIMULATED = "SIMULATED"


class NotificationChannel(str, Enum):
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class SeverityTier(str, Enum):
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SupportedLanguage(str, Enum):
    EN = "en"    # English
    HI = "hi"    # Hindi
    AS = "as"    # Assamese
    BN = "bn"    # Bengali
    MNI = "mni"  # Manipuri / Meitei


class ConsentStatus(str, Enum):
    GRANTED = "GRANTED"
    REVOKED = "REVOKED"
    PENDING = "PENDING"


# ── Citizen Subscription Request ─────────────────────────────────────
class CitizenSubscribeRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    phone_number: Optional[str] = Field(None, pattern=r"^(\+91|91)?[6-9]\d{9}$")
    email: Optional[str] = None
    preferred_language: SupportedLanguage = SupportedLanguage.EN
    zone_id: Optional[str] = "REAL-NER-001"
    corridor_name: Optional[str] = "NH-27 Guwahati–Shillong"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    alert_radius_km: float = Field(25.0, ge=1.0, le=100.0)
    enable_sms: bool = False
    enable_push: bool = True
    enable_in_app: bool = True
    explicit_consent: bool = Field(..., description="Must explicitly grant consent to receive emergency broadcasts")


class CitizenUnsubscribeRequest(BaseModel):
    phone_number: Optional[str] = None
    recipient_id: Optional[str] = None


# ── Operator Real Test SMS Request ────────────────────────────────────
class OperatorTestSmsRequest(BaseModel):
    test_phone_number: str = Field(..., description="Target Indian mobile number, e.g. +919876543210")
    officer_id: str = Field("OFFICER-NER-01", description="Authenticated Officer ID")
    zone_id: str = Field("REAL-NER-001", description="Monitored corridor zone")
    severity: SeverityTier = SeverityTier.WARNING
    language: SupportedLanguage = SupportedLanguage.EN
    explicit_confirmation: bool = Field(..., description="Operator must confirm this is a controlled test dispatch")


class OperatorTestPushRequest(BaseModel):
    push_token: Optional[str] = None
    officer_id: str = "OFFICER-NER-01"
    zone_id: str = "REAL-NER-001"
    severity: SeverityTier = SeverityTier.WATCH
    language: SupportedLanguage = SupportedLanguage.EN


# ── Delivery Webhook Payload ──────────────────────────────────────────
class WebhookStatusPayload(BaseModel):
    provider: str  # 'twilio' | 'msg91' | 'aws_sns' | 'fast2sms'
    provider_message_id: str
    status: str  # e.g. 'delivered', 'failed', 'undelivered', 'sent'
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    signature: Optional[str] = None


# ── Responses ─────────────────────────────────────────────────────────
class NotificationEventResponse(BaseModel):
    notification_id: str
    alert_id: Optional[str] = None
    recipient_masked: str
    channel: str
    severity: str
    zone_id: str
    title: str
    body: str
    language: str
    status: DeliveryState
    provider: Optional[str] = None
    provider_message_id: Optional[str] = None
    attempt_count: int = 0
    created_at: str
    sent_at: Optional[str] = None
    delivered_at: Optional[str] = None
    failed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    is_test: bool = False
    is_simulated: bool = False


class NotificationListResponse(BaseModel):
    total: int
    page: int = 1
    page_size: int = 50
    notifications: List[NotificationEventResponse]


class RecipientSummaryResponse(BaseModel):
    recipient_id: str
    name: str
    role: str
    phone_masked: Optional[str] = None
    preferred_language: str
    zone_id: Optional[str] = None
    corridor_name: Optional[str] = None
    consent_status: str
    enable_sms: bool
    enable_push: bool
    active: bool


class NotificationStatsResponse(BaseModel):
    total_events: int
    sms_total: int
    sms_delivered: int
    sms_failed: int
    sms_not_configured: int
    push_total: int
    push_delivered: int
    push_failed: int
    in_app_total: int
    delivery_success_rate_pct: float
    sms_failure_rate_pct: float
    push_failure_rate_pct: float
    average_delivery_latency_ms: float
    queue_depth: int
    retry_count: int
    daily_matrix: Dict[str, Any]
    seven_day_matrix: Dict[str, Any]
