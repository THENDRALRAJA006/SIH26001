"""LAND-JEPA — Notification and Early Warning Delivery ORM Models.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)

Database entities:
  - NotificationRecipient: Registered citizens and disaster management officers
  - NotificationPreference: Opt-in channels, consent timestamp, quiet hours
  - NotificationEvent: Atomic notification records across all channels
  - SmsDeliveryLog: SMS provider dispatch, DLT template, delivery receipts
  - PushDeliveryLog: Browser Web Push / FCM device dispatch and delivery log
  - NotificationTemplate: Indian DLT-compliant multilingual notification templates
  - DeviceRegistration: Active browser/mobile push tokens
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.alert import Alert
    from app.models.zone import Zone
    from app.models.user import User


class NotificationRecipient(Base, UUIDMixin, TimestampMixin):
    """Citizen or official registered to receive landslide early warnings."""

    __tablename__ = "notification_recipients"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="citizen")
    # 'citizen' | 'field_officer' | 'geologist' | 'disaster_director' | 'admin'

    phone_number: Mapped[str | None] = mapped_column(String(32), index=True)
    phone_masked: Mapped[str | None] = mapped_column(String(32))
    # Stored as masked e.g. +91******1234 for privacy / dashboard display

    email: Mapped[str | None] = mapped_column(String(255), index=True)
    preferred_language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    # 'en' | 'hi' | 'as' | 'bn' | 'mni'

    zone_id: Mapped[str | None] = mapped_column(String(50), index=True)
    corridor_name: Mapped[str | None] = mapped_column(String(255))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    alert_radius_km: Mapped[float] = mapped_column(Float, default=25.0)

    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_test_recipient: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    preferences: Mapped[Optional["NotificationPreference"]] = relationship(
        "NotificationPreference", back_populates="recipient", uselist=False, cascade="all, delete-orphan"
    )
    events: Mapped[list["NotificationEvent"]] = relationship(
        "NotificationEvent", back_populates="recipient"
    )

    def __repr__(self) -> str:
        return f"<NotificationRecipient {self.name} ({self.role}) phone={self.phone_masked}>"


class NotificationPreference(Base, UUIDMixin, TimestampMixin):
    """User channel opt-in preferences and explicit consent records."""

    __tablename__ = "notification_preferences"

    recipient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("notification_recipients.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    enable_push: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_sms: Mapped[bool] = mapped_column(Boolean, default=False)
    enable_in_app: Mapped[bool] = mapped_column(Boolean, default=True)

    min_severity: Mapped[str] = mapped_column(String(20), default="WATCH")
    # 'WATCH' | 'WARNING' | 'CRITICAL'

    quiet_hours_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    quiet_hours_start: Mapped[str | None] = mapped_column(String(10))  # e.g. "22:00"
    quiet_hours_end: Mapped[str | None] = mapped_column(String(10))    # e.g. "06:00"

    # Explicit Citizen Consent (Requirement 7)
    consent_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    # 'GRANTED' | 'REVOKED' | 'PENDING'
    consent_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consent_ip: Mapped[str | None] = mapped_column(String(45))
    consent_source: Mapped[str | None] = mapped_column(String(100), default="web_dashboard")

    recipient: Mapped["NotificationRecipient"] = relationship(
        "NotificationRecipient", back_populates="preferences"
    )


class NotificationEvent(Base, UUIDMixin, TimestampMixin):
    """Atomic notification transaction record across all notification channels."""

    __tablename__ = "notification_events"

    notification_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    alert_id: Mapped[str | None] = mapped_column(String(64), index=True)
    recipient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("notification_recipients.id", ondelete="SET NULL")
    )
    recipient_masked: Mapped[str] = mapped_column(String(100), nullable=False)

    channel: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # 'sms' | 'push' | 'in_app'

    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # 'WATCH' | 'WARNING' | 'CRITICAL'

    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en")

    # Idempotency key prevents duplicate sends: alert_id + recipient + channel + type
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)

    # Delivery states: QUEUED, SUBMITTED, SENT, DELIVERED, FAILED, RETRYING, CANCELLED, NOT_CONFIGURED
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="QUEUED", index=True)

    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    provider: Mapped[str | None] = mapped_column(String(50))
    provider_message_id: Mapped[str | None] = mapped_column(String(128), index=True)

    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(Text)

    is_test: Mapped[bool] = mapped_column(Boolean, default=False)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)

    recipient: Mapped[Optional["NotificationRecipient"]] = relationship(
        "NotificationRecipient", back_populates="events"
    )
    sms_log: Mapped[Optional["SmsDeliveryLog"]] = relationship(
        "SmsDeliveryLog", back_populates="event", uselist=False
    )
    push_log: Mapped[Optional["PushDeliveryLog"]] = relationship(
        "PushDeliveryLog", back_populates="event", uselist=False
    )

    def __repr__(self) -> str:
        return f"<NotificationEvent {self.notification_id} {self.channel} status={self.status}>"


class SmsDeliveryLog(Base, UUIDMixin, TimestampMixin):
    """Detailed SMS provider transmission logs and Indian DLT compliance records."""

    __tablename__ = "sms_delivery_logs"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("notification_events.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    notification_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    # 'twilio' | 'msg91' | 'aws_sns' | 'fast2sms' | 'mock'

    provider_message_id: Mapped[str | None] = mapped_column(String(128), index=True)
    recipient_phone_masked: Mapped[str] = mapped_column(String(32), nullable=False)

    # Indian DLT Compliance Fields
    template_id: Mapped[str | None] = mapped_column(String(64))
    sender_id: Mapped[str | None] = mapped_column(String(20))

    status: Mapped[str] = mapped_column(String(30), nullable=False, default="QUEUED")
    raw_provider_response: Mapped[dict | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(Text)

    event: Mapped["NotificationEvent"] = relationship(
        "NotificationEvent", back_populates="sms_log"
    )


class PushDeliveryLog(Base, UUIDMixin, TimestampMixin):
    """Push notification provider transmission logs."""

    __tablename__ = "push_delivery_logs"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("notification_events.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    notification_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    # 'webpush' | 'fcm' | 'mock'

    provider_message_id: Mapped[str | None] = mapped_column(String(128))
    device_id: Mapped[str | None] = mapped_column(String(128))
    push_token_masked: Mapped[str | None] = mapped_column(String(128))

    status: Mapped[str] = mapped_column(String(30), nullable=False, default="QUEUED")
    raw_provider_response: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)

    event: Mapped["NotificationEvent"] = relationship(
        "NotificationEvent", back_populates="push_log"
    )


class NotificationTemplate(Base, UUIDMixin, TimestampMixin):
    """Indian DLT and Multilingual registered templates."""

    __tablename__ = "notification_templates"

    template_code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    dlt_template_id: Mapped[str | None] = mapped_column(String(64), index=True)
    sender_id: Mapped[str] = mapped_column(String(20), default="LNDJPA")

    channel: Mapped[str] = mapped_column(String(20), nullable=False)  # 'sms' | 'push' | 'in_app'
    severity_tier: Mapped[str] = mapped_column(String(20), nullable=False)  # 'WATCH' | 'WARNING' | 'CRITICAL'
    language: Mapped[str] = mapped_column(String(10), nullable=False)      # 'en' | 'hi' | 'as' | 'bn' | 'mni'

    title_template: Mapped[str] = mapped_column(String(255), nullable=False)
    body_template: Mapped[str] = mapped_column(Text, nullable=False)
    variables_required: Mapped[dict | None] = mapped_column(JSON)
    is_dlt_approved: Mapped[bool] = mapped_column(Boolean, default=True)


class DeviceRegistration(Base, UUIDMixin, TimestampMixin):
    """Device registration for browser and mobile push delivery."""

    __tablename__ = "device_registrations"

    device_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    user_session_id: Mapped[str | None] = mapped_column(String(128), index=True)
    platform: Mapped[str] = mapped_column(String(50), default="browser")  # 'browser' | 'android' | 'ios'
    push_token: Mapped[str] = mapped_column(Text, nullable=False)
    push_endpoint: Mapped[str | None] = mapped_column(Text)
    auth_key: Mapped[str | None] = mapped_column(String(128))
    p256dh_key: Mapped[str | None] = mapped_column(String(255))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
