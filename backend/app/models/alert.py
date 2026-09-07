"""LAND-JEPA — Alert and alert delivery ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.zone import Zone
    from app.models.user import User
    from app.models.risk import RiskPrediction


class Alert(Base, UUIDMixin, TimestampMixin):
    """
    Generated alert for a zone.

    SAFETY RULE: When is_demo=True, NO real notifications may be sent.
    The demo_note field MUST explain this is a test alert.

    In demo mode, alert_deliveries will have is_simulated=True.
    """

    __tablename__ = "alerts"

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False, index=True
    )
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("risk_predictions.id")
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Values: risk_level_change | high_risk | critical | priority_1 | manual
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    # Values: info | warning | critical

    # Multilingual content
    title_en: Mapped[str] = mapped_column(String(500), nullable=False)
    body_en: Mapped[str] = mapped_column(Text, nullable=False)
    title_hi: Mapped[str | None] = mapped_column(String(500))
    body_hi: Mapped[str | None] = mapped_column(Text)

    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    demo_note: Mapped[str | None] = mapped_column(Text)
    # MUST be set when is_demo=True: "DEMO ALERT — not a real emergency"

    zone: Mapped["Zone"] = relationship("Zone", back_populates="alerts")
    deliveries: Mapped[list["AlertDelivery"]] = relationship(back_populates="alert")

    def __repr__(self) -> str:
        demo_tag = " [DEMO]" if self.is_demo else ""
        return f"<Alert {self.alert_type} severity={self.severity}{demo_tag}>"


class AlertDelivery(Base, UUIDMixin, TimestampMixin):
    """Tracks delivery status of an alert to a channel/recipient."""

    __tablename__ = "alert_deliveries"

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("alerts.id"), nullable=False, index=True
    )
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    # Values: web | push | sms | email
    recipient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    recipient_ref: Mapped[str | None] = mapped_column(String(255))
    # phone number or email if no user record

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    # Values: pending | sent | failed | simulated
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    is_simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # True when alert_demo_only=True (no real delivery made)

    alert: Mapped[Alert] = relationship(back_populates="deliveries")

    def __repr__(self) -> str:
        sim = " [SIMULATED]" if self.is_simulated else ""
        return f"<AlertDelivery {self.channel} status={self.status}{sim}>"


class SyncQueue(Base, UUIDMixin, TimestampMixin):
    """Offline sync queue for citizen and field reports."""

    __tablename__ = "sync_queue"

    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Values: citizen_report | field_report
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    device_id: Mapped[str | None] = mapped_column(String(255))
    payload: Mapped[dict] = mapped_column("payload_json", type_=__import__("sqlalchemy.dialects.postgresql", fromlist=["JSONB"]).JSONB, nullable=False)
    sync_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    # Values: pending | syncing | synced | failed
    attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    last_attempted: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<SyncQueue {self.entity_type}/{self.entity_id} status={self.sync_status}>"


class AuditLog(Base):
    """Append-only audit log for all write operations."""

    __tablename__ = "audit_logs"

    from sqlalchemy import BigInteger, func
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    # Values: CREATE | UPDATE | DELETE | LOGIN | LOGOUT | REVIEW
    resource: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_id: Mapped[str | None] = mapped_column(String(255))
    old_data: Mapped[dict | None] = mapped_column(__import__("sqlalchemy.dialects.postgresql", fromlist=["JSONB"]).JSONB)
    new_data: Mapped[dict | None] = mapped_column(__import__("sqlalchemy.dialects.postgresql", fromlist=["JSONB"]).JSONB)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=__import__("sqlalchemy", fromlist=["func"]).func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} {self.resource}/{self.resource_id}>"
