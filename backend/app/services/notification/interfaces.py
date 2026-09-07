"""LAND-JEPA — Notification Provider Interfaces & Data Contracts.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.notification import DeliveryState


@dataclass
class SmsSendResult:
    success: bool
    status: DeliveryState
    provider: str
    provider_message_id: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    is_simulated: bool = False
    sent_at: Optional[datetime] = None


@dataclass
class PushSendResult:
    success: bool
    status: DeliveryState
    provider: str
    provider_message_id: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    is_simulated: bool = False
    sent_at: Optional[datetime] = None


@dataclass
class DeliveryStatusResult:
    status: DeliveryState
    provider_message_id: str
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    raw_status: Optional[str] = None


class BaseSmsProvider(ABC):
    """Abstract interface for vendor-agnostic SMS dispatch."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g. 'twilio', 'msg91', 'aws_sns', 'fast2sms', 'mock')."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if legitimate credentials are configured."""
        pass

    @abstractmethod
    async def send_sms(
        self,
        to: str,
        message: str,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> SmsSendResult:
        """Submit SMS to provider gateway."""
        pass

    @abstractmethod
    async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
        """Poll or query provider delivery receipt."""
        pass


class BasePushProvider(ABC):
    """Abstract interface for vendor-agnostic push notifications."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g. 'webpush', 'fcm', 'mock')."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if push keys/credentials are configured."""
        pass

    @abstractmethod
    async def send_push(
        self,
        push_token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
        auth_key: Optional[str] = None,
        p256dh_key: Optional[str] = None,
    ) -> PushSendResult:
        """Dispatch push notification to browser or mobile client."""
        pass
