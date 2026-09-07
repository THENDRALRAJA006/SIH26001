"""LAND-JEPA — Push Notification Provider Implementations & Adapters.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)

Supported providers:
  - WebPushProvider (W3C Web Push Protocol / VAPID)
  - FirebasePushProvider (FCM v1 REST API)
  - MockPushProvider (Strictly tagged as SIMULATED for test suites)
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings
from app.schemas.notification import DeliveryState
from app.services.notification.interfaces import (
    BasePushProvider,
    PushSendResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class MockPushProvider(BasePushProvider):
    """Sandbox mock push provider returning SIMULATED delivery."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def is_configured(self) -> bool:
        return True

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
        now = datetime.now(timezone.utc)
        mock_id = f"SIM-PUSH-{uuid.uuid4().hex[:12].upper()}"
        token_masked = f"{push_token[:6]}...{push_token[-4:]}" if len(push_token) > 10 else "token-masked"
        logger.info(f"MOCK PUSH DISPATCH [SIMULATED]: token={token_masked} title='{title}' id={mock_id}")
        return PushSendResult(
            success=True,
            status=DeliveryState.SIMULATED,
            provider="mock",
            provider_message_id=mock_id,
            raw_response={"simulated": True, "token_masked": token_masked, "message_id": mock_id},
            is_simulated=True,
            sent_at=now,
        )


class WebPushProvider(BasePushProvider):
    """W3C Web Push VAPID adapter for modern desktop & mobile browsers."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        private_key: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> None:
        self.public_key = public_key or settings.PUSH_VAPID_PUBLIC_KEY
        self.private_key = private_key or settings.PUSH_VAPID_PRIVATE_KEY
        self.subject = subject or settings.PUSH_VAPID_SUBJECT or "mailto:ops@landjepa.gov.in"

    @property
    def provider_name(self) -> str:
        return "webpush"

    def is_configured(self) -> bool:
        return bool(self.public_key and self.private_key and len(self.private_key) >= 16)

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
        if not self.is_configured():
            return PushSendResult(
                success=False,
                status=DeliveryState.NOT_CONFIGURED,
                provider="webpush",
                error_message="WebPush VAPID keys (PUSH_VAPID_PUBLIC_KEY/PUSH_VAPID_PRIVATE_KEY) not configured.",
            )

        target_endpoint = endpoint or push_token
        if not target_endpoint.startswith("http"):
            # Invalid or incomplete endpoint
            return PushSendResult(
                success=False,
                status=DeliveryState.FAILED,
                provider="webpush",
                error_message="Invalid Web Push endpoint URL.",
            )

        payload = json.dumps({
            "title": title,
            "body": body,
            "icon": "/logo192.png",
            "badge": "/badge72.png",
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # In production with pywebpush:
        # webpush(subscription_info, data=payload, vapid_private_key=..., vapid_claims=...)
        msg_id = f"WP-{uuid.uuid4().hex[:10]}"
        try:
            headers = {
                "TTL": "86400",
                "Urgency": "high" if (data or {}).get("severity") == "CRITICAL" else "normal",
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                # Direct post to push service
                resp = await client.post(target_endpoint, content=payload.encode("utf-8"), headers=headers)
                if resp.status_code in (200, 201, 202):
                    return PushSendResult(
                        success=True,
                        status=DeliveryState.SUBMITTED,
                        provider="webpush",
                        provider_message_id=msg_id,
                        sent_at=datetime.now(timezone.utc),
                    )
                else:
                    return PushSendResult(
                        success=False,
                        status=DeliveryState.FAILED,
                        provider="webpush",
                        error_message=f"Web Push endpoint returned HTTP {resp.status_code}",
                    )
        except Exception as e:
            return PushSendResult(
                success=False,
                status=DeliveryState.FAILED,
                provider="webpush",
                error_message=str(e),
            )


class FirebasePushProvider(BasePushProvider):
    """Firebase Cloud Messaging (FCM v1) adapter."""

    def __init__(self, server_key: Optional[str] = None) -> None:
        self.server_key = server_key or settings.FCM_SERVER_KEY

    @property
    def provider_name(self) -> str:
        return "fcm"

    def is_configured(self) -> bool:
        return bool(self.server_key and len(self.server_key) >= 20)

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
        if not self.is_configured():
            return PushSendResult(
                success=False,
                status=DeliveryState.NOT_CONFIGURED,
                provider="fcm",
                error_message="Firebase FCM_SERVER_KEY not configured.",
            )

        url = "https://fcm.googleapis.com/fcm/send"
        headers = {
            "Authorization": f"key={self.server_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "to": push_token,
            "notification": {"title": title, "body": body, "sound": "default"},
            "data": data or {},
            "priority": "high",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                raw = resp.json()
                if resp.status_code == 200 and raw.get("success") == 1:
                    msg_id = f"FCM-{raw.get('results', [{}])[0].get('message_id', uuid.uuid4().hex[:8])}"
                    return PushSendResult(
                        success=True,
                        status=DeliveryState.SUBMITTED,
                        provider="fcm",
                        provider_message_id=msg_id,
                        raw_response=raw,
                        sent_at=datetime.now(timezone.utc),
                    )
                else:
                    return PushSendResult(
                        success=False,
                        status=DeliveryState.FAILED,
                        provider="fcm",
                        error_message=str(raw),
                        raw_response=raw,
                    )
        except Exception as e:
            return PushSendResult(success=False, status=DeliveryState.FAILED, provider="fcm", error_message=str(e))


def get_push_provider(provider_name: Optional[str] = None) -> BasePushProvider:
    """Factory returning configured Push provider."""
    choice = (provider_name or settings.PUSH_PROVIDER or "").lower().strip()

    if choice == "webpush":
        return WebPushProvider()
    elif choice == "fcm":
        return FirebasePushProvider()
    elif choice == "mock":
        return MockPushProvider()
    else:
        return MockPushProvider() if settings.DEMO_MODE else WebPushProvider()
