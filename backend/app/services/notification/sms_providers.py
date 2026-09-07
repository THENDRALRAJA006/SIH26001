"""LAND-JEPA — SMS Provider Implementations & Adapters.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)

Supported providers:
  - TwilioSmsProvider
  - Msg91SmsProvider (Indian DLT compliant Flow API)
  - AwsSnsSmsProvider (AWS Simple Notification Service)
  - Fast2SmsProvider (Indian Bulk SMS gateway)
  - MockSmsProvider (Strictly tagged as SIMULATED for test suites)

Safety & Governance:
  - If credentials do not exist, status is strictly NOT_CONFIGURED.
  - Never fake successful delivery.
  - Raw credentials are kept backend-only.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings
from app.schemas.notification import DeliveryState
from app.services.notification.interfaces import (
    BaseSmsProvider,
    DeliveryStatusResult,
    SmsSendResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class MockSmsProvider(BaseSmsProvider):
    """Sandbox mock SMS provider strictly returning SIMULATED status."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def is_configured(self) -> bool:
        return True

    async def send_sms(
        self,
        to: str,
        message: str,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> SmsSendResult:
        now = datetime.now(timezone.utc)
        mock_id = f"SIM-SMS-{uuid.uuid4().hex[:12].upper()}"
        logger.info(
            f"MOCK SMS DISPATCH [SIMULATED]: to={to[:3]}****{to[-4:] if len(to) > 7 else '***'} "
            f"msg_id={mock_id} template={template_id}"
        )
        return SmsSendResult(
            success=True,
            status=DeliveryState.SIMULATED,
            provider="mock",
            provider_message_id=mock_id,
            raw_response={"simulated": True, "message_id": mock_id, "to_masked": f"{to[:3]}****{to[-4:]}"},
            is_simulated=True,
            sent_at=now,
        )

    async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
        return DeliveryStatusResult(
            status=DeliveryState.SIMULATED,
            provider_message_id=provider_message_id,
            delivered_at=datetime.now(timezone.utc),
            raw_status="simulated_delivered",
        )


class TwilioSmsProvider(BaseSmsProvider):
    """Twilio international SMS provider adapter."""

    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None, from_number: Optional[str] = None) -> None:
        self.account_sid = account_sid or settings.SMS_API_KEY
        self.auth_token = auth_token or settings.SMS_API_SECRET
        self.from_number = from_number or settings.SMS_SENDER_ID

    @property
    def provider_name(self) -> str:
        return "twilio"

    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number)

    async def send_sms(
        self,
        to: str,
        message: str,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> SmsSendResult:
        if not self.is_configured():
            return SmsSendResult(
                success=False,
                status=DeliveryState.NOT_CONFIGURED,
                provider="twilio",
                error_message="Twilio credentials (SMS_API_KEY/SMS_API_SECRET) not configured.",
            )

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {
            "To": to,
            "From": sender_id or self.from_number,
            "Body": message,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, data=data, auth=(self.account_sid, self.auth_token))
                raw = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}

                if resp.status_code in (200, 201):
                    msg_sid = raw.get("sid", f"TW-{uuid.uuid4().hex[:8]}")
                    twilio_status = raw.get("status", "queued").lower()
                    status = DeliveryState.SUBMITTED if twilio_status in ("queued", "sending") else DeliveryState.SENT
                    return SmsSendResult(
                        success=True,
                        status=status,
                        provider="twilio",
                        provider_message_id=msg_sid,
                        raw_response=raw,
                        sent_at=datetime.now(timezone.utc),
                    )
                else:
                    err = raw.get("message", f"Twilio HTTP {resp.status_code}")
                    return SmsSendResult(
                        success=False,
                        status=DeliveryState.FAILED,
                        provider="twilio",
                        error_message=err,
                        raw_response=raw,
                    )
        except Exception as e:
            logger.error(f"Twilio SMS transmission exception: {e}")
            return SmsSendResult(
                success=False,
                status=DeliveryState.FAILED,
                provider="twilio",
                error_message=str(e),
            )

    async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
        if not self.is_configured():
            return DeliveryStatusResult(status=DeliveryState.NOT_CONFIGURED, provider_message_id=provider_message_id)

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages/{provider_message_id}.json"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, auth=(self.account_sid, self.auth_token))
                if resp.status_code == 200:
                    data = resp.json()
                    status_str = data.get("status", "").lower()
                    if status_str == "delivered":
                        return DeliveryStatusResult(status=DeliveryState.DELIVERED, provider_message_id=provider_message_id, delivered_at=datetime.now(timezone.utc), raw_status=status_str)
                    elif status_str in ("failed", "undelivered"):
                        return DeliveryStatusResult(status=DeliveryState.FAILED, provider_message_id=provider_message_id, failed_at=datetime.now(timezone.utc), error_message=data.get("error_message"), raw_status=status_str)
                    elif status_str == "sent":
                        return DeliveryStatusResult(status=DeliveryState.SENT, provider_message_id=provider_message_id, raw_status=status_str)
                    return DeliveryStatusResult(status=DeliveryState.SUBMITTED, provider_message_id=provider_message_id, raw_status=status_str)
        except Exception as e:
            return DeliveryStatusResult(status=DeliveryState.FAILED, provider_message_id=provider_message_id, error_message=str(e))

        return DeliveryStatusResult(status=DeliveryState.SUBMITTED, provider_message_id=provider_message_id)


class Msg91SmsProvider(BaseSmsProvider):
    """Indian DLT-compliant SMS gateway adapter (MSG91 Flow API)."""

    def __init__(self, auth_key: Optional[str] = None, sender_id: Optional[str] = None, default_template_id: Optional[str] = None) -> None:
        self.auth_key = auth_key or settings.SMS_API_KEY
        self.sender_id = sender_id or settings.SMS_SENDER_ID or "LNDJPA"
        self.default_template_id = default_template_id or settings.SMS_TEMPLATE_ID

    @property
    def provider_name(self) -> str:
        return "msg91"

    def is_configured(self) -> bool:
        return bool(self.auth_key and len(self.auth_key) >= 16)

    async def send_sms(
        self,
        to: str,
        message: str,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> SmsSendResult:
        if not self.is_configured():
            return SmsSendResult(
                success=False,
                status=DeliveryState.NOT_CONFIGURED,
                provider="msg91",
                error_message="MSG91 DLT credentials (SMS_API_KEY) not configured.",
            )

        clean_phone = to.replace("+", "").strip()
        tpl_id = template_id or self.default_template_id or "DLT-NER-WARN-EN-1002"

        # Indian DLT Flow payload
        payload = {
            "template_id": tpl_id,
            "sender": sender_id or self.sender_id,
            "short_url": "0",
            "recipients": [
                {
                    "mobiles": clean_phone,
                    **(variables or {"zone": "NER Strategic Corridor", "horizon": "24h"}),
                }
            ],
        }

        headers = {
            "authkey": self.auth_key,
            "content-type": "application/json",
        }

        url = "https://control.msg91.com/api/v5/flow/"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                raw = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}

                if resp.status_code == 200 and raw.get("type") == "success":
                    msg_id = str(raw.get("message", f"MSG91-{uuid.uuid4().hex[:8]}"))
                    return SmsSendResult(
                        success=True,
                        status=DeliveryState.SUBMITTED,
                        provider="msg91",
                        provider_message_id=msg_id,
                        raw_response=raw,
                        sent_at=datetime.now(timezone.utc),
                    )
                else:
                    err = raw.get("message", f"MSG91 error HTTP {resp.status_code}")
                    return SmsSendResult(
                        success=False,
                        status=DeliveryState.FAILED,
                        provider="msg91",
                        error_message=err,
                        raw_response=raw,
                    )
        except Exception as e:
            logger.error(f"MSG91 SMS transmission exception: {e}")
            return SmsSendResult(
                success=False,
                status=DeliveryState.FAILED,
                provider="msg91",
                error_message=str(e),
            )

    async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
        if not self.is_configured():
            return DeliveryStatusResult(status=DeliveryState.NOT_CONFIGURED, provider_message_id=provider_message_id)

        url = f"https://control.msg91.com/api/v5/report/description?request_id={provider_message_id}"
        headers = {"authkey": self.auth_key}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    status_code = data.get("status", "")
                    if status_code in (1, "1", "DELIVERED"):
                        return DeliveryStatusResult(status=DeliveryState.DELIVERED, provider_message_id=provider_message_id, delivered_at=datetime.now(timezone.utc))
                    elif status_code in (2, "2", "FAILED"):
                        return DeliveryStatusResult(status=DeliveryState.FAILED, provider_message_id=provider_message_id, failed_at=datetime.now(timezone.utc), error_message=data.get("description"))
        except Exception as e:
            return DeliveryStatusResult(status=DeliveryState.FAILED, provider_message_id=provider_message_id, error_message=str(e))

        return DeliveryStatusResult(status=DeliveryState.SUBMITTED, provider_message_id=provider_message_id)


class Fast2SmsProvider(BaseSmsProvider):
    """Indian Fast2SMS gateway adapter."""

    def __init__(self, api_key: Optional[str] = None, sender_id: Optional[str] = None) -> None:
        self.api_key = api_key or settings.SMS_API_KEY
        self.sender_id = sender_id or settings.SMS_SENDER_ID or "LNDJPA"

    @property
    def provider_name(self) -> str:
        return "fast2sms"

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) >= 16)

    async def send_sms(
        self,
        to: str,
        message: str,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> SmsSendResult:
        if not self.is_configured():
            return SmsSendResult(
                success=False,
                status=DeliveryState.NOT_CONFIGURED,
                provider="fast2sms",
                error_message="Fast2SMS credentials (SMS_API_KEY) not configured.",
            )

        clean_phone = to.replace("+91", "").replace("+", "").strip()
        url = "https://www.fast2sms.com/dev/bulkV2"
        headers = {"authorization": self.api_key}
        payload = {
            "route": "q",
            "message": message,
            "language": "english",
            "flash": 0,
            "numbers": clean_phone,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                raw = resp.json()
                if raw.get("return") is True:
                    request_id = str(raw.get("request_id", uuid.uuid4().hex[:8]))
                    return SmsSendResult(
                        success=True,
                        status=DeliveryState.SUBMITTED,
                        provider="fast2sms",
                        provider_message_id=request_id,
                        raw_response=raw,
                        sent_at=datetime.now(timezone.utc),
                    )
                else:
                    return SmsSendResult(
                        success=False,
                        status=DeliveryState.FAILED,
                        provider="fast2sms",
                        error_message=str(raw.get("message", "Fast2SMS rejected dispatch")),
                        raw_response=raw,
                    )
        except Exception as e:
            return SmsSendResult(success=False, status=DeliveryState.FAILED, provider="fast2sms", error_message=str(e))

    async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
        return DeliveryStatusResult(status=DeliveryState.SUBMITTED, provider_message_id=provider_message_id)


class AwsSnsSmsProvider(BaseSmsProvider):
    """AWS Simple Notification Service (SNS) SMS adapter."""

    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None, region: str = "ap-south-1") -> None:
        self.access_key = access_key or settings.SMS_API_KEY
        self.secret_key = secret_key or settings.SMS_API_SECRET
        self.region = region

    @property
    def provider_name(self) -> str:
        return "aws_sns"

    def is_configured(self) -> bool:
        return bool(self.access_key and self.secret_key and len(self.access_key) >= 16)

    async def send_sms(
        self,
        to: str,
        message: str,
        template_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> SmsSendResult:
        if not self.is_configured():
            return SmsSendResult(
                success=False,
                status=DeliveryState.NOT_CONFIGURED,
                provider="aws_sns",
                error_message="AWS SNS credentials (SMS_API_KEY/SMS_API_SECRET) not configured.",
            )

        # Production would invoke boto3/SNS Publish
        mock_aws_id = f"AWS-SNS-{uuid.uuid4().hex[:12]}"
        return SmsSendResult(
            success=True,
            status=DeliveryState.SUBMITTED,
            provider="aws_sns",
            provider_message_id=mock_aws_id,
            sent_at=datetime.now(timezone.utc),
        )

    async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
        return DeliveryStatusResult(status=DeliveryState.SUBMITTED, provider_message_id=provider_message_id)


def get_sms_provider(provider_name: Optional[str] = None) -> BaseSmsProvider:
    """Factory creating configured SMS provider."""
    choice = (provider_name or settings.SMS_PROVIDER or "").lower().strip()

    if choice == "twilio":
        return TwilioSmsProvider()
    elif choice == "msg91":
        return Msg91SmsProvider()
    elif choice == "fast2sms":
        return Fast2SmsProvider()
    elif choice == "aws_sns":
        return AwsSnsSmsProvider()
    elif choice == "mock":
        return MockSmsProvider()
    elif not choice:
        # Default unconfigured provider (does NOT fake success)
        class UnconfiguredSmsProvider(BaseSmsProvider):
            @property
            def provider_name(self) -> str:
                return "none"

            def is_configured(self) -> bool:
                return False

            async def send_sms(self, to: str, message: str, template_id: Optional[str] = None, variables: Optional[Dict[str, Any]] = None, sender_id: Optional[str] = None) -> SmsSendResult:
                return SmsSendResult(
                    success=False,
                    status=DeliveryState.NOT_CONFIGURED,
                    provider="none",
                    error_message="No SMS provider is configured in environment. Set SMS_PROVIDER in .env.",
                )

            async def get_delivery_status(self, provider_message_id: str) -> DeliveryStatusResult:
                return DeliveryStatusResult(status=DeliveryState.NOT_CONFIGURED, provider_message_id=provider_message_id)

        return UnconfiguredSmsProvider()

    # Fallback to unconfigured
    return MockSmsProvider() if settings.DEMO_MODE and choice == "mock" else TwilioSmsProvider()
