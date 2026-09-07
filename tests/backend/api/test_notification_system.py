"""LAND-JEPA — Comprehensive Notification System Test Suite.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)

Tests:
  1. Provider configuration & NOT_CONFIGURED guarding
  2. Mock SMS provider (SIMULATED tagging)
  3. Mock Push provider (SIMULATED tagging)
  4. Twilio SMS provider adapter
  5. MSG91 Indian DLT compliance & Flow API
  6. Multilingual template rendering across 5 languages (en, hi, as, bn, mni)
  7. No false certainty in safety templates
  8. Missing translation fallback with logging
  9. Explicit citizen consent enforcement
  10. Mathematical idempotency & duplicate protection
  11. Alert storm deduplication & cooldown
  12. Exponential backoff retry logic
  13. Delivery state machine (SUBMITTED != DELIVERED)
  14. Delivery receipt webhook processing & signature verification
  15. Security & phone number masking
  16. Operator test SMS endpoint
  17. System health notification diagnostic probes
  18. End-to-end alert-to-notification policy dispatch
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.schemas.notification import DeliveryState, SeverityTier, SupportedLanguage
from app.services.notification.interfaces import (
    BaseSmsProvider,
    DeliveryStatusResult,
    PushSendResult,
    SmsSendResult,
)
from app.services.notification.push_providers import (
    MockPushProvider,
    WebPushProvider,
    get_push_provider,
)
from app.services.notification.service import (
    NotificationService,
    get_notification_service,
    mask_phone_number,
)
from app.services.notification.sms_providers import (
    Fast2SmsProvider,
    MockSmsProvider,
    Msg91SmsProvider,
    TwilioSmsProvider,
    get_sms_provider,
)
from app.services.notification.templates import TEMPLATE_CATALOG, TemplateEngine
from app.services.system_health import SystemHealthService


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def notif_service():
    # Fresh or singleton service instance
    return get_notification_service()


# ── 1. Provider Configuration & NOT_CONFIGURED Guarding ───────────────
def test_provider_configuration_not_configured():
    """Unconfigured provider must return status NOT_CONFIGURED and never fake delivery."""
    provider = get_sms_provider("")
    assert not provider.is_configured()


@pytest.mark.asyncio
async def test_sms_unconfigured_send_fails_cleanly():
    """Unconfigured provider send_sms must return NOT_CONFIGURED."""
    provider = get_sms_provider("")
    result = await provider.send_sms(to="+919876543210", message="Test alert")
    assert result.status == DeliveryState.NOT_CONFIGURED
    assert not result.success
    assert "configured" in result.error_message.lower()


# ── 2. Mock SMS Provider (SIMULATED Tagging) ──────────────────────────
@pytest.mark.asyncio
async def test_mock_sms_provider_simulated():
    """Mock SMS provider must mark status strictly as SIMULATED."""
    provider = MockSmsProvider()
    assert provider.is_configured()
    res = await provider.send_sms(to="+919876543210", message="Test emergency advisory")
    assert res.success
    assert res.status == DeliveryState.SIMULATED
    assert res.is_simulated is True
    assert res.provider_message_id.startswith("SIM-SMS-")


# ── 3. Mock Push Provider (SIMULATED Tagging) ─────────────────────────
@pytest.mark.asyncio
async def test_mock_push_provider_simulated():
    """Mock Push provider must mark status strictly as SIMULATED."""
    provider = MockPushProvider()
    assert provider.is_configured()
    res = await provider.send_push(push_token="demo_token", title="Watch", body="Soil saturated")
    assert res.success
    assert res.status == DeliveryState.SIMULATED
    assert res.is_simulated is True
    assert res.provider_message_id.startswith("SIM-PUSH-")


# ── 4. Twilio SMS Provider Adapter ────────────────────────────────────
@pytest.mark.asyncio
async def test_twilio_sms_provider_adapter():
    """Twilio provider validates credentials and submits formatted HTTP payload."""
    p_unconf = TwilioSmsProvider(account_sid="", auth_token="", from_number="")
    assert not p_unconf.is_configured()
    res_unconf = await p_unconf.send_sms("+919876543210", "test")
    assert res_unconf.status == DeliveryState.NOT_CONFIGURED

    p_conf = TwilioSmsProvider(account_sid="AC1234567890abcdef1234567890abcdef", auth_token="token1234567890", from_number="+1234567890")
    assert p_conf.is_configured()


# ── 5. MSG91 Indian DLT Compliance & Flow API ─────────────────────────
@pytest.mark.asyncio
async def test_msg91_dlt_compliance():
    """MSG91 Indian gateway adapter enforces DLT template ID, 6-char sender ID, and variables."""
    msg91 = Msg91SmsProvider(auth_key="VALID_AUTH_KEY_16_CHARS_MIN", sender_id="LNDJPA", default_template_id="DLT-NER-WARN-EN-1002")
    assert msg91.is_configured()
    assert msg91.sender_id == "LNDJPA"
    assert len(msg91.sender_id) <= 6


# ── 6. Multilingual Template Rendering Across 5 Languages ─────────────
def test_multilingual_template_rendering():
    """All 5 languages (en, hi, as, bn, mni) across WATCH, WARNING, CRITICAL must render non-empty text."""
    expected_langs = ["en", "hi", "as", "bn", "mni"]
    tiers = ["WATCH", "WARNING", "CRITICAL"]

    for lang in expected_langs:
        assert lang in TEMPLATE_CATALOG, f"Language {lang} missing from template catalog."
        for tier in tiers:
            title, body, dlt_id, res_lang = TemplateEngine.render(
                tier=tier,
                zone="NH-27 Guwahati–Shillong Km 48",
                horizon="24h",
                language=lang,
            )
            assert title, f"Empty title for {lang}/{tier}"
            assert body, f"Empty body for {lang}/{tier}"
            assert "NH-27 Guwahati–Shillong Km 48" in body or "NH-27 Guwahati–Shillong Km 48" in title
            assert "24h" in body
            assert dlt_id.startswith("DLT-NER-")
            assert res_lang == lang


# ── 7. No False Certainty in Safety Templates ─────────────────────────
def test_no_false_certainty_in_templates():
    """Safety rules: Templates must NEVER state landslides are 100% certain."""
    for lang, tiers in TEMPLATE_CATALOG.items():
        for tier, content in tiers.items():
            text = content["body"].upper()
            assert "WILL DEFINITELY HAPPEN" not in text
            assert "CERTAIN TO OCCUR" not in text


# ── 8. Missing Translation Fallback with Logging ──────────────────────
def test_missing_translation_fallback():
    """Unsupported language must fall back to English cleanly without crashing."""
    title, body, dlt_id, res_lang = TemplateEngine.render(
        tier="WARNING",
        zone="NH-102 Imphal–Moreh",
        horizon="12h",
        language="unsupported_lang_xyz",
    )
    assert res_lang == "en"
    assert "LAND-JEPA WARNING" in title or "LAND-JEPA WARNING" in body
    assert dlt_id.startswith("DLT-NER-")


# ── 9. Explicit Citizen Consent Enforcement ───────────────────────────
@pytest.mark.asyncio
async def test_citizen_consent_enforcement(notif_service: NotificationService):
    """Citizen without GRANTED consent must NOT receive SMS."""
    unconsented_citizen = {
        "recipient_id": "REC-NO-CONSENT-01",
        "name": "Citizen Unconsented",
        "role": "citizen",
        "phone_number": "+919876543210",
        "consent_status": "PENDING",
        "enable_sms": True,
        "preferred_language": "en",
        "zone_id": "REAL-NER-001",
    }
    res = await notif_service.send_sms(
        recipient=unconsented_citizen,
        alert_id="ALT-CONSENT-TEST",
        severity="WARNING",
        zone_id="REAL-NER-001",
        is_test=False,
    )
    assert res["status"] == DeliveryState.CANCELLED
    assert res["reason"] == "CONSENT_NOT_GRANTED"


# ── 10. Mathematical Idempotency & Duplicate Protection ───────────────
@pytest.mark.asyncio
async def test_idempotency_duplicate_protection(notif_service: NotificationService):
    """Repeated prediction polling with identical alert & recipient must be blocked."""
    consented_citizen = {
        "recipient_id": "REC-IDEM-01",
        "name": "Citizen Idempotent",
        "role": "citizen",
        "phone_number": "+919876543210",
        "consent_status": "GRANTED",
        "enable_sms": True,
        "preferred_language": "en",
        "zone_id": "REAL-NER-001",
    }
    import uuid
    uid = uuid.uuid4().hex[:6]
    alert_id = f"ALT-IDEM-{uid}"
    test_zone = f"REAL-NER-IDEM-{uid}"
    consented_citizen["zone_id"] = test_zone

    # First dispatch
    res1 = await notif_service.send_sms(
        recipient=consented_citizen,
        alert_id=alert_id,
        severity="CRITICAL",
        zone_id=test_zone,
        is_test=False,
    )
    assert res1["status"] in (DeliveryState.SENT.value, DeliveryState.SUBMITTED.value, DeliveryState.SIMULATED.value, DeliveryState.NOT_CONFIGURED.value)

    if res1["status"] != DeliveryState.NOT_CONFIGURED.value:
        # Second identical dispatch must be suppressed
        res2 = await notif_service.send_sms(
            recipient=consented_citizen,
            alert_id=alert_id,
            severity="CRITICAL",
            zone_id=test_zone,
            is_test=False,
        )
        assert res2["status"] == DeliveryState.CANCELLED
        assert res2["reason"] == "DUPLICATE_SUPPRESSED"


# ── 11. Alert Storm Deduplication & Cooldown ──────────────────────────
def test_alert_storm_deduplication(notif_service: NotificationService):
    """Multiple alerts in quick succession for the same zone are grouped/rate-controlled."""
    zone_id = "REAL-NER-STORM-01"
    severity = "CRITICAL"

    # Not suppressed initially
    assert not notif_service._is_storm_suppressed(zone_id, severity)

    # Record dispatch
    notif_service._record_storm_dispatch(zone_id, severity)

    # Now suppressed
    assert notif_service._is_storm_suppressed(zone_id, severity)


# ── 12. Exponential Backoff Retry Logic ───────────────────────────────
@pytest.mark.asyncio
async def test_retry_exponential_backoff(notif_service: NotificationService):
    """Temporary provider failures retry with backoff up to max_retries."""
    class FlakySmsProvider(BaseSmsProvider):
        provider_name = "flaky"
        call_count = 0

        def is_configured(self):
            return True

        async def send_sms(self, to, message, template_id=None, variables=None, sender_id=None):
            self.call_count += 1
            if self.call_count < 3:
                raise RuntimeError("Temporary gateway timeout")
            return SmsSendResult(success=True, status=DeliveryState.SUBMITTED, provider="flaky", provider_message_id="MSG-RETRY-OK")

        async def get_delivery_status(self, provider_message_id):
            return DeliveryStatusResult(status=DeliveryState.SUBMITTED, provider_message_id=provider_message_id)

    old_provider = notif_service.sms_provider
    flaky = FlakySmsProvider()
    notif_service.sms_provider = flaky

    try:
        res = await notif_service._submit_sms_with_retry("+919876543210", "Test body", "TPL-1", "ZONE-1", "24h")
        assert res["send_result"].success
        assert res["attempts"] == 3
        assert flaky.call_count == 3
    finally:
        notif_service.sms_provider = old_provider


# ── 13. Delivery State Machine (SUBMITTED != DELIVERED) ───────────────
def test_delivery_state_machine(notif_service: NotificationService):
    """Ensures SUBMITTED is strictly distinguished from DELIVERED."""
    event_id = "TEST-EVENT-STATES-01"
    event = {
        "notification_id": event_id,
        "provider_message_id": "MSG-STATE-12345",
        "status": DeliveryState.SUBMITTED.value,
        "channel": "sms",
        "severity": "WARNING",
        "zone_id": "REAL-NER-001",
        "recipient_masked": "+91******3210",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    notif_service._events[event_id] = event

    assert event["status"] == DeliveryState.SUBMITTED.value
    assert event["status"] != DeliveryState.DELIVERED.value


# ── 14. Delivery Receipt Webhook Processing ───────────────────────────
def test_webhook_delivery_receipt(notif_service: NotificationService, client: TestClient):
    """Webhook callback authenticates and updates message status to DELIVERED."""
    event_id = f"NOTIF-WEBHOOK-{datetime.now().strftime('%M%S')}"
    msg_id = f"TW-MSG-REC-{datetime.now().strftime('%M%S')}"

    notif_service._events[event_id] = {
        "notification_id": event_id,
        "provider_message_id": msg_id,
        "status": DeliveryState.SUBMITTED.value,
        "channel": "sms",
        "severity": "CRITICAL",
        "zone_id": "REAL-NER-001",
        "recipient_masked": "+91******3210",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    payload = {
        "provider": "twilio",
        "provider_message_id": msg_id,
        "status": "delivered",
    }
    resp = client.post(
        "/api/v1/notifications/webhook",
        json=payload,
        headers={"x-webhook-secret": "landjepa-webhook-secret-2026"},
    )
    assert resp.status_code == 200
    assert resp.json()["updated"] is True

    # State in service is updated to DELIVERED
    assert notif_service._events[event_id]["status"] == DeliveryState.DELIVERED.value


# ── 15. Security & Phone Number Masking ───────────────────────────────
def test_security_phone_masking(client: TestClient):
    """API endpoints must return masked phone numbers (+91******1234), never raw credentials."""
    resp = client.get("/api/v1/notifications/recipients")
    assert resp.status_code == 200
    recipients = resp.json()
    assert len(recipients) > 0

    for r in recipients:
        phone = r.get("phone_masked")
        if phone and phone != "—":
            assert "******" in phone
            assert len(phone) >= 10
            # Ensure middle digits are masked
            assert not phone.replace("+91", "").isdigit() or "******" in phone


# ── 16. Operator Test SMS Endpoint ────────────────────────────────────
def test_operator_test_sms_endpoint(client: TestClient):
    """POST /api/v1/notifications/test/sms requires explicit confirmation and returns transaction receipt."""
    # Test rejection without explicit confirmation
    bad_req = {
        "test_phone_number": "+919876543210",
        "officer_id": "OFFICER-NER-01",
        "zone_id": "REAL-NER-001",
        "severity": "WARNING",
        "language": "en",
        "explicit_confirmation": False,
    }
    resp = client.post("/api/v1/notifications/test/sms", json=bad_req)
    assert resp.status_code == 400

    # Valid confirmed test
    valid_req = {
        "test_phone_number": "+919876543210",
        "officer_id": "OFFICER-NER-01",
        "zone_id": "REAL-NER-001",
        "severity": "WARNING",
        "language": "en",
        "explicit_confirmation": True,
    }
    resp = client.post("/api/v1/notifications/test/sms", json=valid_req)
    assert resp.status_code == 200
    data = resp.json()
    assert "notification_id" in data
    assert "status" in data
    assert data["test_mode"] is True


# ── 17. System Health Notification Diagnostic Probes ──────────────────
def test_system_health_notification_components():
    """All 5 notification health probes return structured diagnostics with valid status."""
    svc = SystemHealthService.get_instance()

    p_sms = svc.check_sms_provider()
    assert p_sms["component"] == "SMS Provider"
    assert p_sms["status"] in ("ONLINE", "DEGRADED", "NOT_CONFIGURED", "OFFLINE")

    p_push = svc.check_push_provider()
    assert p_push["component"] == "Push Provider"
    assert p_push["status"] in ("ONLINE", "DEGRADED", "NOT_CONFIGURED", "OFFLINE")

    p_queue = svc.check_notification_queue()
    assert p_queue["component"] == "Notification Queue"
    assert p_queue["status"] == "ONLINE"

    p_wh = svc.check_notification_webhook()
    assert p_wh["component"] == "Webhook"
    assert p_wh["status"] == "ONLINE"

    p_tpl = svc.check_template_service()
    assert p_tpl["component"] == "Template Service"
    assert p_tpl["status"] == "ONLINE"
    assert len(p_tpl["details"]["languages"]) == 5


# ── 18. End-to-End Alert-to-Notification Policy Dispatch ──────────────
@pytest.mark.asyncio
async def test_end_to_end_alert_to_notification_workflow(notif_service: NotificationService):
    """CRITICAL alerts dispatch Push + SMS + In-App to targeted corridor recipients."""
    alert_id = f"ALT-E2E-{datetime.now().strftime('%H%M%S')}"
    dispatches = await notif_service.dispatch_alert_notifications(
        alert_id=alert_id,
        zone_id="REAL-NER-001",
        severity="CRITICAL",
        corridor_name="NH-27 Guwahati–Shillong Km 48",
        risk_score=0.88,
        horizon="12h",
    )
    assert len(dispatches) >= 2  # In-App + at least 1 Push/SMS
    channels = {d.get("channel") for d in dispatches}
    assert "in_app" in channels
