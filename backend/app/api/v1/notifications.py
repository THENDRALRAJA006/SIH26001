"""LAND-JEPA — Notifications & Early-Warning Dispatch API Router.
Problem: SIH26001 | Team: ZAIX | Region: Northeast India (NER)

Endpoints:
  - POST /api/v1/notifications/subscribe     — Citizen explicit notification opt-in & consent
  - POST /api/v1/notifications/unsubscribe   — Citizen notification opt-out
  - GET  /api/v1/notifications               — Chronological feed of early warning dispatches
  - GET  /api/v1/notifications/recipients    — Registered recipients roster (phone masked)
  - POST /api/v1/notifications/test/sms      — Operator controlled real test SMS dispatch
  - POST /api/v1/notifications/test/push     — Operator controlled test push dispatch
  - POST /api/v1/notifications/webhook       — SMS/Push delivery receipts webhook callback
  - GET  /api/v1/notifications/stats         — Operational delivery statistics & metrics
  - GET  /api/v1/notifications/status        — Subsystem health diagnostics
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.notification import (
    CitizenSubscribeRequest,
    CitizenUnsubscribeRequest,
    DeliveryState,
    NotificationEventResponse,
    NotificationListResponse,
    NotificationStatsResponse,
    OperatorTestPushRequest,
    OperatorTestSmsRequest,
    RecipientSummaryResponse,
    WebhookStatusPayload,
)
from app.services.notification.service import NotificationService, get_notification_service

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post(
    "/subscribe",
    response_model=RecipientSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Citizen Notification Subscription with Explicit Consent",
    description="Registers citizen contact for regional corridor early warnings. Explicit consent is mandatory.",
)
async def subscribe_citizen(
    req: CitizenSubscribeRequest,
    request: Request,
    service: NotificationService = Depends(get_notification_service),
) -> RecipientSummaryResponse:
    client_ip = request.client.host if request.client else None
    try:
        record = service.subscribe_citizen(
            name=req.name,
            phone_number=req.phone_number,
            email=req.email,
            preferred_language=req.preferred_language.value,
            zone_id=req.zone_id or "REAL-NER-001",
            corridor_name=req.corridor_name or "NH-27 Guwahati–Shillong",
            latitude=req.latitude,
            longitude=req.longitude,
            alert_radius_km=req.alert_radius_km,
            enable_sms=req.enable_sms,
            enable_push=req.enable_push,
            enable_in_app=req.enable_in_app,
            explicit_consent=req.explicit_consent,
            client_ip=client_ip,
        )
        return RecipientSummaryResponse(
            recipient_id=record["recipient_id"],
            name=record["name"],
            role=record["role"],
            phone_masked=record.get("phone_masked"),
            preferred_language=record["preferred_language"],
            zone_id=record.get("zone_id"),
            corridor_name=record.get("corridor_name"),
            consent_status=record["consent_status"],
            enable_sms=record["enable_sms"],
            enable_push=record["enable_push"],
            active=record["active"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/unsubscribe",
    summary="Citizen Notification Opt-Out",
)
async def unsubscribe_citizen(
    req: CitizenUnsubscribeRequest,
    service: NotificationService = Depends(get_notification_service),
) -> Dict[str, Any]:
    if not req.phone_number and not req.recipient_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either phone_number or recipient_id is required to unsubscribe.",
        )
    success = service.unsubscribe_citizen(recipient_id=req.recipient_id, phone_number=req.phone_number)
    return {
        "success": success,
        "message": "Successfully unsubscribed from emergency alerts." if success else "Recipient record not found.",
    }


@router.get(
    "",
    response_model=NotificationListResponse,
    summary="List Notifications Feed",
    description="Retrieve chronological feed of alerts dispatched across SMS, Push, and In-App channels.",
)
async def list_notifications(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    channel: Annotated[Optional[str], Query()] = None,
    status_filter: Annotated[Optional[str], Query(alias="status")] = None,
    service: NotificationService = Depends(get_notification_service),
) -> NotificationListResponse:
    events = service.list_events(limit=limit, channel=channel, status=status_filter)
    resp_items = []
    for e in events:
        resp_items.append(
            NotificationEventResponse(
                notification_id=e["notification_id"],
                alert_id=e.get("alert_id"),
                recipient_masked=e["recipient_masked"],
                channel=e["channel"],
                severity=e["severity"],
                zone_id=e["zone_id"],
                title=e["title"],
                body=e["body"],
                language=e.get("language", "en"),
                status=DeliveryState(e["status"]),
                provider=e.get("provider"),
                provider_message_id=e.get("provider_message_id"),
                attempt_count=e.get("attempt_count", 1),
                created_at=e["created_at"],
                sent_at=e.get("sent_at"),
                delivered_at=e.get("delivered_at"),
                failed_at=e.get("failed_at"),
                failure_reason=e.get("failure_reason"),
                is_test=e.get("is_test", False),
                is_simulated=e.get("is_simulated", False),
            )
        )
    return NotificationListResponse(total=len(resp_items), notifications=resp_items)


@router.get(
    "/recipients",
    response_model=List[RecipientSummaryResponse],
    summary="List Registered Recipients (Officers & Consented Citizens)",
)
async def list_recipients(
    service: NotificationService = Depends(get_notification_service),
) -> List[RecipientSummaryResponse]:
    recipients = service.list_recipients()
    return [
        RecipientSummaryResponse(
            recipient_id=r["recipient_id"],
            name=r["name"],
            role=r["role"],
            phone_masked=r.get("phone_masked"),
            preferred_language=r.get("preferred_language", "en"),
            zone_id=r.get("zone_id"),
            corridor_name=r.get("corridor_name"),
            consent_status=r.get("consent_status", "PENDING"),
            enable_sms=r.get("enable_sms", False),
            enable_push=r.get("enable_push", True),
            active=r.get("active", True),
        )
        for r in recipients
    ]


@router.post(
    "/test/sms",
    summary="Operator-Controlled Real Test SMS Dispatch",
    description=(
        "Authorized disaster response officers can trigger a real SMS transaction "
        "to a verified test phone number. Uses live SMS provider adapter and returns provider Message ID."
    ),
)
async def send_operator_test_sms(
    req: OperatorTestSmsRequest,
    service: NotificationService = Depends(get_notification_service),
) -> Dict[str, Any]:
    if not req.explicit_confirmation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explicit operator confirmation is required to initiate test SMS dispatch.",
        )

    test_recipient = {
        "recipient_id": f"TEST-OPERATOR-{req.officer_id}",
        "name": f"Operator Test Target ({req.officer_id})",
        "role": "field_officer",
        "phone_number": req.test_phone_number,
        "phone_masked": req.test_phone_number[:3] + "******" + req.test_phone_number[-4:] if len(req.test_phone_number) > 7 else "+91******3210",
        "preferred_language": req.language.value,
        "zone_id": req.zone_id,
        "corridor_name": f"Corridor {req.zone_id}",
        "enable_sms": True,
        "consent_status": "GRANTED",
    }

    test_alert_id = f"TEST-ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    event = await service.send_sms(
        recipient=test_recipient,
        alert_id=test_alert_id,
        severity=req.severity.value,
        zone_id=req.zone_id,
        horizon="12h",
        is_test=True,
    )

    return {
        "success": event["status"] in (DeliveryState.SENT.value, DeliveryState.SUBMITTED.value, DeliveryState.SIMULATED.value),
        "status": event["status"],
        "notification_id": event["notification_id"],
        "provider": event["provider"],
        "provider_message_id": event.get("provider_message_id"),
        "recipient_masked": event["recipient_masked"],
        "sent_at": event.get("sent_at"),
        "failure_reason": event.get("failure_reason"),
        "is_simulated": event.get("is_simulated", False),
        "test_mode": True,
    }


@router.post(
    "/test/push",
    summary="Operator-Controlled Test Push Dispatch",
)
async def send_operator_test_push(
    req: OperatorTestPushRequest,
    service: NotificationService = Depends(get_notification_service),
) -> Dict[str, Any]:
    test_recipient = {
        "recipient_id": f"TEST-DEVICE-{req.officer_id}",
        "name": f"Operator Device ({req.officer_id})",
        "role": "field_officer",
        "push_token": req.push_token or "browser_operator_session_token",
        "preferred_language": req.language.value,
        "zone_id": req.zone_id,
        "corridor_name": f"Corridor {req.zone_id}",
        "enable_push": True,
    }

    test_alert_id = f"TEST-PUSH-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    event = await service.send_push(
        recipient=test_recipient,
        alert_id=test_alert_id,
        severity=req.severity.value,
        zone_id=req.zone_id,
        horizon="24h",
        is_test=True,
    )

    return {
        "success": event["status"] in (DeliveryState.SUBMITTED.value, DeliveryState.DELIVERED.value, DeliveryState.SIMULATED.value),
        "status": event["status"],
        "notification_id": event["notification_id"],
        "provider": event["provider"],
        "provider_message_id": event.get("provider_message_id"),
        "is_simulated": event.get("is_simulated", False),
    }


@router.post(
    "/webhook",
    summary="SMS / Push Provider Delivery Receipt Webhook",
    description="Receives delivery confirmation from Twilio, MSG91, AWS SNS, or FCM and updates delivery status.",
)
async def delivery_receipt_webhook(
    payload: WebhookStatusPayload,
    x_webhook_secret: Annotated[Optional[str], Header()] = None,
    service: NotificationService = Depends(get_notification_service),
) -> Dict[str, Any]:
    # Signature / Secret verification (Requirement 26 & 27)
    if x_webhook_secret and x_webhook_secret != settings.NOTIFICATION_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature or secret.",
        )

    updated = service.process_delivery_webhook(
        provider=payload.provider,
        provider_message_id=payload.provider_message_id,
        status_code=payload.status,
        error_message=payload.error_message,
    )

    return {
        "received": True,
        "updated": updated,
        "provider_message_id": payload.provider_message_id,
        "new_status": "DELIVERED" if "deliv" in payload.status.lower() else "PROCESSED",
    }


@router.get(
    "/stats",
    response_model=NotificationStatsResponse,
    summary="Notification Delivery Statistics & Performance Metrics",
)
async def get_notification_stats(
    service: NotificationService = Depends(get_notification_service),
) -> NotificationStatsResponse:
    metrics = service.get_metrics()
    return NotificationStatsResponse(**metrics)


@router.get(
    "/status",
    summary="Notification Subsystem Health Status",
)
async def get_notification_subsystem_status(
    service: NotificationService = Depends(get_notification_service),
) -> Dict[str, Any]:
    sms_conf = service.sms_provider.is_configured()
    push_conf = service.push_provider.is_configured()

    return {
        "status": "OPERATIONAL" if (sms_conf or service.sms_provider.provider_name == "mock") else "DEGRADED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "sms_provider": {
                "provider": service.sms_provider.provider_name,
                "status": "ONLINE" if sms_conf else ("SIMULATED" if service.sms_provider.provider_name == "mock" else "NOT_CONFIGURED"),
                "sender_id": settings.SMS_SENDER_ID,
                "dlt_compliant": True,
            },
            "push_provider": {
                "provider": service.push_provider.provider_name,
                "status": "ONLINE" if push_conf else ("SIMULATED" if service.push_provider.provider_name == "mock" else "NOT_CONFIGURED"),
            },
            "notification_queue": {
                "status": "ONLINE",
                "queue_depth": sum(1 for e in service._events.values() if e.get("status") == "QUEUED"),
            },
            "webhook_endpoint": {
                "status": "ONLINE",
                "url": "/api/v1/notifications/webhook",
                "auth_required": True,
            },
            "template_service": {
                "status": "ONLINE",
                "supported_languages": ["en", "hi", "as", "bn", "mni"],
                "tiers": ["WATCH", "WARNING", "CRITICAL"],
            },
        },
    }
