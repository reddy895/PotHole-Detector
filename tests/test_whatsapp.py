"""Unit tests for WhatsApp notifier configuration, throttling, and state."""
import pytest
from utils.whatsapp_notifier import WhatsAppNotifier


def test_whatsapp_notifier_initialization():
    notifier = WhatsAppNotifier(
        authority_phone="+919876543210",
        enabled=True,
        min_severity="High",
        cooldown_seconds=30.0,
    )
    assert notifier.is_configured is True
    assert notifier.authority_phone == "+919876543210"
    assert notifier.min_severity == "High"
    assert notifier.cooldown_seconds == 30.0


def test_whatsapp_notifier_unconfigured():
    notifier = WhatsAppNotifier(enabled=False, authority_phone=None)
    assert notifier.is_configured is False


def test_whatsapp_status_when_offline():
    notifier = WhatsAppNotifier(port=59999)  # unreachable port
    assert notifier.is_service_running() is False
    status = notifier.get_status()
    assert status["ready"] is False
    assert status["status"] == "offline"
