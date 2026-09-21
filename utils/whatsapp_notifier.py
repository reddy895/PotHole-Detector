"""WhatsApp Bot Notification Manager for AI Pothole Detection System.

Handles starting the Node.js WhatsApp microservice, monitoring authentication,
and automatically dispatching road hazard alerts with image snapshots to authorities.
"""
from typing import Optional, Dict, Any, Set
from pathlib import Path
import subprocess
import threading
import time
import cv2
import numpy as np
import requests

from config import config

# Ensure project root is available
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WhatsAppNotifier:
    """Manages WhatsApp Web client lifecycle and automated hazard dispatches."""

    def __init__(
        self,
        authority_phone: Optional[str] = None,
        enabled: bool = False,
        min_severity: str = "Medium",
        cooldown_seconds: float = 25.0,
        port: int = 5005,
    ) -> None:
        self.enabled = enabled
        raw_phone = authority_phone or config.WHATSAPP_AUTHORITY_PHONE
        if raw_phone:
            cleaned = "".join(c for c in raw_phone if c.isdigit() or c == "+")
            if cleaned.isdigit() and len(cleaned) == 10:
                cleaned = f"+91{cleaned}"
            self.authority_phone = cleaned
        else:
            self.authority_phone = "+919591152862"

        self.min_severity = min_severity
        self.cooldown_seconds = cooldown_seconds
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._process: Optional[subprocess.Popen] = None
        self._last_alert_time: float = 0.0
        self._alerted_track_ids: Set[int] = set()
        self._lock = threading.Lock()

        self.alerts_dir = config.OUTPUTS_DIR / "whatsapp_alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        """True if notifier is enabled with a valid authority recipient."""
        return bool(self.enabled and self.authority_phone and len(self.authority_phone.strip()) >= 8)

    def is_service_running(self) -> bool:
        """Check if the WhatsApp microservice is currently running and responding."""
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Fetch current authentication and readiness state from microservice."""
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=1.5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"status": "offline", "ready": False, "user": None}

    def start_service(self) -> bool:
        """Spawn the background Node.js WhatsApp microservice if not already running."""
        if self.is_service_running():
            return True

        bot_script = _PROJECT_ROOT / "whatsapp_bot" / "server.js"
        if not bot_script.is_file():
            print(f"[WHATSAPP ERROR] Server script not found at: {bot_script}")
            return False

        print(f"[WHATSAPP] Launching WhatsApp bot on port {self.port}...")
        env = dict(subprocess.os.environ)
        env["WHATSAPP_PORT"] = str(self.port)
        env["NODE_ENV"] = "production"

        try:
            self._process = subprocess.Popen(
                ["node", str(bot_script)],
                cwd=str(_PROJECT_ROOT / "whatsapp_bot"),
                env=env,
            )
            # Wait up to 3s for HTTP server
            for _ in range(15):
                time.sleep(0.2)
                if self.is_service_running():
                    return True
        except Exception as e:
            print(f"[WHATSAPP ERROR] Failed to start microservice: {e}")
            return False

        return self.is_service_running()

    def wait_for_authentication(self, timeout_seconds: int = 3) -> bool:
        """Quickly check readiness; never block or delay detection pipeline."""
        start_t = time.time()
        while (time.time() - start_t) < timeout_seconds:
            status = self.get_status()
            if status.get("ready"):
                user_num = status.get("user") or "Authorized User"
                print(f"[WHATSAPP CONNECTED] Linked to account: +{user_num}")
                return True
            time.sleep(0.3)

        return False

    def send_pothole_alert(
        self,
        detection,  # PotholeDetection
        frame: np.ndarray,
        source_name: str = "Live Feed",
        location_desc: Optional[str] = None,
        async_dispatch: bool = False,
    ) -> bool:
        """Format and dispatch an urgent hazard alert to the municipal authority.

        Args:
            detection: PotholeDetection instance.
            frame: Raw or annotated BGR frame.
            source_name: Name of video/camera source.
            location_desc: Optional GPS / route description.
            async_dispatch: If True, deliver HTTP request in background thread to avoid CV lag.

        Returns:
            True if alert was accepted for dispatch.
        """
        if not self.is_configured:
            return False

        # Severity filter check
        severity_hierarchy = {"Low": 1, "Medium": 2, "High": 3}
        det_rank = severity_hierarchy.get(detection.severity, 1)
        min_rank = severity_hierarchy.get(self.min_severity, 1)
        if det_rank < min_rank:
            return False  # Skip minor severity potholes

        with self._lock:
            now = time.time()
            # Cooldown check: prevent duplicate spamming
            if (now - self._last_alert_time) < self.cooldown_seconds:
                return False

            track_id = getattr(detection, "track_id", None)
            if track_id is not None and track_id in self._alerted_track_ids:
                return False

            # Mark as alerted
            self._last_alert_time = now
            if track_id is not None:
                self._alerted_track_ids.add(track_id)

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        loc_str = location_desc if location_desc else f"Dashcam Source ({source_name})"
        message = (
            f"🚨 *MUNICIPAL ROAD HAZARD: POTHOLE DETECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Location/Source:* {loc_str}\n"
            f"⚠️ *Severity:* *{detection.severity.upper()}*\n"
            f"🎯 *Confidence:* {detection.confidence * 100:.1f}%\n"
            f"🆔 *Pothole Track ID:* #{track_id or 'N/A'}\n"
            f"🕒 *Detection Time:* {timestamp_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Automated AI Pothole Alert_"
        )

        payload = {
            "phone": self.authority_phone,
            "message": message,
        }

        def _do_send():
            try:
                resp = requests.post(f"{self.base_url}/send", json=payload, timeout=5.0)
                if resp.status_code == 200 and resp.json().get("success"):
                    print(f"\n[WHATSAPP ALERT DISPATCHED] Pothole #{track_id or 'N/A'} sent to {self.authority_phone}")
                else:
                    err = resp.json().get("error") if resp.status_code == 200 else resp.text
                    print(f"\n[WHATSAPP ALERT FAILED] {err}")
            except Exception as e:
                print(f"\n[WHATSAPP ALERT ERROR] Failed to deliver alert: {e}")

        if async_dispatch:
            thread = threading.Thread(target=_do_send, daemon=True)
            thread.start()
            return True
        else:
            _do_send()
            return True

    def stop_service(self) -> None:
        """Terminate the microservice process cleanly."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
