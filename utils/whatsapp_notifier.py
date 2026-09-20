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
        self.authority_phone = authority_phone or config.WHATSAPP_AUTHORITY_PHONE
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
            resp = requests.get(f"{self.base_url}/status", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Fetch current authentication and readiness state from microservice."""
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=2.5)
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

        print(f"[WHATSAPP] Launching WhatsApp microservice on port {self.port}...")
        env = {
            "PATH": subprocess.os.environ.get("PATH", ""),
            "WHATSAPP_PORT": str(self.port),
            "NODE_ENV": "production",
        }

        # Run process interactively to show QR code in terminal stdout
        try:
            self._process = subprocess.Popen(
                ["node", str(bot_script)],
                cwd=str(_PROJECT_ROOT / "whatsapp_bot"),
                env=env,
            )
            # Wait up to 10s for the HTTP server to bind
            for _ in range(20):
                time.sleep(0.5)
                if self.is_service_running():
                    return True
        except Exception as e:
            print(f"[WHATSAPP ERROR] Failed to start microservice: {e}")
            return False

        return self.is_service_running()

    def wait_for_authentication(self, timeout_seconds: int = 90) -> bool:
        """Poll until WhatsApp Web QR code is scanned and client is ready."""
        print("[WHATSAPP] Checking authentication status...")
        start_t = time.time()
        qr_prompt_shown = False

        while (time.time() - start_t) < timeout_seconds:
            status = self.get_status()
            if status.get("ready"):
                user_num = status.get("user") or "Authorized User"
                print(f"\n[WHATSAPP CONNECTED] Successfully linked to WhatsApp account: +{user_num}")
                return True

            if status.get("status") == "qr_ready" and not qr_prompt_shown:
                qr_img = status.get("qr_image_path") or (_PROJECT_ROOT / "outputs" / "whatsapp_qr.png")
                print("\n" + "=" * 54)
                print("           WHATSAPP QR CODE READY FOR SCAN")
                print("=" * 54)
                print("Please scan the QR code above with your WhatsApp app:")
                print("  Open WhatsApp -> Settings -> Linked Devices -> Link a Device")
                if Path(qr_img).is_file():
                    print(f"QR Image also viewable at: {qr_img}")
                print("=" * 54 + "\n")
                qr_prompt_shown = True

            time.sleep(1.0)

        print("[WHATSAPP WARN] Authentication timed out. Alerts will be queued or skipped.")
        return False

    def send_pothole_alert(
        self,
        detection,  # PotholeDetection
        frame: np.ndarray,
        source_name: str = "Live Feed",
        location_desc: Optional[str] = None,
    ) -> bool:
        """Format and dispatch an urgent hazard alert to the municipal authority.

        Args:
            detection: PotholeDetection instance.
            frame: Raw or annotated BGR frame.
            source_name: Name of video/camera source.
            location_desc: Optional GPS / route description.

        Returns:
            True if message was accepted and sent by microservice.
        """
        if not self.is_configured:
            return False

        # Severity filter check
        severity_hierarchy = {"Low": 1, "Medium": 2, "High": 3}
        det_rank = severity_hierarchy.get(detection.severity, 1)
        min_rank = severity_hierarchy.get(self.min_severity, 2)
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

        # Create annotated snapshot of the frame for the authority
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        clean_time = time.strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"alert_{clean_time}_track{track_id or 'unknown'}.jpg"
        snapshot_path = self.alerts_dir / snapshot_name

        snapshot_frame = frame.copy()
        # Highlight pothole bounding box on snapshot
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        box_color = (0, 30, 220) if detection.severity == "High" else (0, 165, 255)
        cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), box_color, 3, cv2.LINE_AA)
        label = f"HAZARD: #{track_id or 1} {detection.label} [{detection.severity}]"
        cv2.putText(
            snapshot_frame, label, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(snapshot_path), snapshot_frame)

        # Build professional alert template
        loc_str = location_desc if location_desc else f"Dashcam Source ({source_name})"
        message = (
            f"🚨 *MUNICIPAL ROAD HAZARD REPORT: POTHOLE DETECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Location/Route:* {loc_str}\n"
            f"⚠️ *Hazard Severity:* *{detection.severity.upper()}*\n"
            f"🎯 *Detection Confidence:* {detection.confidence * 100:.1f}%\n"
            f"📐 *Relative Road Area:* {detection.area_ratio * 100:.2f}%\n"
            f"🆔 *Pothole Track ID:* #{track_id or 'N/A'}\n"
            f"🕒 *Detection Time:* {timestamp_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📸 _Annotated camera capture attached above._\n"
            f"_Automated dispatch generated by AI Pothole Detection System._"
        )

        payload = {
            "phone": self.authority_phone,
            "message": message,
            "image_path": str(snapshot_path),
        }

        try:
            resp = requests.post(f"{self.base_url}/send", json=payload, timeout=12.0)
            if resp.status_code == 200 and resp.json().get("success"):
                print(f"\n[WHATSAPP ALERT DISPATCHED] Pothole #{track_id} sent to authority ({self.authority_phone})")
                return True
            else:
                err = resp.json().get("error") if resp.status_code != 200 else "Unknown error"
                print(f"\n[WHATSAPP ALERT FAILED] {err}")
                return False
        except Exception as e:
            print(f"\n[WHATSAPP ALERT ERROR] Failed to deliver alert: {e}")
            return False

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
