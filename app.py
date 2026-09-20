"""Flask Web Application for AI Pothole Detection System.

Provides two modes:
  1. Live Video  - Real-time detection via webcam stream (MJPEG)
  2. Video Feed  - Upload a video file and stream annotated output

REST API endpoints:
  GET  /api/system  — hardware & model info
  POST /api/conf    — live confidence threshold update
"""
import sys
import os
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import (
    Flask,
    Response,
    render_template,
    request,
    jsonify,
    send_from_directory,
)
import cv2
import numpy as np

from config import config
from detector import PotholeDetector
from utils.helpers import get_device_info
from utils.whatsapp_notifier import WhatsAppNotifier

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Shared WhatsApp Notifier instance
whatsapp_notifier = WhatsAppNotifier(
    authority_phone=config.WHATSAPP_AUTHORITY_PHONE,
    enabled=config.WHATSAPP_ENABLED,
    min_severity=config.WHATSAPP_MIN_SEVERITY,
    port=config.WHATSAPP_PORT,
)


# ---------------------------------------------------------------------------
# Global detector (lazy-loaded)
# ---------------------------------------------------------------------------
_detector = None
_detector_lock = threading.Lock()


def get_detector():
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = PotholeDetector()
    return _detector


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
webcam_active = False
webcam_lock = threading.Lock()
upload_stream_path = None


# ---------------------------------------------------------------------------
# MJPEG helpers
# ---------------------------------------------------------------------------

def _encode_jpeg(frame):
    """Encode BGR frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes()


def _mjpeg_boundary(frame_bytes):
    """Wrap frame bytes in MJPEG boundary."""
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
    )


# ---------------------------------------------------------------------------
# Webcam stream generator
# ---------------------------------------------------------------------------

def webcam_generator():
    global webcam_active
    detector = get_detector()
    cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX, cv2.CAP_ANY)

    if not cap.isOpened():
        err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_frame, "Camera not available", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 60, 220), 2, cv2.LINE_AA)
        yield _mjpeg_boundary(_encode_jpeg(err_frame))
        return

    try:
        while True:
            with webcam_lock:
                active = webcam_active
            if not active:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            result = detector.detect(frame)
            annotated = detector.draw_annotations(frame, result, show_hud=True)

            # Automated WhatsApp alert dispatch for high-severity potholes
            if whatsapp_notifier.is_configured and result.count > 0:
                for det in result.detections:
                    if det.severity in ("Medium", "High"):
                        whatsapp_notifier.send_pothole_alert(det, annotated, source_name="Webcam Stream")
                        break

            yield _mjpeg_boundary(_encode_jpeg(annotated))
    finally:
        cap.release()



# ---------------------------------------------------------------------------
# Video file stream generator
# ---------------------------------------------------------------------------

def video_file_generator(video_path):
    detector = get_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_frame, "Cannot open video", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 60, 220), 2, cv2.LINE_AA)
        yield _mjpeg_boundary(_encode_jpeg(err_frame))
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / video_fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            result = detector.detect(frame)
            annotated = detector.draw_annotations(frame, result, show_hud=True)

            # Automated WhatsApp alert dispatch for high-severity potholes
            if whatsapp_notifier.is_configured and result.count > 0:
                for det in result.detections:
                    if det.severity in ("Medium", "High"):
                        whatsapp_notifier.send_pothole_alert(
                            det, annotated, source_name=Path(video_path).name
                        )
                        break

            yield _mjpeg_boundary(_encode_jpeg(annotated))
            time.sleep(delay)
    finally:
        cap.release()



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/webcam/start", methods=["POST"])
def webcam_start():
    global webcam_active
    with webcam_lock:
        webcam_active = True
    return jsonify({"status": "started"})


@app.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    global webcam_active
    with webcam_lock:
        webcam_active = False
    return jsonify({"status": "stopped"})


@app.route("/stream/webcam")
def stream_webcam():
    global webcam_active
    with webcam_lock:
        webcam_active = True
    return Response(
        webcam_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/upload/video", methods=["POST"])
def upload_video():
    global upload_stream_path
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    safe_name = f.filename.replace(" ", "_")
    save_path = UPLOAD_DIR / safe_name
    f.save(str(save_path))
    upload_stream_path = str(save_path)
    return jsonify({"status": "uploaded", "filename": safe_name})


@app.route("/stream/video")
def stream_video():
    global upload_stream_path
    if not upload_stream_path:
        return jsonify({"error": "No video uploaded"}), 400
    return Response(
        video_file_generator(upload_stream_path),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(config.OUTPUTS_DIR), filename)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/system", methods=["GET"])
def api_system():
    """Return hardware and model information as JSON.

    Response schema::

        {
          "model_path": "models/pothole.pt",
          "model_exists": true,
          "device": "cuda:0",
          "is_cuda": true,
          "device_name": "NVIDIA GeForce RTX 2050",
          "cuda_available": true,
          "vram_mb": 4096,
          "confidence_threshold": 0.35,
          "iou_threshold": 0.45,
          "image_size": 640
        }
    """
    hw = get_device_info(config.FORCE_DEVICE)
    detector = get_detector() if _detector is not None else None
    conf = detector.confidence_threshold if detector else config.CONFIDENCE_THRESHOLD

    return jsonify({
        "model_path": str(config.MODEL_PATH),
        "model_exists": config.MODEL_PATH.is_file(),
        "device": hw["device"],
        "is_cuda": hw["is_cuda"],
        "device_name": hw["device_name"],
        "cuda_available": hw["cuda_available"],
        "vram_mb": hw["vram_mb"],
        "confidence_threshold": conf,
        "iou_threshold": config.IOU_THRESHOLD,
        "image_size": config.IMAGE_SIZE,
    })


@app.route("/api/conf", methods=["POST"])
def api_set_confidence():
    """Update the running detector's confidence threshold without restart.

    Request body (JSON)::

        {"threshold": 0.45}

    Response::

        {"status": "ok", "confidence_threshold": 0.45}

    Allows live tuning from the dashboard or external tooling — useful when
    switching between road surfaces with different detection difficulty.
    """
    data = request.get_json(silent=True) or {}
    threshold = data.get("threshold")

    if threshold is None:
        return jsonify({"error": "Missing 'threshold' field in JSON body"}), 400

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({"error": "'threshold' must be a float"}), 400

    if not (0.0 < threshold < 1.0):
        return jsonify({"error": "'threshold' must be between 0.0 and 1.0 (exclusive)"}), 400

    detector = get_detector()
    detector.confidence_threshold = threshold
    return jsonify({"status": "ok", "confidence_threshold": threshold})


# ---------------------------------------------------------------------------
# WhatsApp Bot REST Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/whatsapp/status", methods=["GET"])
def api_whatsapp_status():
    """Retrieve WhatsApp microservice connectivity, authentication, and QR status."""
    is_running = whatsapp_notifier.is_service_running()
    details = whatsapp_notifier.get_status() if is_running else {}
    qr_img = config.OUTPUTS_DIR / "whatsapp_qr.png"
    return jsonify({
        "service_running": is_running,
        "enabled": whatsapp_notifier.enabled,
        "is_configured": whatsapp_notifier.is_configured,
        "authority_phone": whatsapp_notifier.authority_phone,
        "min_severity": whatsapp_notifier.min_severity,
        "status": details.get("status", "offline"),
        "ready": details.get("ready", False),
        "connected_user": details.get("user"),
        "qr_available": details.get("qr_available", False) or qr_img.is_file(),
    })


@app.route("/api/whatsapp/start", methods=["POST"])
def api_whatsapp_start():
    """Start the WhatsApp bot microservice in background."""
    started = whatsapp_notifier.start_service()
    return jsonify({
        "success": started,
        "running": whatsapp_notifier.is_service_running(),
        "status": whatsapp_notifier.get_status(),
    })


@app.route("/api/whatsapp/configure", methods=["POST"])
def api_whatsapp_configure():
    """Configure authority recipient phone number and toggle alerts."""
    data = request.get_json(silent=True) or {}
    phone = data.get("phone")
    enabled = data.get("enabled")
    min_severity = data.get("min_severity")

    if phone is not None:
        whatsapp_notifier.authority_phone = str(phone).strip()
    if enabled is not None:
        whatsapp_notifier.enabled = bool(enabled)
    if min_severity in ("Low", "Medium", "High"):
        whatsapp_notifier.min_severity = min_severity

    # Automatically start service if enabled
    if whatsapp_notifier.enabled and not whatsapp_notifier.is_service_running():
        whatsapp_notifier.start_service()

    return jsonify({
        "status": "ok",
        "enabled": whatsapp_notifier.enabled,
        "authority_phone": whatsapp_notifier.authority_phone,
        "min_severity": whatsapp_notifier.min_severity,
        "is_configured": whatsapp_notifier.is_configured,
    })


@app.route("/api/whatsapp/qr.png", methods=["GET"])
def api_whatsapp_qr_image():
    """Serve the latest WhatsApp authentication QR code image."""
    qr_path = config.OUTPUTS_DIR / "whatsapp_qr.png"
    if qr_path.is_file():
        return send_from_directory(str(config.OUTPUTS_DIR), "whatsapp_qr.png", mimetype="image/png")
    return jsonify({"error": "QR code image not generated yet. Start service first."}), 404



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  AI Pothole Detection - Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50)
    print("[INFO] Loading YOLO model...")
    try:
        get_detector()
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        print(f"[WARN] Model not loaded yet: {e}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
