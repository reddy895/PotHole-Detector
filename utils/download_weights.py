"""Download pretrained YOLO pothole detection model weights if missing.

Run directly:
    python utils/download_weights.py
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from utils.logger import logger


# Public Roboflow-hosted pothole YOLO weights (YOLOv8n trained on pothole dataset)
_PRETRAINED_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
)


def ensure_model_weights(target_path: Path = None) -> Path:
    """Ensure that a valid YOLO pothole weights file exists on disk.

    If the weights are not found at *target_path*, falls back to downloading
    the standard YOLOv8n base model so the system can start without manual
    weight placement.

    Args:
        target_path: Override path for the model file. Defaults to
            ``config.MODEL_PATH``.

    Returns:
        Path to the resolved weights file.

    Raises:
        RuntimeError: If no weights can be found or downloaded.
    """
    if target_path is None:
        target_path = config.MODEL_PATH

    # 1. Primary path — custom trained weights
    if target_path.is_file() and target_path.stat().st_size > 1024 * 1024:
        logger.info(
            "Model weights verified at: %s (%.1f MB)",
            target_path,
            target_path.stat().st_size / (1024 * 1024),
        )
        return target_path

    logger.warning("Model weights not found at: %s", target_path)

    # 2. Fallback — download YOLOv8n base weights via ultralytics auto-download
    dest_path = config.MODELS_DIR / "pothole.pt"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Attempting to download base YOLO weights from: %s", _PRETRAINED_URL)
    logger.info("Destination: %s", dest_path)

    try:
        import requests

        response = requests.get(_PRETRAINED_URL, stream=True, timeout=120, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 64  # 64 KB chunks

        temp_path = dest_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 4) < chunk_size:
                        pct = (downloaded / total_size) * 100
                        logger.info(
                            "Download progress: %.1f%% (%.1f / %.1f MB)",
                            pct,
                            downloaded / (1024 * 1024),
                            total_size / (1024 * 1024),
                        )

        # Atomic rename: avoids leaving a corrupt partial file on disk
        temp_path.replace(dest_path)
        logger.info(
            "Downloaded model weights to: %s (%.1f MB)",
            dest_path,
            dest_path.stat().st_size / (1024 * 1024),
        )
        return dest_path

    except Exception as exc:
        logger.error("Failed to download weights: %s", exc)

        # 3. Last resort — let ultralytics pull yolov8n.pt itself
        logger.warning("Falling back to ultralytics auto-download of yolov8n.pt ...")
        try:
            from ultralytics import YOLO
            YOLO("yolov8n.pt")  # triggers ultralytics built-in download
            fallback_path = Path("yolov8n.pt")
            logger.info("Fallback model available at: %s", fallback_path)
            return fallback_path
        except Exception as exc2:
            raise RuntimeError(
                f"No YOLO weights found and all download attempts failed.\n"
                f"Place your trained 'pothole.pt' into '{config.MODELS_DIR}' manually.\n"
                f"Primary error: {exc}\n"
                f"Fallback error: {exc2}"
            ) from exc


if __name__ == "__main__":
    resolved = ensure_model_weights()
    print(f"[OK] Model weights ready at: {resolved}")
