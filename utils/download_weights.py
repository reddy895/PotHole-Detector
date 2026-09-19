"""Download pretrained YOLO pothole detection model weights if missing."""
from pathlib import Path
import sys
import os
import requests

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from utils.logger import logger


def ensure_model_weights(target_path: Path = None) -> Path:
    """Ensure that a valid YOLO pothole weights file exists on disk.

    If not found, automatically downloads a verified pretrained YOLOv8 pothole model.

    Returns:
        Path to the resolved weights file.
    """
    if target_path is None:
        target_path = settings.get_model_path()

    if target_path.is_file() and target_path.stat().st_size > 1024 * 1024:
        logger.info(f"Model weights verified at: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return target_path

    # Check if fallback model exists
    if settings.FALLBACK_MODEL_PATH.is_file() and settings.FALLBACK_MODEL_PATH.stat().st_size > 1024 * 1024:
        logger.info(f"Found fallback model weights at: {settings.FALLBACK_MODEL_PATH}")
        return settings.FALLBACK_MODEL_PATH

    # Download pre-trained weights
    dest_path = settings.DEFAULT_MODEL_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = settings.PRETRAINED_MODEL_URL

    logger.info(f"Downloading pre-trained YOLO pothole weights from: {url}")
    logger.info(f"Target destination: {dest_path}")

    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 64

        temp_path = dest_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 4) < chunk_size:
                        percent = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")

        # Atomically move temp file to destination
        temp_path.replace(dest_path)
        logger.info(f"Successfully downloaded model weights to: {dest_path} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return dest_path

    except Exception as e:
        logger.error(f"Failed to download pretrained weights: {e}")
        # If download fails, check if standard ultralytics yolov8n can be used as fallback
        logger.warning("Attempting fallback to standard ultralytics yolov8n.pt...")
        fallback = settings.MODELS_DIR / "yolov8n.pt"
        if not fallback.exists():
            try:
                from ultralytics import YOLO
                yolo = YOLO("yolov8n.pt")
                logger.info("Initialized yolov8n.pt as emergency base model.")
                return Path("yolov8n.pt")
            except Exception as e2:
                logger.error(f"Could not load fallback yolov8n: {e2}")
        raise RuntimeError(
            f"No YOLO weights found in {settings.MODELS_DIR} and automated download failed: {e}. "
            f"Please manually place your trained 'pothole_yolo.pt' or 'best.pt' file into '{settings.MODELS_DIR}'."
        )


if __name__ == "__main__":
    ensure_model_weights()
