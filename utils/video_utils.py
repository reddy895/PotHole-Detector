"""Helper utilities for video streams, terminal reporting, file I/O, and detection logging."""
from typing import Optional, Tuple
from pathlib import Path
import json
import sys
import time
import cv2
import numpy as np

# Ensure project root is on sys.path so `config` can be found when this
# module is imported standalone or as part of the `utils` package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import config


def print_banner(source_name: str, model_path: str, device_name: str) -> None:
    """Print standard system header banner to the terminal."""
    banner = f"""
========================================
      POTHOLE DETECTION SYSTEM
========================================
Model:  {model_path}
Device: {device_name}
Source: {source_name}
Status: Running
========================================
"""
    print(banner.strip())
    print()


def print_detection_status(
    count: int,
    highest_confidence: float,
    fps: float,
    throttle_interval: float = 0.3,
    last_print_time: Optional[float] = None,
) -> float:
    """Print real-time detection telemetry to terminal.

    Args:
        count: Number of detected potholes.
        highest_confidence: Highest confidence score (0.0 to 1.0).
        fps: Current smoothed frames per second.
        throttle_interval: Minimum seconds between prints.
        last_print_time: Timestamp of last print (None to always print).

    Returns:
        Updated print timestamp (float).
    """
    now = time.time()
    if last_print_time is not None and (now - last_print_time) < throttle_interval:
        return last_print_time

    if count > 0:
        conf_pct = highest_confidence * 100
        print(f"[POTHOLE DETECTED]")
        print(f"Count: {count}")
        print(f"Highest Confidence: {conf_pct:.1f}%")
        print(f"FPS: {fps:.1f}")
        print("-" * 25)
    else:
        print(f"[NO POTHOLE]")
        print(f"FPS: {fps:.1f}")
        print("-" * 25)

    return now


def save_annotated_image(image: np.ndarray, original_name: str) -> Path:
    """Save an annotated BGR frame to the outputs directory.

    Args:
        image: OpenCV BGR image array.
        original_name: Filename of the original input (used to build output name).

    Returns:
        Absolute path of the saved file.
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_stem = Path(original_name).stem
    out_path = config.OUTPUTS_DIR / f"annotated_{clean_stem}.jpg"
    cv2.imwrite(str(out_path), image)
    return out_path


def create_video_writer(
    output_filename: str,
    fps: float,
    frame_size: Tuple[int, int],
) -> Tuple[cv2.VideoWriter, Path]:
    """Create an OpenCV VideoWriter configured for MP4 output.

    Args:
        output_filename: Target video filename (relative to outputs/).
        fps: Target frames per second.
        frame_size: (width, height) tuple.

    Returns:
        Tuple of (cv2.VideoWriter, absolute output_path).
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / output_filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    return writer, out_path


# ---------------------------------------------------------------------------
# Detection event logger
# ---------------------------------------------------------------------------

#: Default log path — one JSONL file per session, appended across frames.
_DEFAULT_LOG_PATH = config.OUTPUTS_DIR / "detection_log.jsonl"


def save_detection_log(
    result,  # DetectionResult — avoid circular import with type hint string
    source_name: str,
    frame_idx: int,
    log_path: Optional[Path] = None,
) -> None:
    """Append a structured JSONL record for the current frame's detections.

    Each record captures all detection metadata for offline analysis,
    dashboarding, or pothole frequency mapping.

    Output format (one JSON object per line):
    ::

        {
          "timestamp": "2026-09-19T17:23:11.042",
          "source": "road.mp4",
          "frame": 142,
          "fps": 28.3,
          "inference_ms": 35.2,
          "count": 2,
          "max_confidence": 0.91,
          "detections": [
            {
              "bbox": [120, 80, 340, 210],
              "centroid": [230, 145],
              "area": 45600,
              "area_ratio": 0.074,
              "confidence": 0.91,
              "severity": "Medium",
              "class_name": "Pothole"
            },
            ...
          ]
        }

    Args:
        result: DetectionResult from PotholeDetector.detect().
        source_name: Human-readable source identifier (filename or "webcam").
        frame_idx: 1-based index of the current frame in the session.
        log_path: Override path for the JSONL file. Defaults to
            ``outputs/detection_log.jsonl``.
    """
    if log_path is None:
        log_path = _DEFAULT_LOG_PATH

    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}",
        "source": source_name,
        "frame": frame_idx,
        "fps": round(result.fps, 2),
        "inference_ms": round(result.inference_time_ms, 2),
        "count": result.count,
        "max_confidence": round(result.max_confidence, 4),
        "detections": [
            {
                "bbox": list(det.bbox),
                "centroid": list(det.centroid),
                "area": det.area,
                "area_ratio": round(det.area_ratio, 5),
                "confidence": round(det.confidence, 4),
                "severity": det.severity,
                "class_name": det.class_name,
            }
            for det in result.detections
        ],
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
