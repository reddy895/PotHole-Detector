"""Helper utilities for video streams, terminal reporting, and file I/O."""
from typing import Optional, Tuple
from pathlib import Path
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
        fps: Current processing frames per second.
        throttle_interval: Minimum seconds between prints to avoid terminal flood.
        last_print_time: Timestamp of last print.

    Returns:
        Updated print timestamp.
    """
    now = time.time()
    if last_print_time is not None and (now - last_print_time) < throttle_interval:
        return last_print_time

    if count > 0:
        conf_pct = highest_confidence * 100
        print(f"[POTHOLE DETECTED]")
        print(f"Count: {count}")
        print(f"Highest Confidence: {conf_pct:.1f}%")
        print(f"FPS: {int(round(fps))}")
        print("-" * 25)
    else:
        print(f"[NO POTHOLE]")
        print(f"FPS: {int(round(fps))}")
        print("-" * 25)

    return now


def save_annotated_image(image: np.ndarray, original_name: str) -> Path:
    """Save annotated image frame to the outputs directory.

    Args:
        image: OpenCV BGR image.
        original_name: Name of original input file.

    Returns:
        Path to saved file.
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
    """Create OpenCV VideoWriter configured for MP4 output.

    Args:
        output_filename: Target video name.
        fps: Target frames per second.
        frame_size: (width, height) tuple.

    Returns:
        Tuple of (cv2.VideoWriter, output_path).
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / output_filename

    # Use mp4v or XVID codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

    return writer, out_path
