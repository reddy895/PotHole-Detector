"""Helper utilities for video streams, terminal reporting, file I/O, and detection logging."""
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

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
        print("[POTHOLE DETECTED]")
        print(f"Count: {count}")
        print(f"Highest Confidence: {conf_pct:.1f}%")
        print(f"FPS: {fps:.1f}")
        print("-" * 25)
    else:
        print("[NO POTHOLE]")
        print(f"FPS: {fps:.1f}")
        print("-" * 25)

    return now


def print_progress_bar(
    frame_idx: int,
    total_frames: int,
    fps: float,
    current_potholes: int,
    highest_confidence: float,
    bar_length: int = 25,
) -> None:
    """Print an in-place dynamic progress update on a single line (no scrolling spam).

    Args:
        frame_idx: Current frame number (1-based).
        total_frames: Total number of frames in the video.
        fps: Current processing frames per second.
        current_potholes: Count of unique potholes tracked so far.
        highest_confidence: Highest confidence score (0.0 to 1.0) observed so far.
        bar_length: Length of ASCII progress bar.
    """
    total = max(1, total_frames)
    pct = min(100.0, (frame_idx / total) * 100)
    filled = int(bar_length * frame_idx // total)
    bar = "=" * filled + "-" * (bar_length - filled)
    conf_str = f"{highest_confidence * 100:.1f}%" if highest_confidence > 0 else "0.0%"
    line = (
        f"\r[PROCESSING] [{bar}] {pct:5.1f}% | "
        f"Frame {frame_idx}/{total_frames} | "
        f"FPS: {fps:4.1f} | "
        f"Potholes: {current_potholes} | "
        f"Max Conf: {conf_str}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def print_webcam_status(
    frame_idx: int,
    fps: float,
    current_count: int,
    total_unique: int,
    highest_confidence: float,
) -> None:
    """Print an in-place dynamic status update for webcam feed on a single line.

    Args:
        frame_idx: Current frame count.
        fps: Current processing frames per second.
        current_count: Number of potholes in the current frame.
        total_unique: Total unique potholes tracked so far.
        highest_confidence: Peak confidence observed across all detections so far.
    """
    conf_str = f"{highest_confidence * 100:.1f}%" if highest_confidence > 0 else "0.0%"
    line = (
        f"\r[LIVE WEBCAM] Frame {frame_idx:5d} | "
        f"FPS: {fps:4.1f} | "
        f"Current: {current_count} | "
        f"Total Unique: {total_unique} | "
        f"Max Conf: {conf_str}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def print_final_summary(
    source_name: str,
    processed_frames: int,
    total_frames: int,
    unique_potholes: int,
    total_instances: int,
    highest_confidence: float,
    avg_fps: float,
    output_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> None:
    """Print the final detection summary report to terminal.

    Args:
        source_name: Name of video/source processed.
        processed_frames: Number of frames successfully processed.
        total_frames: Total number of frames in the source (or processed if unknown).
        unique_potholes: Number of distinct unique potholes identified.
        total_instances: Total pothole detection occurrences summed across all frames.
        highest_confidence: Maximum confidence score observed (0.0 to 1.0).
        avg_fps: Average frames per second achieved during processing.
        output_path: Optional path to saved annotated output video.
        log_path: Optional path to saved JSONL detection log.
    """
    conf_str = f"{highest_confidence * 100:.1f}%" if highest_confidence > 0 else "N/A"
    total_str = f"{total_frames}" if total_frames > 0 else f"{processed_frames}"

    print("\n")
    print("=" * 60)
    print("              POTHOLE DETECTION FINAL SUMMARY")
    print("=" * 60)
    print(f"  Source:                    {source_name}")
    print(f"  Processed Frames:          {processed_frames} / {total_str}")
    if unique_potholes > 0:
        print(f"  Total Potholes Detected:   {unique_potholes} unique pothole(s)")
        print(f"  Total Detection Events:    {total_instances} frame instances")
    else:
        print("  Total Potholes Detected:   0 potholes")
    print(f"  Highest Confidence:        {conf_str}")
    print(f"  Average FPS:               {avg_fps:.1f}")
    if output_path:
        suffix = output_path.suffix.lower()
        media_type = "Image" if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else "Video"
        lbl = f"Annotated {media_type} Saved:"
        print(f"  {lbl:<27}{output_path}")

    if log_path:
        print(f"  Detection Log:             {log_path}")

    print("=" * 60)
    print()



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
