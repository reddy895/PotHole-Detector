"""Unit tests for video utilities, reporting, and event logging."""
import tempfile
from pathlib import Path
import json
import pytest
from detector import PotholeDetection, DetectionResult
from utils.video_utils import (
    print_banner,
    print_progress_bar,
    print_final_summary,
    save_detection_log,
)


def test_save_detection_log():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        log_path = Path(tmp.name)

    try:
        det = PotholeDetection(
            x1=50, y1=60, x2=150, y2=160,
            confidence=0.88, class_id=0, class_name="Pothole", label="Pothole 88%",
            frame_width=640, frame_height=480,
        )
        result = DetectionResult(
            detections=[det], count=1, fps=12.5, inference_time_ms=45.0,
            max_confidence=0.88, frame_width=640, frame_height=480,
        )
        save_detection_log(result, source_name="test.mp4", frame_idx=1, log_path=log_path)

        assert log_path.is_file()
        content = log_path.read_text().strip()
        data = json.loads(content)
        assert data["source"] == "test.mp4"
        assert data["count"] == 1
        assert len(data["detections"]) == 1
        assert data["detections"][0]["confidence"] == 0.88
    finally:
        if log_path.is_file():
            log_path.unlink()


def test_print_progress_bar_smoke():
    # Smoke test to ensure no uncaught exceptions
    print_progress_bar(
        frame_idx=50,
        total_frames=100,
        fps=12.5,
        current_potholes=3,
        highest_confidence=0.91,
    )


def test_print_final_summary_smoke():
    # Smoke test to ensure formatting executes cleanly
    print_final_summary(
        source_name="road.mp4",
        processed_frames=100,
        total_frames=100,
        unique_potholes=4,
        total_instances=82,
        highest_confidence=0.94,
        avg_fps=12.5,
    )
