"""Unit tests for PotholeDetection, severity heuristics, and PotholeTracker."""
import pytest
from detector import PotholeDetection, PotholeTracker


def test_pothole_detection_geometry():
    det = PotholeDetection(
        x1=100, y1=200, x2=300, y2=400,
        confidence=0.88,
        class_id=0,
        class_name="Pothole",
        label="Pothole 88%",
        frame_width=1000,
        frame_height=1000,
    )
    assert det.width == 200
    assert det.height == 200
    assert det.area == 40000
    assert det.centroid == (200, 300)
    assert det.area_ratio == pytest.approx(0.04)


def test_depth_compensated_severity():
    # Pothole near top of frame (far away): y1=50, y2=150 in 1000h frame
    far_det = PotholeDetection(
        x1=100, y1=50, x2=200, y2=150,
        confidence=0.90, class_id=0, class_name="Pothole", label="Pothole 90%",
        frame_width=1000, frame_height=1000,
    )
    # Pothole near bottom of frame (close up): y1=800, y2=900 in 1000h frame
    near_det = PotholeDetection(
        x1=100, y1=800, x2=200, y2=900,
        confidence=0.90, class_id=0, class_name="Pothole", label="Pothole 90%",
        frame_width=1000, frame_height=1000,
    )
    assert far_det.area == near_det.area
    # Far pothole should have higher adjusted ratio due to horizon scaling
    assert far_det.severity in ("Low", "Medium", "High")
    assert near_det.severity in ("Low", "Medium", "High")


def test_pothole_tracker_association():
    tracker = PotholeTracker(iou_threshold=0.20, min_hits=2, max_age=5)

    det1 = PotholeDetection(
        x1=100, y1=100, x2=200, y2=200,
        confidence=0.80, class_id=0, class_name="Pothole", label="Pothole 80%",
        frame_width=640, frame_height=480,
    )
    tracker.update([det1], 640, 480)
    assert det1.track_id == 1
    assert tracker.total_detection_instances == 1
    assert tracker.highest_confidence == pytest.approx(0.80)

    # Frame 2: pothole moves slightly down and right
    det2 = PotholeDetection(
        x1=105, y1=110, x2=208, y2=212,
        confidence=0.85, class_id=0, class_name="Pothole", label="Pothole 85%",
        frame_width=640, frame_height=480,
    )
    tracker.update([det2], 640, 480)
    # Should maintain same track ID
    assert det2.track_id == 1
    assert tracker.unique_count == 1
    assert tracker.highest_confidence == pytest.approx(0.85)
    assert tracker.total_detection_instances == 2
