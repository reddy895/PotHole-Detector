"""Civil engineering metrics for pothole volumetric severity estimation.

Ground-plane homography model
------------------------------
Dashcam footage has a natural perspective: objects near the bottom of the
frame are close to the vehicle; objects near the top are far away.
We exploit this by interpolating a metres-per-pixel scale based on the
detection centroid's vertical position.

    y_norm = centroid_y / frame_height   (0=top/far, 1=bottom/close)
    scale_m_per_px = SCALE_FAR + (SCALE_NEAR - SCALE_FAR) * y_norm

Volumetric approximation
------------------------
Crater shape is modelled as an inverted parabolic cap:
    Volume = 0.5 × surface_area_m2 × depth_m

Asphalt patch weight uses standard cold-mix bitumen density (2 400 kg/m³).

Tier thresholds (depth-based civil classification)
---------------------------------------------------
    Tier 1  Surface Raveling   depth < 2.5 cm   green   low priority
    Tier 2  Moderate Crater    depth < 5.0 cm   orange  medium priority
    Tier 3  Critical Hazard    depth ≥ 5.0 cm   red     immediate action
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Camera / ground-plane calibration
# ---------------------------------------------------------------------------
SCALE_NEAR_M_PX: float = 0.004   # m/px at bottom of frame  (~2-3 m ahead)
SCALE_FAR_M_PX: float  = 0.018   # m/px at top of frame     (~12-15 m ahead)

ASPHALT_DENSITY_KG_M3: float = 2400.0   # kg/m³ — standard compacted cold-mix

DEPTH_MIN_CM: float = 1.0
DEPTH_MAX_CM: float = 12.0

# Tier depth thresholds (cm)
TIER1_MAX_DEPTH: float = 2.5
TIER2_MAX_DEPTH: float = 5.0

# BGR colours for each tier (OpenCV format)
TIER_COLORS: Dict[str, tuple] = {
    "Tier 1": (40,  210, 80),    # green
    "Tier 2": (0,   150, 255),   # orange
    "Tier 3": (20,  20,  220),   # bold red
    "Unknown": (160, 160, 160),  # grey fallback
}

# Tier short labels for on-screen display
TIER_SHORT: Dict[str, str] = {
    "Tier 1": "T1",
    "Tier 2": "T2",
    "Tier 3": "T3",
    "Unknown": "??",
}

# RHI deduction per detection instance
RHI_DEDUCTIONS: Dict[str, int] = {"Tier 1": 2, "Tier 2": 6, "Tier 3": 15, "Unknown": 1}


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _adaptive_scale(centroid_y: float, frame_height: int) -> float:
    """Return the adaptive ground-plane scale (m/px) for a given Y position."""
    y_norm = max(0.0, min(1.0, centroid_y / max(1, frame_height)))
    # y_norm=0 → far (top) → large scale;  y_norm=1 → near (bottom) → small scale
    return SCALE_FAR_M_PX + (SCALE_NEAR_M_PX - SCALE_FAR_M_PX) * y_norm


def compute_civil_metrics(
    x1: int, y1: int, x2: int, y2: int,
    frame_height: int,
) -> dict:
    """Compute civil engineering metrics for a single pothole bounding box.

    All coordinates must already be clamped to valid frame bounds before calling.

    Args:
        x1, y1, x2, y2:  Bounding box pixel corners.
        frame_height:     Full frame pixel height (used for perspective scaling).

    Returns:
        dict with keys:
            depth_cm (float)      — estimated crater depth
            area_real_m2 (float)  — estimated surface area in m²
            volume_m3 (float)     — estimated crater volume in m³
            weight_kg (float)     — cold-mix asphalt patch weight in kg
            tier (str)            — "Tier 1" / "Tier 2" / "Tier 3"
    """
    w_px = max(1, x2 - x1)
    h_px = max(1, y2 - y1)
    area_px = w_px * h_px
    centroid_y = (y1 + y2) * 0.5

    # Ground-plane adaptive scale
    scale = _adaptive_scale(centroid_y, frame_height)

    # Real-world surface area (m²)
    area_real_m2 = area_px * (scale ** 2)

    # Depth estimate: bbox height proportional to frame height acts as
    # a depth proxy — a taller box implies a larger / deeper crater.
    depth_norm = h_px / max(1, frame_height)          # [0, 1]
    depth_cm   = DEPTH_MIN_CM + depth_norm * (DEPTH_MAX_CM - DEPTH_MIN_CM)
    depth_cm   = max(DEPTH_MIN_CM, min(DEPTH_MAX_CM, depth_cm))

    # Inverted parabolic cap volume approximation
    depth_m   = depth_cm / 100.0
    volume_m3 = 0.5 * area_real_m2 * depth_m

    # Cold-mix asphalt patch weight
    weight_kg = volume_m3 * ASPHALT_DENSITY_KG_M3

    # Tier classification by depth
    if depth_cm < TIER1_MAX_DEPTH:
        tier = "Tier 1"
    elif depth_cm < TIER2_MAX_DEPTH:
        tier = "Tier 2"
    else:
        tier = "Tier 3"

    return {
        "depth_cm":    round(depth_cm,    1),
        "area_real_m2": round(area_real_m2, 5),
        "volume_m3":   round(volume_m3,   7),
        "weight_kg":   round(weight_kg,   2),
        "tier":        tier,
    }


# ---------------------------------------------------------------------------
# Road Health Index
# ---------------------------------------------------------------------------

def calculate_rhi(detections: list) -> int:
    """Compute the Road Health Index (0–100) for the detections in a frame.

    Formula:
        RHI = max(0, 100 − Σ deduction_per_detection)

    Args:
        detections: List of objects with a ``.tier`` attribute.

    Returns:
        Integer RHI in [0, 100].
    """
    deduction = sum(
        RHI_DEDUCTIONS.get(getattr(d, "tier", "Unknown"), 1)
        for d in detections
    )
    return max(0, 100 - deduction)


def rhi_color(rhi: int) -> tuple:
    """BGR colour for a given RHI score.

    Returns:
        Green (>80), Yellow (50–80), Red (<50).
    """
    if rhi > 80:
        return (60, 200, 60)     # green
    if rhi > 50:
        return (0,  200, 240)    # yellow (BGR)
    return (30, 30, 220)         # red


# ---------------------------------------------------------------------------
# InspectionRegistry — persistent multi-frame pothole deduplication
# ---------------------------------------------------------------------------

class InspectionRegistry:
    """Persistent pothole registry keyed by ByteTrack persistent IDs.

    Lifecycle of a tracked pothole
    --------------------------------
    1. First detected → entry created in ``_registry``
    2. Each frame → entry updated with highest-confidence crop / metadata
    3. Missing for ``max_missing_frames`` consecutive frames → permanently
       marked ``logged=True``; best crop pushed to ``crops_queue`` for sidebar
    4. ``logged`` entries are never double-counted in ``unique_logged_count``

    Args:
        max_missing_frames: Frames of absence before a pothole is considered
            departed and logged. Default 15 frames.
        sidebar_capacity:   Max entries in the sidebar FIFO crop queue.
    """

    def __init__(
        self,
        max_missing_frames: int = 15,
        sidebar_capacity: int = 4,
    ) -> None:
        self._registry: Dict[int, dict] = {}
        self.max_missing_frames = max_missing_frames
        # FIFO queue of highest-confidence crops for the inspection-log sidebar
        self.crops_queue: Deque[dict] = deque(maxlen=sidebar_capacity)
        self._max_conf_seen: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_frame(self, detections: list, frame: "np.ndarray") -> None:
        """Process all detections from the current frame.

        Extracts and stores crops, updates best metadata, and ages out
        tracks that have been missing too long.

        Args:
            detections: List of ``PotholeDetection`` objects (track_id must
                be set by ByteTrack — ``None`` IDs are silently skipped).
            frame:      Raw BGR frame (used for crop extraction).
        """
        h, w = frame.shape[:2]
        current_ids: set = set()

        for det in detections:
            tid = getattr(det, "track_id", None)
            if tid is None:
                continue
            current_ids.add(tid)

            # ---- safe crop extraction (fully clamped) ----
            cy1 = int(np.clip(det.y1, 0, h - 1))
            cy2 = int(np.clip(det.y2, 1, h))
            cx1 = int(np.clip(det.x1, 0, w - 1))
            cx2 = int(np.clip(det.x2, 1, w))
            crop = frame[cy1:cy2, cx1:cx2].copy() if cy2 > cy1 and cx2 > cx1 else None

            conf  = float(getattr(det, "confidence", 0.0))
            area  = int(getattr(det, "area", 0))
            tier  = str(getattr(det, "tier",      "Unknown"))
            wkg   = float(getattr(det, "weight_kg", 0.0))
            dpcm  = float(getattr(det, "depth_cm",  0.0))

            if tid not in self._registry:
                self._registry[tid] = {
                    "track_id":      tid,
                    "max_conf":      conf,
                    "max_area":      area,
                    "best_crop":     crop,
                    "logged":        False,
                    "tier":          tier,
                    "weight_kg":     wkg,
                    "depth_cm":      dpcm,
                    "frames_missing": 0,
                    "first_seen":    time.time(),
                }
            else:
                entry = self._registry[tid]
                entry["frames_missing"] = 0   # reset age counter
                if conf > entry["max_conf"]:
                    entry["max_conf"]  = conf
                    if crop is not None:
                        entry["best_crop"] = crop
                if area > entry["max_area"]:
                    entry["max_area"]  = area
                    entry["tier"]      = tier
                    entry["weight_kg"] = wkg
                    entry["depth_cm"]  = dpcm

            if conf > self._max_conf_seen:
                self._max_conf_seen = conf

        # ---- age out absent tracks ----
        for tid, entry in self._registry.items():
            if tid in current_ids or entry["logged"]:
                continue
            entry["frames_missing"] += 1
            if entry["frames_missing"] >= self.max_missing_frames:
                self._log_entry(tid)

    def flush_all(self) -> None:
        """Force-log every remaining unlogged entry (call at session end)."""
        for tid in list(self._registry.keys()):
            if not self._registry[tid]["logged"]:
                self._log_entry(tid)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unique_logged_count(self) -> int:
        """Number of potholes permanently logged (departed from scene)."""
        return sum(1 for e in self._registry.values() if e["logged"])

    @property
    def total_unique_count(self) -> int:
        """Total unique track IDs seen (active + logged)."""
        return len(self._registry)

    @property
    def max_confidence(self) -> float:
        """Highest detection confidence observed across the entire session."""
        return self._max_conf_seen

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _log_entry(self, tid: int) -> None:
        """Mark a pothole as logged and push its best crop to the sidebar queue."""
        entry = self._registry[tid]
        if entry["logged"]:
            return
        entry["logged"] = True
        crop = entry.get("best_crop")
        if crop is not None and crop.size > 0:
            self.crops_queue.append({
                "crop":       crop,
                "track_id":   tid,
                "tier":       entry["tier"],
                "weight_kg":  entry["weight_kg"],
                "depth_cm":   entry["depth_cm"],
                "confidence": entry["max_conf"],
            })
