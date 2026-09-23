"""Multi-panel UI composition helpers for the Pothole Detection pipeline.

This module is intentionally decoupled from the YOLO inference loop to keep
UI rendering from causing FPS drops in the main detection thread.

Public API
----------
CriticalPotholeTracker
    Identifies the single largest pothole across frames with temporal hysteresis
    to prevent the "CRITICAL" label from flickering between near-equal detections.

ThumbnailSidebarManager
    Maintains a rolling queue of cropped pothole sub-images and renders the
    30%-wide sidebar panel that sits to the right of the main view.

Module functions (also importable directly from detector.py)
------------------------------------------------------------
calculate_pothole_areas(boxes)
create_thumbnail_sidebar(crops_queue, sidebar_width, frame_height)
combine_views(main_frame, sidebar)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Type alias — avoids a circular import with detector.py
# ---------------------------------------------------------------------------
try:
    from detector import PotholeDetection  # noqa: F401 – used for type hints only
except ImportError:
    PotholeDetection = object  # type: ignore[misc,assignment]


# ===========================================================================
# 1.  Standalone module functions (also re-exported from detector.py)
# ===========================================================================

def calculate_pothole_areas(boxes: List) -> List[int]:
    """Return pixel areas (width × height) for every detection.

    Args:
        boxes: List of ``PotholeDetection`` objects (or any object with
               ``.x1``, ``.y1``, ``.x2``, ``.y2`` integer attributes).

    Returns:
        List of non-negative integer areas, same length and order as *boxes*.
    """
    return [max(0, (d.x2 - d.x1) * (d.y2 - d.y1)) for d in boxes]


def create_thumbnail_sidebar(
    crops_queue,
    sidebar_width: int,
    frame_height: int,
):
    """Render the right-side inspection-log sidebar panel (Pillar 3).

    Each card shows a letterboxed pothole crop (aspect-ratio preserved,
    dark slate padding) plus civil engineering metadata:
      - ``ID: #N | TierX``  (tier-colour coded)
      - ``Asphalt: X.Xkg``   (estimated patch weight)
      - ``XX%``              (detection confidence)

    Empty slots display a "Scanning road surface..." centred placeholder.

    Args:
        crops_queue: Deque of crop dicts from ``InspectionRegistry``.
            Keys: ``crop``, ``track_id``, ``tier``, ``weight_kg``,
            ``depth_cm``, ``confidence``.
        sidebar_width: Pixel width of the rendered sidebar.
        frame_height: Pixel height to match the main frame.

    Returns:
        BGR uint8 ndarray of shape ``(frame_height, sidebar_width, 3)``.
    """
    from utils.civil_metrics import TIER_COLORS, TIER_SHORT

    # ---- layout constants ------------------------------------------------
    MAX_SLOTS = 4
    HEADER_H  = 38
    FOOTER_H  = 22
    CARD_PAD  = 4
    INFO_H    = 44    # pixels reserved below thumbnail for text

    SLOT_H = max(1, (frame_height - HEADER_H - FOOTER_H) // MAX_SLOTS)

    # ---- colour palette --------------------------------------------------
    BG_COLOR   = (14,  16,  22)   # very dark navy
    HEADER_BG  = (22,  26,  36)
    DIVIDER    = (45,  50,  66)
    ACCENT     = (30, 210, 200)   # teal
    SLATE      = (30,  30,  30)   # letterbox padding colour
    TEXT_DIM   = (110, 120, 140)
    CONF_GRN   = (60, 200, 100)
    WEIGHT_ORG = (50, 155, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fsm  = 0.38    # small
    fxs  = 0.33    # extra-small
    th1  = 1

    canvas = np.full((frame_height, sidebar_width, 3), BG_COLOR, dtype=np.uint8)

    # ---- header bar -------------------------------------------------------
    cv2.rectangle(canvas, (0, 0), (sidebar_width, HEADER_H), HEADER_BG, -1)
    cv2.line(canvas, (0, HEADER_H - 1), (sidebar_width, HEADER_H - 1), DIVIDER, 1)

    title = "INSPECTION LOG"
    (tw, _), _ = cv2.getTextSize(title, font, 0.42, 2)
    tx = max(4, (sidebar_width - tw) // 2)
    cv2.putText(canvas, title, (tx, 24), font, 0.42, ACCENT, 2, cv2.LINE_AA)

    dot_on  = int(time.time()) % 2 == 0
    dot_col = CONF_GRN if dot_on else TEXT_DIM
    cv2.circle(canvas, (8, HEADER_H - 10), 4, dot_col, -1, cv2.LINE_AA)
    cv2.putText(canvas, "LIVE", (16, HEADER_H - 5),
                font, fxs, dot_col, th1, cv2.LINE_AA)

    # ---- card slots -------------------------------------------------------
    crops_list = list(crops_queue)

    for slot in range(MAX_SLOTS):
        y_top  = HEADER_H + slot * SLOT_H
        y_bot  = min(frame_height - FOOTER_H, y_top + SLOT_H)
        card_h = y_bot - y_top
        if card_h <= 0:
            break

        if slot > 0:
            cv2.line(canvas, (CARD_PAD, y_top),
                     (sidebar_width - CARD_PAD, y_top), DIVIDER, 1)

        # Most-recent crop at top (reverse indexing)
        rev_idx = len(crops_list) - 1 - slot

        if 0 <= rev_idx < len(crops_list):
            entry    = crops_list[rev_idx]
            crop     = entry.get("crop")
            tid      = entry.get("track_id", -1)
            tier     = entry.get("tier",      "Unknown")
            wkg      = entry.get("weight_kg", 0.0)
            conf     = entry.get("confidence", 0.0)
            tier_col = TIER_COLORS.get(tier,  (160, 160, 160))
            tier_s   = TIER_SHORT.get(tier,   "?")

            # ---- letterboxed thumbnail ------------------------------------
            thumb_h = max(1, card_h - INFO_H - CARD_PAD * 2)
            thumb_w = sidebar_width - CARD_PAD * 2

            if crop is not None and crop.size > 0:
                try:
                    ch, cw = crop.shape[:2]
                    scale  = min(thumb_w / max(1, cw), thumb_h / max(1, ch))
                    nw     = max(1, int(cw * scale))
                    nh     = max(1, int(ch * scale))
                    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

                    lb = np.full((thumb_h, thumb_w, 3), SLATE, dtype=np.uint8)
                    oy = (thumb_h - nh) // 2
                    ox = (thumb_w - nw) // 2
                    lb[oy:oy + nh, ox:ox + nw] = resized

                    py1 = y_top + CARD_PAD
                    py2 = min(frame_height - FOOTER_H, py1 + thumb_h)
                    actual = py2 - py1
                    if actual > 0:
                        canvas[py1:py2, CARD_PAD:CARD_PAD + thumb_w] = lb[:actual]

                    # Tier-coloured border around thumbnail
                    cv2.rectangle(canvas,
                                  (CARD_PAD, py1), (CARD_PAD + thumb_w - 1, py2 - 1),
                                  tier_col, 1)
                except Exception:
                    pass

            # ---- metadata bar below thumbnail -----------------------------
            info_bg_y  = max(0, y_bot - INFO_H)
            info_y_row1 = info_bg_y + 14
            info_y_row2 = info_bg_y + 30

            cv2.rectangle(canvas, (0, info_bg_y), (sidebar_width, y_bot),
                          (12, 14, 20), -1)

            # Row 1: ID + Tier (tier-coloured)
            id_txt = f"ID: #{tid} | {tier_s}"
            cv2.putText(canvas, id_txt, (CARD_PAD + 2, info_y_row1),
                        font, fsm, tier_col, th1, cv2.LINE_AA)

            # Row 2: Asphalt weight left, confidence right
            wkg_txt  = f"Asphalt: {wkg:.1f}kg"
            conf_txt = f"{conf * 100:.0f}%"
            (csw, _), _ = cv2.getTextSize(conf_txt, font, fxs, th1)
            cv2.putText(canvas, wkg_txt, (CARD_PAD + 2, info_y_row2),
                        font, fxs, WEIGHT_ORG, th1, cv2.LINE_AA)
            cv2.putText(canvas, conf_txt,
                        (sidebar_width - csw - CARD_PAD - 2, info_y_row2),
                        font, fxs, CONF_GRN, th1, cv2.LINE_AA)

        else:
            # ---- empty slot placeholder -----------------------------------
            cy   = y_top + card_h // 2 - 6
            cx   = sidebar_width // 2
            dash = (40, 45, 60)

            for dx in range(CARD_PAD + 4, sidebar_width - CARD_PAD, 8):
                se = min(dx + 4, sidebar_width - CARD_PAD - 1)
                cv2.line(canvas, (dx, y_top + CARD_PAD), (se, y_top + CARD_PAD), dash, 1)
                cv2.line(canvas, (dx, y_bot - CARD_PAD), (se, y_bot - CARD_PAD), dash, 1)
            for dy in range(y_top + CARD_PAD, y_bot - CARD_PAD, 8):
                se = min(dy + 4, y_bot - CARD_PAD - 1)
                cv2.line(canvas, (CARD_PAD, dy), (CARD_PAD, se), dash, 1)
                cv2.line(canvas, (sidebar_width - CARD_PAD - 1, dy),
                         (sidebar_width - CARD_PAD - 1, se), dash, 1)

            r = min(12, card_h // 5)
            cv2.circle(canvas, (cx, cy), r, dash, 1, cv2.LINE_AA)
            cv2.line(canvas, (cx - r - 3, cy), (cx + r + 3, cy), dash, 1)
            cv2.line(canvas, (cx, cy - r - 3), (cx, cy + r + 3), dash, 1)

            scan1 = "Scanning road"
            scan2 = "surface..."
            (sw1, _), _ = cv2.getTextSize(scan1, font, fxs, th1)
            cv2.putText(canvas, scan1, (cx - sw1 // 2, cy + r + 14),
                        font, fxs, TEXT_DIM, th1, cv2.LINE_AA)
            cv2.putText(canvas, scan2, (cx - sw1 // 2, cy + r + 26),
                        font, fxs, TEXT_DIM, th1, cv2.LINE_AA)

    # ---- footer strip -----------------------------------------------------
    footer_y = frame_height - FOOTER_H
    if footer_y > HEADER_H:
        cv2.rectangle(canvas, (0, footer_y), (sidebar_width, frame_height),
                      HEADER_BG, -1)
        cv2.putText(canvas, time.strftime("%H:%M:%S"),
                    (CARD_PAD, frame_height - 6),
                    font, fxs, TEXT_DIM, th1, cv2.LINE_AA)
        logged_txt = f"{len(crops_list)} logged"
        (lw, _), _ = cv2.getTextSize(logged_txt, font, fxs, th1)
        cv2.putText(canvas, logged_txt,
                    (sidebar_width - lw - CARD_PAD, frame_height - 6),
                    font, fxs, CONF_GRN, th1, cv2.LINE_AA)

    return canvas


def combine_views(main_frame: np.ndarray, sidebar: np.ndarray) -> np.ndarray:
    """Horizontally concatenate the main annotated frame and the sidebar panel.

    Handles height mismatches by padding the shorter panel with black rows so
    that ``np.hstack`` never raises a shape error.

    Args:
        main_frame: BGR ndarray — the YOLO-annotated primary view.
        sidebar:    BGR ndarray — the thumbnail sidebar from
                    :func:`create_thumbnail_sidebar`.

    Returns:
        Composite BGR ndarray of shape
        ``(max(main_h, sidebar_h), main_w + sidebar_w, 3)``.
    """
    mh, mw = main_frame.shape[:2]
    sh, sw = sidebar.shape[:2]

    target_h = max(mh, sh)

    # Pad main frame vertically if shorter
    if mh < target_h:
        pad = np.zeros((target_h - mh, mw, 3), dtype=np.uint8)
        main_frame = np.vstack([main_frame, pad])

    # Pad sidebar vertically if shorter
    if sh < target_h:
        pad = np.zeros((target_h - sh, sw, 3), dtype=np.uint8)
        sidebar = np.vstack([sidebar, pad])

    # Thin separator line between the two panels
    separator = np.full((target_h, 2, 3), (55, 60, 75), dtype=np.uint8)

    return np.hstack([main_frame, separator, sidebar])


# ===========================================================================
# 2.  CriticalPotholeTracker — temporal hysteresis for "largest pothole" label
# ===========================================================================

class CriticalPotholeTracker:
    """Identifies and stably tracks the single largest pothole across frames.

    The "critical" designation (thick red border + warning label) is applied
    to the pothole with the greatest bounding-box area in each frame.
    A **hysteresis margin** (default 10 %) prevents the label from flickering
    when two similarly-sized potholes alternate in being marginally larger.

    The critical ID is reset automatically when no detections are present,
    and switches to a new track only when the challenger's area exceeds the
    incumbent's area by more than ``hysteresis_margin``.

    Usage::

        tracker = CriticalPotholeTracker()
        ...
        critical_id = tracker.update(result.detections)
        annotated = detector.draw_annotations(frame, result,
                                              critical_id=critical_id)

    Args:
        hysteresis_margin: Fractional area advantage required for a new
            detection to displace the current champion.  Default 0.10 (10 %).
    """

    def __init__(self, hysteresis_margin: float = 0.10) -> None:
        self._margin = max(0.0, hysteresis_margin)
        self._critical_track_id: Optional[int] = None
        self._critical_area: int = 0
        self._frames_without_detection: int = 0
        # How many consecutive frames without any detection before we reset
        self._reset_after_frames: int = 5

    def update(self, detections: List) -> Optional[int]:
        """Compute and return the track_id of the "critical" pothole this frame.

        Args:
            detections: List of ``PotholeDetection`` objects from the current
                        ``DetectionResult``.

        Returns:
            ``track_id`` (int) of the largest stable pothole, or ``None`` if
            there are no detections or no track IDs have been assigned yet.
        """
        if not detections:
            self._frames_without_detection += 1
            if self._frames_without_detection >= self._reset_after_frames:
                self._critical_track_id = None
                self._critical_area = 0
            return None

        self._frames_without_detection = 0

        # Find the largest detection by pixel area this frame
        largest = max(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
        largest_area = max(0, (largest.x2 - largest.x1) * (largest.y2 - largest.y1))

        if self._critical_track_id is None:
            # First detection ever — just set it
            self._critical_track_id = largest.track_id
            self._critical_area = largest_area
        else:
            # Check if incumbent is still visible
            incumbent_visible = any(
                d.track_id == self._critical_track_id for d in detections
            )

            if not incumbent_visible:
                # Old champion disappeared — switch unconditionally
                self._critical_track_id = largest.track_id
                self._critical_area = largest_area
            elif largest_area > self._critical_area * (1.0 + self._margin):
                # Challenger is significantly larger — switch
                self._critical_track_id = largest.track_id
                self._critical_area = largest_area
            else:
                # Keep incumbent — update its recorded area to the current frame's value
                for d in detections:
                    if d.track_id == self._critical_track_id:
                        self._critical_area = max(0, (d.x2 - d.x1) * (d.y2 - d.y1))
                        break

        return self._critical_track_id

    def reset(self) -> None:
        """Forcibly clear the current critical pothole state."""
        self._critical_track_id = None
        self._critical_area = 0
        self._frames_without_detection = 0


# ===========================================================================
# 3.  ThumbnailSidebarManager — rolling crop queue + sidebar render
# ===========================================================================

class ThumbnailSidebarManager:
    """Manages a rolling buffer of detected pothole sub-image crops.

    On every frame with detections, call :meth:`push_detections` to extract
    and store cropped bounding-box regions.  Call :meth:`render_sidebar` to
    obtain the rendered sidebar panel for compositing.

    The manager keeps the most recent ``maxlen`` crops across *all* detections
    (not just the largest).  Crops are stored with metadata for overlay labels.

    Args:
        maxlen: Maximum number of crops retained in the rolling buffer.
        main_view_ratio: Fraction of total window width allocated to the main
            view.  The sidebar takes ``1 - main_view_ratio`` of the width.
    """

    def __init__(
        self,
        maxlen: int = 5,
        main_view_ratio: float = 0.72,
    ) -> None:
        self._maxlen = max(1, maxlen)
        self.main_view_ratio = max(0.5, min(0.9, main_view_ratio))
        self._queue: Deque[Dict] = deque(maxlen=self._maxlen)

    # ------------------------------------------------------------------
    # Sidebar width helper
    # ------------------------------------------------------------------

    def sidebar_width(self, total_width: int) -> int:
        """Compute sidebar pixel width given total composite canvas width."""
        return max(120, total_width - int(total_width * self.main_view_ratio))

    # ------------------------------------------------------------------
    # Crop extraction
    # ------------------------------------------------------------------

    def push_detections(
        self,
        frame: np.ndarray,
        detections: List,
        frame_idx: int,
    ) -> None:
        """Extract and store bounding-box crops for all detections this frame.

        Coordinates are clamped to valid frame bounds before slicing to
        prevent any ``cv2`` boundary crash.

        Args:
            frame:      Raw (un-annotated) BGR frame from the camera/video.
            detections: List of ``PotholeDetection`` from ``DetectionResult``.
            frame_idx:  Current frame index (1-based) for thumbnail labels.
        """
        if frame is None or frame.size == 0 or not detections:
            return

        h, w = frame.shape[:2]

        for det in detections:
            # Clamp slice boundaries
            y1 = max(0, min(det.y1, h - 1))
            y2 = max(0, min(det.y2, h))
            x1 = max(0, min(det.x1, w - 1))
            x2 = max(0, min(det.x2, w))

            if y2 <= y1 or x2 <= x1:
                continue  # degenerate box — skip

            crop = frame[y1:y2, x1:x2].copy()
            area = max(0, (det.x2 - det.x1) * (det.y2 - det.y1))

            self._queue.append(
                {
                    "crop": crop,
                    "frame_idx": frame_idx,
                    "confidence": float(det.confidence),
                    "area": area,
                    "timestamp": time.time(),
                }
            )

    # ------------------------------------------------------------------
    # Sidebar rendering
    # ------------------------------------------------------------------

    def render_sidebar(self, sidebar_width: int, frame_height: int) -> np.ndarray:
        """Render the thumbnail sidebar panel.

        Delegates to the module-level :func:`create_thumbnail_sidebar` function.

        Args:
            sidebar_width: Pixel width for the rendered panel.
            frame_height:  Pixel height to match the main annotated frame.

        Returns:
            BGR uint8 ndarray of shape ``(frame_height, sidebar_width, 3)``.
        """
        return create_thumbnail_sidebar(self._queue, sidebar_width, frame_height)

    @property
    def queue(self) -> Deque[Dict]:
        """Read-only access to the internal crop queue."""
        return self._queue
