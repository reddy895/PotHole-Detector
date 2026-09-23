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
    crops_queue: Deque[Dict],
    sidebar_width: int,
    frame_height: int,
) -> np.ndarray:
    """Render the right-side thumbnail sidebar panel.

    Each slot shows a pothole crop with confidence, area, and frame-index
    overlaid in a clean dark-on-white badge style.  Empty slots are filled
    with a subtle placeholder pattern.

    Args:
        crops_queue: A ``collections.deque`` of crop dicts produced by
            :class:`ThumbnailSidebarManager`.  Each dict has keys:
            ``crop`` (BGR ndarray), ``frame_idx`` (int),
            ``confidence`` (float 0-1), ``area`` (int px²),
            ``timestamp`` (float epoch seconds).
        sidebar_width: Pixel width of the rendered sidebar panel.
        frame_height: Pixel height to match the main frame.

    Returns:
        BGR uint8 ndarray of shape ``(frame_height, sidebar_width, 3)``.
    """
    # ---- constants --------------------------------------------------------
    MAX_SLOTS = 5
    SLOT_H = max(1, frame_height // MAX_SLOTS)
    BG_COLOR = (18, 20, 26)          # very dark navy
    HEADER_BG = (28, 32, 42)         # slightly lighter for header bar
    DIVIDER_COLOR = (50, 55, 70)
    ACCENT_CYAN = (210, 200, 20)     # BGR → warm gold for header text
    TEXT_WHITE = (230, 235, 245)
    TEXT_DIM = (130, 140, 160)
    CONF_GREEN = (60, 200, 100)
    AREA_ORANGE = (60, 160, 255)     # BGR orange

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_sm = 0.42
    font_xs = 0.36
    thick1 = 1
    thick2 = 2

    canvas = np.full((frame_height, sidebar_width, 3), BG_COLOR, dtype=np.uint8)

    # ---- header bar -------------------------------------------------------
    header_h = 36
    cv2.rectangle(canvas, (0, 0), (sidebar_width, header_h), HEADER_BG, -1)
    cv2.rectangle(canvas, (0, header_h - 1), (sidebar_width, header_h), DIVIDER_COLOR, 1)

    title = "DETECTED POTHOLES"
    (tw, _), _ = cv2.getTextSize(title, font, font_sm, thick2)
    tx = max(4, (sidebar_width - tw) // 2)
    cv2.putText(canvas, title, (tx, 24), font, font_sm, ACCENT_CYAN, thick2, cv2.LINE_AA)

    # count badge
    count_str = f"{len(crops_queue)}/{MAX_SLOTS}"
    (cw, _), _ = cv2.getTextSize(count_str, font, font_xs, thick1)
    cv2.putText(canvas, count_str, (sidebar_width - cw - 6, 22),
                font, font_xs, TEXT_DIM, thick1, cv2.LINE_AA)

    # ---- slots ------------------------------------------------------------
    crops_list = list(crops_queue)  # newest first (deque appendleft) or FIFO

    for slot in range(MAX_SLOTS):
        y_top = header_h + slot * SLOT_H
        y_bot = min(frame_height, y_top + SLOT_H)
        slot_h = y_bot - y_top
        if slot_h <= 0:
            break

        # thin divider between slots
        if slot > 0:
            cv2.line(canvas, (4, y_top), (sidebar_width - 4, y_top), DIVIDER_COLOR, 1)

        if slot < len(crops_list):
            entry = crops_list[-(slot + 1)]  # show most-recent at top
            crop: np.ndarray = entry["crop"]
            conf: float = entry["confidence"]
            area: int = entry["area"]
            fidx: int = entry["frame_idx"]

            # ---- thumbnail image ------------------------------------------
            thumb_h = max(1, slot_h - 38)  # leave room for text bar below
            thumb_w = sidebar_width - 8    # 4px margin each side
            if crop is not None and crop.size > 0:
                try:
                    thumb = cv2.resize(crop, (thumb_w, thumb_h),
                                       interpolation=cv2.INTER_LINEAR)
                    # clamp paste region
                    paste_y1 = y_top + 2
                    paste_y2 = paste_y1 + thumb_h
                    paste_y2 = min(frame_height, paste_y2)
                    actual_h = paste_y2 - paste_y1
                    if actual_h > 0:
                        canvas[paste_y1:paste_y2, 4:4 + thumb_w] = thumb[:actual_h]
                except cv2.error:
                    pass

            # ---- info bar beneath thumbnail -------------------------------
            info_y = min(frame_height - 2, y_bot - 34)

            # dark backing for readability
            bar_top = max(0, info_y - 2)
            bar_bot = min(frame_height, y_bot - 1)
            cv2.rectangle(canvas, (0, bar_top), (sidebar_width, bar_bot),
                          (12, 14, 20), -1)

            # Confidence
            conf_str = f"Conf: {conf * 100:.0f}%"
            cv2.putText(canvas, conf_str, (6, info_y + 12),
                        font, font_xs, CONF_GREEN, thick1, cv2.LINE_AA)

            # Area
            area_k = area / 1000.0
            area_str = f"Area: {area_k:.1f}k px"
            (aw, _), _ = cv2.getTextSize(area_str, font, font_xs, thick1)
            cv2.putText(canvas, area_str, (sidebar_width - aw - 4, info_y + 12),
                        font, font_xs, AREA_ORANGE, thick1, cv2.LINE_AA)

            # Frame index (bottom row)
            fidx_str = f"Frame #{fidx}"
            cv2.putText(canvas, fidx_str, (6, info_y + 26),
                        font, font_xs, TEXT_DIM, thick1, cv2.LINE_AA)

            # thin cyan border around thumbnail
            thumb_border_y2 = min(frame_height - 1, info_y - 2)
            cv2.rectangle(canvas, (3, y_top + 1), (sidebar_width - 4, thumb_border_y2),
                          DIVIDER_COLOR, 1)

        else:
            # ---- placeholder slot -----------------------------------------
            cx = sidebar_width // 2
            cy = y_top + slot_h // 2

            # dashed border approximation (series of short lines)
            dash_col = (45, 50, 65)
            for dx in range(6, sidebar_width - 6, 10):
                cv2.line(canvas, (dx, y_top + 3), (min(dx + 6, sidebar_width - 6), y_top + 3),
                         dash_col, 1)
                cv2.line(canvas, (dx, y_bot - 4), (min(dx + 6, sidebar_width - 6), y_bot - 4),
                         dash_col, 1)
            for dy in range(y_top + 6, y_bot - 6, 10):
                cv2.line(canvas, (3, dy), (3, min(dy + 6, y_bot - 6)), dash_col, 1)
                cv2.line(canvas, (sidebar_width - 4, dy),
                         (sidebar_width - 4, min(dy + 6, y_bot - 6)), dash_col, 1)

            # scan icon (simple reticle)
            r = min(18, slot_h // 4)
            cv2.circle(canvas, (cx, cy - 6), r, dash_col, 1, cv2.LINE_AA)
            cv2.line(canvas, (cx - r - 4, cy - 6), (cx + r + 4, cy - 6), dash_col, 1)
            cv2.line(canvas, (cx, cy - r - 10), (cx, cy + r - 2), dash_col, 1)

            # "scanning" text
            scan_text = "Scanning..."
            (stw, _), _ = cv2.getTextSize(scan_text, font, font_xs, thick1)
            cv2.putText(canvas, scan_text, (cx - stw // 2, cy + r + 10),
                        font, font_xs, TEXT_DIM, thick1, cv2.LINE_AA)

    # ---- bottom status strip ----------------------------------------------
    strip_y = frame_height - 22
    if strip_y > header_h:
        cv2.rectangle(canvas, (0, strip_y), (sidebar_width, frame_height), HEADER_BG, -1)
        ts = time.strftime("%H:%M:%S")
        cv2.putText(canvas, ts, (6, frame_height - 6),
                    font, font_xs, TEXT_DIM, thick1, cv2.LINE_AA)

        label_r = "FEED LIVE"
        (lrw, _), _ = cv2.getTextSize(label_r, font, font_xs, thick1)
        # blinking dot (odd second = on)
        dot_color = CONF_GREEN if int(time.time()) % 2 == 0 else TEXT_DIM
        cv2.circle(canvas, (sidebar_width - lrw - 18, frame_height - 10), 4, dot_color, -1)
        cv2.putText(canvas, label_r, (sidebar_width - lrw - 6, frame_height - 6),
                    font, font_xs, dot_color, thick1, cv2.LINE_AA)

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
