"""Core YOLO inference and annotation engine for pothole detection."""
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import config
from utils.ui_composer import (
    calculate_pothole_areas,
    combine_views,
    create_thumbnail_sidebar,
)

__all__ = [
    "calculate_pothole_areas",
    "combine_views",
    "create_thumbnail_sidebar",
]


@dataclass
class PotholeDetection:
    """Represents a single detected pothole in a frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    label: str
    # Frame dimensions needed for relative-area severity calculation
    frame_width: int = 0
    frame_height: int = 0
    track_id: Optional[int] = None

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Bounding box area in pixels²."""
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[int, int]:
        """(cx, cy) pixel centroid of the bounding box."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def area_ratio(self) -> float:
        """Fraction of the frame area occupied by this detection (0.0–1.0).

        Returns 0.0 if frame dimensions were not supplied.
        """
        frame_area = self.frame_width * self.frame_height
        if frame_area == 0:
            return 0.0
        return self.area / frame_area

    @property
    def severity(self) -> str:
        """Qualitative severity class based on perspective-normalized bbox area.

        Raw area ratio (bbox_area / frame_area) is divided by a depth scale
        factor derived from the detection's vertical position in the frame.
        This compensates for perspective: as a dashcam approaches a pothole,
        its apparent pixel size grows — without this correction a small pothole
        would be falsely escalated to "High" just because the car drove closer.

        Depth scale is linearly interpolated between:
            DEPTH_FAR_SCALE  (config default: 0.25) at the top of the frame
            DEPTH_NEAR_SCALE (config default: 1.0 ) at the bottom of the frame

        Thresholds (applied to adjusted ratio):
            adjusted_ratio < 2%   → "Low"
            adjusted_ratio < 8%   → "Medium"
            adjusted_ratio >= 8%  → "High"

        Returns:
            One of "Low", "Medium", "High", or "Unknown" when frame dims missing.
        """
        ratio = self.area_ratio
        if ratio == 0.0 and (self.frame_width == 0 or self.frame_height == 0):
            return "Unknown"

        # Depth compensation via Y-position heuristic.
        # y_norm=0 → top of frame (far away); y_norm=1 → bottom (right in front).
        if self.frame_height > 0:
            y_norm = max(0.0, min(1.0, self.centroid[1] / self.frame_height))
            far_s = config.DEPTH_FAR_SCALE
            near_s = config.DEPTH_NEAR_SCALE
            depth_scale = far_s + (near_s - far_s) * y_norm
            adjusted_ratio = ratio / depth_scale
        else:
            adjusted_ratio = ratio

        if adjusted_ratio < 0.02:
            return "Low"
        if adjusted_ratio < 0.08:
            return "Medium"
        return "High"


@dataclass
class DetectionResult:
    """Aggregated output from a single frame/image inference pass."""
    detections: List[PotholeDetection]
    count: int
    fps: float
    inference_time_ms: float
    max_confidence: float
    # Frame dimensions stored here so downstream code never needs the raw frame
    frame_width: int = 0
    frame_height: int = 0


class PotholeDetector:
    """Decoupled computer vision pothole detector powered by Ultralytics YOLO.

    Attributes:
        model_path: Path to the YOLO .pt weights file.
        confidence_threshold: Minimum detection confidence (0–1).
        iou_threshold: NMS IoU overlap threshold (0–1).
        device: Torch device string (e.g. 'cuda', 'cpu').
    """

    # EMA smoothing factor for FPS readout (lower α → smoother but more lag)
    _FPS_EMA_ALPHA: float = 0.10

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        self.model_path = Path(model_path) if model_path else config.MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else config.CONFIDENCE_THRESHOLD
        )
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else config.IOU_THRESHOLD
        )
        self.device = config.get_device()
        self.device_name = config.get_device_name()

        # Smoothed FPS — initialised to 0 until first inference runs
        self._fps_ema: float = 0.0

        # Hooks for future extensibility (WhatsApp bot, GPS tracking, road hazard alerts)
        self._alert_handlers: List[Callable[[DetectionResult, np.ndarray], None]] = []

        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model weights and transfer to optimal compute device."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                f"Please ensure 'pothole.pt' is placed inside '{config.MODELS_DIR}' directory."
            )

        try:
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}") from e

    def register_alert_handler(self, handler: Callable[[DetectionResult, np.ndarray], None]) -> None:
        """Register a callback triggered on every frame that has detections.

        Use this to attach external modules without modifying the core engine:
            - WhatsApp / Telegram alerts
            - GPS coordinate tagging
            - Hazard database writes

        Args:
            handler: Callable accepting (DetectionResult, frame_bgr).
        """
        self._alert_handlers.append(handler)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Execute inference on a single BGR image frame.

        Args:
            frame: OpenCV BGR image (H×W×3 uint8).

        Returns:
            DetectionResult with smoothed FPS, all detections, and frame dims.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame: frame is empty or None.")

        frame_height, frame_width = frame.shape[:2]
        start_time = time.perf_counter()

        # Run YOLO inference (image size auto-selected for device: 320 on CPU, 640 on GPU)
        raw_results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=config.get_image_size(),
            device=self.device,
            verbose=False,
        )

        elapsed = time.perf_counter() - start_time
        inference_time_ms = elapsed * 1000.0
        raw_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # Exponential Moving Average smoothing: avoids single-frame FPS spikes
        if self._fps_ema == 0.0:
            self._fps_ema = raw_fps  # seed with first value
        else:
            self._fps_ema = (
                self._FPS_EMA_ALPHA * raw_fps + (1.0 - self._FPS_EMA_ALPHA) * self._fps_ema
            )
        smoothed_fps = self._fps_ema

        detections: List[PotholeDetection] = []
        confidences: List[float] = []

        if raw_results and len(raw_results) > 0:
            result = raw_results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, score, class_id in zip(xyxy, confs, cls_ids, strict=False):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # Clamp coordinates to valid frame bounds
                    x1 = max(0, min(x1, frame_width - 1))
                    y1 = max(0, min(y1, frame_height - 1))
                    x2 = max(0, min(x2, frame_width - 1))
                    y2 = max(0, min(y2, frame_height - 1))

                    score_val = float(score)
                    confidences.append(score_val)

                    conf_pct = int(round(score_val * 100))
                    label = f"Pothole {conf_pct}%"

                    class_name = "Pothole"
                    if hasattr(self.model, "names") and self.model.names:
                        class_name = self.model.names.get(class_id, "Pothole")

                    detections.append(
                        PotholeDetection(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=score_val,
                            class_id=int(class_id),
                            class_name=class_name,
                            label=label,
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )
                    )

        detection_result = DetectionResult(
            detections=detections,
            count=len(detections),
            fps=smoothed_fps,
            inference_time_ms=inference_time_ms,
            max_confidence=max(confidences) if confidences else 0.0,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # Trigger registered event hooks if potholes detected
        if detection_result.count > 0:
            for handler in self._alert_handlers:
                try:
                    handler(detection_result, frame)
                except Exception:
                    pass

        return detection_result

    def draw_annotations(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        show_hud: bool = True,
        total_count: Optional[int] = None,
        override_fps: Optional[float] = None,
        whatsapp_msg: Optional[str] = None,
        critical_id: Optional[int] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, severity labels, corner accents, and HUD overlay.

        Args:
            frame: OpenCV BGR image.
            result: DetectionResult from :meth:`detect`.
            show_hud: Render the count/FPS status bar in the top corners.
            total_count: Cumulative unique potholes detected across video stream.
            override_fps: Optional custom FPS to display on HUD.
            whatsapp_msg: Optional WhatsApp alert text to display on-screen.
            critical_id: Optional track_id of the pothole to render in
                "CRITICAL / LARGEST" style (thick red border + warning label).
                Supplied by :class:`~utils.ui_composer.CriticalPotholeTracker`.

        Returns:
            Annotated copy of the input frame (does not modify in-place).
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Severity colour coding (BGR)
        _SEVERITY_COLORS = {
            "Low":     (0, 200, 80),    # green
            "Medium":  (0, 165, 255),   # orange
            "High":    (0, 30, 220),    # red
            "Unknown": (20, 120, 255),  # default blue
        }

        # Pulsing flag for the critical pothole (alternates every ~0.4 s)
        _pulse_on = (int(time.time() * 2.5) % 2) == 0

        for det in result.detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            is_critical = (
                critical_id is not None
                and det.track_id is not None
                and det.track_id == critical_id
            )

            if is_critical:
                # ---- Critical pothole: thick double-border red styling ----
                CRIT_OUTER = (0, 0, 200)    # deep red outer rect
                CRIT_INNER = (20, 20, 255)  # bright red inner rect
                CRIT_ACCENT = (0, 0, 255)   # corner accent color
                outer_thick = 5
                inner_thick = 2
                inner_offset = outer_thick + 1

                # Outer rect (always on)
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    CRIT_OUTER, outer_thick, cv2.LINE_AA,
                )
                # Inner rect (pulsing)
                if _pulse_on:
                    ix1 = max(0, x1 + inner_offset)
                    iy1 = max(0, y1 + inner_offset)
                    ix2 = min(width - 1, x2 - inner_offset)
                    iy2 = min(height - 1, y2 - inner_offset)
                    if ix2 > ix1 and iy2 > iy1:
                        cv2.rectangle(
                            annotated, (ix1, iy1), (ix2, iy2),
                            CRIT_INNER, inner_thick, cv2.LINE_AA,
                        )

                # Critical corner accents (longer, thicker)
                corner_len = min(28, max(10, int(min(x2 - x1, y2 - y1) * 0.25)))
                accent_t = outer_thick + 1
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), CRIT_ACCENT, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), CRIT_ACCENT, accent_t, cv2.LINE_AA)

                # ---- Map-pin marker drawn above the bounding box centre ----
                # pin_cx is the horizontal centre of the box; pin touches the top edge
                pin_cx = (x1 + x2) // 2
                pin_r = 14          # radius of the circular pin head
                pin_stem = 16       # length of the stem below the circle
                # keep the whole pin on-screen
                pin_head_cy = max(pin_r + 2, y1 - pin_stem - pin_r)
                pin_tip_y   = min(height - 1, pin_head_cy + pin_r + pin_stem)

                # White halo so the pin is visible on any background
                cv2.circle(annotated, (pin_cx, pin_head_cy), pin_r + 3,
                           (255, 255, 255), -1, cv2.LINE_AA)
                # Pulsing fill: brighter red when pulse is on
                pin_fill = (0, 0, 230) if _pulse_on else (0, 0, 180)
                cv2.circle(annotated, (pin_cx, pin_head_cy), pin_r,
                           pin_fill, -1, cv2.LINE_AA)
                # Dark hole in centre (pin eye)
                cv2.circle(annotated, (pin_cx, pin_head_cy), pin_r // 3,
                           (20, 20, 20), -1, cv2.LINE_AA)
                # Stem: white border line then coloured line
                cv2.line(annotated,
                         (pin_cx, pin_head_cy + pin_r),
                         (pin_cx, pin_tip_y),
                         (255, 255, 255), 5, cv2.LINE_AA)
                cv2.line(annotated,
                         (pin_cx, pin_head_cy + pin_r),
                         (pin_cx, pin_tip_y),
                         pin_fill, 3, cv2.LINE_AA)
                # Teardrop tip (small filled triangle at the bottom of the stem)
                tip_pts = np.array([
                    [pin_cx - 5, pin_tip_y - 4],
                    [pin_cx + 5, pin_tip_y - 4],
                    [pin_cx,     pin_tip_y + 4],
                ], dtype=np.int32)
                cv2.fillPoly(annotated, [tip_pts], pin_fill, cv2.LINE_AA)

                # Critical label with area stats
                det_w = det.x2 - det.x1
                det_h = det.y2 - det.y1
                area_k = det.area / 1000.0
                label = f"! CRITICAL / LARGEST  {det_w}x{det_h}px  Area:{area_k:.1f}k"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.58
                font_thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

                pad_x, pad_y = 8, 6
                # Place badge below the pin head so it doesn't overlap
                badge_ref_y = max(pin_head_cy + pin_r + pin_stem, y1)
                if badge_ref_y - (text_h + pad_y * 2 + 2) > 0:
                    badge_y1 = badge_ref_y - (text_h + pad_y * 2 + 2)
                    badge_y2 = badge_ref_y
                    text_y = badge_ref_y - pad_y - 1
                else:
                    badge_y1 = y2
                    badge_y2 = y2 + (text_h + pad_y * 2 + 2)
                    text_y = y2 + text_h + pad_y + 1

                badge_x1 = max(0, x1)
                badge_x2 = min(width - 1, x1 + text_w + pad_x * 2)

                # Badge background: deep red, with optional pulsing brightness
                badge_bg = (0, 0, 180) if _pulse_on else (0, 0, 140)
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
                # thin highlight border on badge
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), CRIT_INNER, 1)
                cv2.putText(
                    annotated, label, (badge_x1 + pad_x, text_y),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
                )


            else:
                # ---- Normal pothole: existing severity-colour styling ----
                box_color = _SEVERITY_COLORS.get(det.severity, config.BOX_COLOR)

                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    box_color, config.BOX_THICKNESS, cv2.LINE_AA,
                )

                # High-visibility corner accents
                corner_len = min(18, max(6, int(min(x2 - x1, y2 - y1) * 0.2)))
                accent_c = config.CORNER_ACCENT_COLOR
                accent_t = config.BOX_THICKNESS + 1
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), accent_c, accent_t, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), accent_c, accent_t, cv2.LINE_AA)

                # Label badge: e.g. "#1 Pothole 87% [High]"
                track_prefix = f"#{det.track_id} " if det.track_id is not None else ""
                label = f"{track_prefix}{det.label} [{det.severity}]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.60
                font_thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

                pad_x, pad_y = 6, 5
                if y1 - (text_h + pad_y * 2) > 0:
                    badge_y1 = y1 - (text_h + pad_y * 2)
                    badge_y2 = y1
                    text_y = y1 - pad_y
                else:
                    badge_y1 = y1
                    badge_y2 = y1 + (text_h + pad_y * 2)
                    text_y = y1 + text_h + pad_y

                badge_x1 = x1
                badge_x2 = min(width - 1, x1 + text_w + pad_x * 2)

                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
                cv2.putText(
                    annotated, label, (badge_x1 + pad_x, text_y),
                    font, font_scale, config.LABEL_TEXT_COLOR, font_thickness, cv2.LINE_AA,
                )

        # HUD overlay
        if show_hud:
            font = cv2.FONT_HERSHEY_SIMPLEX
            hud_bg = (18, 20, 24)
            card_h = 36
            margin = 12

            # Pothole count card (top-left)
            if total_count is not None:
                count_text = f"Potholes: {result.count} (Total: {total_count})"
            else:
                count_text = f"Potholes: {result.count}"
            status_color = (40, 140, 255) if result.count > 0 else (80, 210, 80)
            (cw, _), _ = cv2.getTextSize(count_text, font, 0.6, 2)
            card_w = cw + 46

            cv2.rectangle(annotated, (margin, margin), (margin + card_w, margin + card_h), hud_bg, -1)
            cv2.rectangle(annotated, (margin, margin), (margin + card_w, margin + card_h), (60, 60, 60), 1)
            cv2.circle(annotated, (margin + 16, margin + card_h // 2), 6, status_color, -1)
            cv2.putText(
                annotated, count_text, (margin + 32, margin + 24),
                font, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

            # FPS card (top-right) — shows smoothed EMA value or pacing FPS
            display_fps = override_fps if override_fps is not None else result.fps
            fps_text = f"FPS: {display_fps:.1f}"
            (fw, _), _ = cv2.getTextSize(fps_text, font, 0.6, 2)
            card2_w = fw + 32
            x2_card = width - margin - card2_w

            if x2_card > margin + card_w + 10:
                cv2.rectangle(annotated, (x2_card, margin), (x2_card + card2_w, margin + card_h), hud_bg, -1)
                cv2.rectangle(annotated, (x2_card, margin), (x2_card + card2_w, margin + card_h), (60, 60, 60), 1)
                cv2.putText(
                    annotated, fps_text, (x2_card + 16, margin + 24),
                    font, 0.6, (240, 240, 240), 2, cv2.LINE_AA,
                )

            if whatsapp_msg:
                msg_font_scale = 0.55
                (mw, mh), _ = cv2.getTextSize(whatsapp_msg, font, msg_font_scale, 2)
                card3_h = 36
                card3_w = mw + 36
                card3_x = max(margin, (width - card3_w) // 2)
                card3_y = height - margin - card3_h

                cv2.rectangle(annotated, (card3_x, card3_y), (card3_x + card3_w, card3_y + card3_h), (16, 40, 20), -1)
                cv2.rectangle(annotated, (card3_x, card3_y), (card3_x + card3_w, card3_y + card3_h), (40, 210, 80), 2)
                cv2.putText(
                    annotated, whatsapp_msg, (card3_x + 18, card3_y + 24),
                    font, msg_font_scale, (220, 255, 220), 2, cv2.LINE_AA,
                )

        return annotated


class ThreadedInferencePipeline:
    """Runs YOLO inference in a background thread so video I/O never stalls.

    The main loop feeds raw frames into an input queue; the worker thread
    pulls frames, runs ``PotholeDetector.detect()``, and pushes
    ``DetectionResult`` objects into an output queue.  The main loop reads
    results without blocking beyond a short timeout, and falls back to the
    last known result when the worker hasn't finished yet.

    Usage::

        with ThreadedInferencePipeline(detector) as pipe:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pipe.put(frame)
                result = pipe.get()   # returns last known result if worker busy
                ...

    Args:
        detector: An initialised :class:`PotholeDetector`.
        maxsize: Input queue depth.  Keep at 1 to always run inference on the
            freshest frame (drop stale frames when the queue is full).
    """

    def __init__(self, detector: "PotholeDetector", maxsize: int = 1) -> None:
        self._detector = detector
        self._in_q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._out_q: queue.Queue = queue.Queue(maxsize=2)
        self._last_result: Optional[DetectionResult] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="inference-worker"
        )

    def start(self) -> "ThreadedInferencePipeline":
        """Start the background inference thread."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        try:
            self._in_q.put_nowait(None)  # sentinel to unblock worker
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)

    def put(self, frame: np.ndarray) -> None:
        """Submit a frame for inference.

        If the worker is still busy (queue full), the stale pending frame is
        discarded so the worker always processes the most recent input.
        """
        try:
            self._in_q.put_nowait(frame)
        except queue.Full:
            try:
                self._in_q.get_nowait()   # discard stale frame
            except queue.Empty:
                pass
            try:
                self._in_q.put_nowait(frame)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.04) -> Optional[DetectionResult]:
        """Return the most recent completed ``DetectionResult``.

        Blocks up to *timeout* seconds for a new result; returns the cached
        last result (or ``None`` before the first result arrives) if the
        worker hasn't finished yet.
        """
        try:
            result = self._out_q.get(timeout=timeout)
            self._last_result = result
        except queue.Empty:
            result = self._last_result
        return result

    def _worker(self) -> None:
        """Background thread body: frames in → detect() → results out."""
        while not self._stop_event.is_set():
            try:
                frame = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break   # sentinel — shut down cleanly
            try:
                result = self._detector.detect(frame)
                # Drain output queue before pushing to avoid stale accumulation
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
                self._out_q.put_nowait(result)
            except Exception:
                pass

    # ---- context manager ----
    def __enter__(self) -> "ThreadedInferencePipeline":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


class TrackedPothole:
    """Represents a single pothole tracked across consecutive video frames."""

    def __init__(self, track_id: int, initial_det: PotholeDetection):
        self.track_id = track_id
        self.box = (initial_det.x1, initial_det.y1, initial_det.x2, initial_det.y2)
        self.centroid = initial_det.centroid
        self.max_confidence = initial_det.confidence
        self.hits = 1
        self.age = 0
        self.severities = [initial_det.severity]

    def update(self, det: PotholeDetection) -> None:
        self.box = (det.x1, det.y1, det.x2, det.y2)
        self.centroid = det.centroid
        self.max_confidence = max(self.max_confidence, det.confidence)
        self.hits += 1
        self.age = 0
        self.severities.append(det.severity)

    @property
    def peak_severity(self) -> str:
        order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
        return max(self.severities, key=lambda s: order.get(s, 0))


class PotholeTracker:
    """Tracks potholes across video frames to count unique potholes and monitor peak confidence.

    Matches detections across consecutive frames using IoU (Intersection-over-Union)
    and normalized centroid proximity, keeping track of unique pothole IDs and
    the highest confidence score seen across the entire video stream.
    """

    def __init__(
        self,
        iou_threshold: float = 0.20,
        max_age: int = 15,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: dict[int, TrackedPothole] = {}
        self.max_confidence_overall: float = 0.0
        self.total_detection_instances: int = 0

    def update(
        self,
        detections: List[PotholeDetection],
        frame_width: int,
        frame_height: int,
    ) -> int:
        """Update tracker with detections from the current frame.

        Populates ``det.track_id`` on matched/new detections.

        Returns:
            Current count of confirmed unique potholes.
        """
        self.total_detection_instances += len(detections)
        for det in detections:
            if det.confidence > self.max_confidence_overall:
                self.max_confidence_overall = det.confidence

        diag = (frame_width**2 + frame_height**2)**0.5 if (frame_width and frame_height) else 1.0

        active_tracks = [t for t in self.tracks.values() if t.age <= self.max_age]
        matched_track_ids = set()
        matched_det_indices = set()

        for track in active_tracks:
            best_score = 0.0
            best_idx = None
            tx1, ty1, tx2, ty2 = track.box
            tcx, tcy = track.centroid

            for di, det in enumerate(detections):
                if di in matched_det_indices:
                    continue
                # Bounding box IoU
                ix1 = max(tx1, det.x1)
                iy1 = max(ty1, det.y1)
                ix2 = min(tx2, det.x2)
                iy2 = min(ty2, det.y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (tx2 - tx1) * (ty2 - ty1) + det.area - inter
                iou = inter / union if union > 0 else 0.0

                # Centroid proximity normalized by frame diagonal
                dcx, dcy = det.centroid
                cdist = ((tcx - dcx)**2 + (tcy - dcy)**2)**0.5 / diag

                score = iou
                if score == 0.0 and cdist < 0.10:
                    score = 0.25

                if score > best_score:
                    best_score = score
                    best_idx = di

            if best_idx is not None and best_score >= self.iou_threshold:
                det = detections[best_idx]
                track.update(det)
                det.track_id = track.track_id
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(best_idx)

        # Increment age for active tracks not matched in this frame
        for track in self.tracks.values():
            if track.track_id not in matched_track_ids:
                track.age += 1

        # Unmatched detections initialize new tracks
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                tid = self.next_id
                self.next_id += 1
                new_track = TrackedPothole(tid, det)
                self.tracks[tid] = new_track
                det.track_id = tid

        return self.unique_count

    @property
    def unique_count(self) -> int:
        """Count of confirmed unique potholes (seen in at least min_hits frames)."""
        confirmed = [t for t in self.tracks.values() if t.hits >= self.min_hits]
        if confirmed:
            return len(confirmed)
        return len(self.tracks) if self.tracks else 0

    @property
    def highest_confidence(self) -> float:
        """Maximum confidence score observed across all detections."""
        return self.max_confidence_overall

