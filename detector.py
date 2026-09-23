"""Core YOLO inference and annotation engine for pothole detection."""
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import config
from utils.civil_metrics import (
    TIER_COLORS,
    TIER_SHORT,
    compute_civil_metrics,
    rhi_color,
)
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
    # ---- Civil engineering metrics (populated by detect() when available) ----
    depth_cm: float = 0.0         # estimated crater depth in cm
    area_real_m2: float = 0.0     # estimated real-world surface area (m²)
    volume_m3: float = 0.0        # estimated crater volume (m³)
    weight_kg: float = 0.0        # cold-mix asphalt patch weight (kg)
    tier: str = "Unknown"         # civil severity tier: "Tier 1/2/3"

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
        tracking_mode: bool = False,
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

        # When True, detect() calls model.track(persist=True) with ByteTrack
        # instead of model.predict(), providing persistent cross-frame track IDs.
        self.tracking_mode: bool = tracking_mode

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
        """Execute inference (or tracking) on a single BGR image frame.

        When ``self.tracking_mode`` is ``True``, uses ``model.track()`` with
        ``persist=True`` and the ByteTrack algorithm to obtain persistent
        cross-frame track IDs.  Falls back to ``model.predict()`` gracefully
        if the tracker is unavailable.

        Civil engineering metrics (depth, volume, weight, tier) are computed
        inline for every detection without any disk I/O.

        Args:
            frame: OpenCV BGR image (H×W×3 uint8).

        Returns:
            DetectionResult with smoothed FPS, all detections, and frame dims.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame: frame is empty or None.")

        frame_height, frame_width = frame.shape[:2]
        start_time = time.perf_counter()

        # ---- YOLO inference: track or predict depending on mode ----
        if self.tracking_mode:
            try:
                raw_results = self.model.track(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=config.get_image_size(),
                    device=self.device,
                    verbose=False,
                    persist=True,          # maintain tracker state across frames
                    tracker="bytetrack.yaml",
                )
            except Exception:
                # Graceful fallback if bytetrack.yaml is unavailable
                raw_results = self.model.predict(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=config.get_image_size(),
                    device=self.device,
                    verbose=False,
                )
        else:
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
            self._fps_ema = raw_fps
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
                xyxy    = boxes.xyxy.cpu().numpy()
                confs   = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                # ByteTrack track IDs — None tensor means tracking unavailable
                if self.tracking_mode and boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                else:
                    track_ids = [None] * len(xyxy)

                for box, score, class_id, tid in zip(
                    xyxy, confs, cls_ids, track_ids, strict=False
                ):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # Strict clamping to prevent any boundary slice crash
                    x1 = int(np.clip(x1, 0, frame_width  - 1))
                    y1 = int(np.clip(y1, 0, frame_height - 1))
                    x2 = int(np.clip(x2, 1, frame_width))
                    y2 = int(np.clip(y2, 1, frame_height))

                    score_val = float(score)
                    confidences.append(score_val)

                    # ---- Civil engineering metrics ----
                    metrics = compute_civil_metrics(x1, y1, x2, y2, frame_height)

                    conf_pct   = int(round(score_val * 100))
                    class_name = "Pothole"
                    if hasattr(self.model, "names") and self.model.names:
                        class_name = self.model.names.get(class_id, "Pothole")

                    detections.append(
                        PotholeDetection(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=score_val,
                            class_id=int(class_id),
                            class_name=class_name,
                            label=f"Pothole {conf_pct}%",
                            frame_width=frame_width,
                            frame_height=frame_height,
                            track_id=int(tid) if tid is not None else None,
                            depth_cm=metrics["depth_cm"],
                            area_real_m2=metrics["area_real_m2"],
                            volume_m3=metrics["volume_m3"],
                            weight_kg=metrics["weight_kg"],
                            tier=metrics["tier"],
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
        rhi: Optional[int] = None,
        total_logged: int = 0,
    ) -> np.ndarray:
        """Draw industrial-grade bounding boxes, tier labels, map-pin, and HUD.

        Args:
            frame:        OpenCV BGR image.
            result:       DetectionResult from :meth:`detect`.
            show_hud:     Render the translucent top HUD banner.
            total_count:  Total unique potholes (active + logged).
            override_fps: Custom FPS to display (overrides result.fps).
            whatsapp_msg: Optional WhatsApp dispatch message for bottom bar.
            critical_id:  track_id of the largest pothole (gets map-pin).
            rhi:          Road Health Index 0-100 (None = not computed yet).
            total_logged: Number of permanently logged unique potholes.

        Returns:
            Annotated copy of the input frame (does not modify in-place).
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Pulsing flag (alternates every ~0.4 s) used for critical pothole
        _pulse_on = (int(time.time() * 2.5) % 2) == 0

        # ================================================================
        # 1.  Per-detection bounding boxes, labels, map-pin
        # ================================================================
        for det in result.detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            is_critical = (
                critical_id is not None
                and det.track_id is not None
                and det.track_id == critical_id
            )

            tier_col = TIER_COLORS.get(det.tier, config.BOX_COLOR)

            if is_critical:
                # ---- Critical: thick pulsing red double-border + map-pin ----
                CRIT_OUTER  = (0, 0, 200)
                CRIT_INNER  = (20, 20, 255)
                CRIT_ACCENT = (0, 0, 255)
                outer_thick = 5
                inner_offset = outer_thick + 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              CRIT_OUTER, outer_thick, cv2.LINE_AA)
                if _pulse_on:
                    ix1 = int(np.clip(x1 + inner_offset, 0, width  - 1))
                    iy1 = int(np.clip(y1 + inner_offset, 0, height - 1))
                    ix2 = int(np.clip(x2 - inner_offset, 1, width))
                    iy2 = int(np.clip(y2 - inner_offset, 1, height))
                    if ix2 > ix1 and iy2 > iy1:
                        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2),
                                      CRIT_INNER, 2, cv2.LINE_AA)

                # Critical corner accents
                cl = min(28, max(10, int(min(x2 - x1, y2 - y1) * 0.25)))
                at = outer_thick + 1
                for (ax, ay, bx, by) in [
                    (x1, y1, x1 + cl, y1), (x1, y1, x1, y1 + cl),
                    (x2, y1, x2 - cl, y1), (x2, y1, x2, y1 + cl),
                    (x1, y2, x1 + cl, y2), (x1, y2, x1, y2 - cl),
                    (x2, y2, x2 - cl, y2), (x2, y2, x2, y2 - cl),
                ]:
                    cv2.line(annotated, (ax, ay), (bx, by), CRIT_ACCENT, at, cv2.LINE_AA)

                # ---- Map-pin marker above bounding box centre ----
                pin_cx = (x1 + x2) // 2
                pin_r  = 14
                pin_stem = 16
                pin_head_cy = max(pin_r + 2, y1 - pin_stem - pin_r)
                pin_tip_y   = min(height - 1, pin_head_cy + pin_r + pin_stem)
                pin_fill = (0, 0, 230) if _pulse_on else (0, 0, 180)

                cv2.circle(annotated, (pin_cx, pin_head_cy), pin_r + 3,
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (pin_cx, pin_head_cy), pin_r,
                           pin_fill, -1, cv2.LINE_AA)
                cv2.circle(annotated, (pin_cx, pin_head_cy), pin_r // 3,
                           (20, 20, 20), -1, cv2.LINE_AA)
                cv2.line(annotated,
                         (pin_cx, pin_head_cy + pin_r), (pin_cx, pin_tip_y),
                         (255, 255, 255), 5, cv2.LINE_AA)
                cv2.line(annotated,
                         (pin_cx, pin_head_cy + pin_r), (pin_cx, pin_tip_y),
                         pin_fill, 3, cv2.LINE_AA)
                tip_pts = np.array([
                    [pin_cx - 5, pin_tip_y - 4],
                    [pin_cx + 5, pin_tip_y - 4],
                    [pin_cx,     pin_tip_y + 4],
                ], dtype=np.int32)
                cv2.fillPoly(annotated, [tip_pts], pin_fill, cv2.LINE_AA)

                # Critical badge label
                tier_s = TIER_SHORT.get(det.tier, det.tier)
                label  = f"! LARGEST | {tier_s} | {det.depth_cm:.1f}cm | {det.weight_kg:.1f}kg"
                badge_bg = (0, 0, 180) if _pulse_on else (0, 0, 140)
                badge_col = CRIT_INNER

            else:
                # ---- Normal: tier-coloured box with corner accents ----
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              tier_col, config.BOX_THICKNESS, cv2.LINE_AA)

                cl = min(18, max(6, int(min(x2 - x1, y2 - y1) * 0.2)))
                at = config.BOX_THICKNESS + 1
                acc = config.CORNER_ACCENT_COLOR
                for (ax, ay, bx, by) in [
                    (x1, y1, x1 + cl, y1), (x1, y1, x1, y1 + cl),
                    (x2, y1, x2 - cl, y1), (x2, y1, x2, y1 + cl),
                    (x1, y2, x1 + cl, y2), (x1, y2, x1, y2 - cl),
                    (x2, y2, x2 - cl, y2), (x2, y2, x2, y2 - cl),
                ]:
                    cv2.line(annotated, (ax, ay), (bx, by), acc, at, cv2.LINE_AA)

                # Normal badge label: #ID TierX | depth | weight | conf%
                tid_s  = f"#{det.track_id} " if det.track_id is not None else ""
                tier_s = TIER_SHORT.get(det.tier, "?")
                label  = (
                    f"{tid_s}{tier_s} | "
                    f"{det.depth_cm:.1f}cm | "
                    f"{det.weight_kg:.1f}kg | "
                    f"{det.confidence * 100:.0f}%"
                )
                badge_bg  = tier_col
                badge_col = tier_col

            # ---- Shared badge rendering ----
            font_scale    = 0.50
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness)
            pad_x, pad_y = 6, 4

            if y1 - (text_h + pad_y * 2) > 0:
                badge_y1 = y1 - (text_h + pad_y * 2)
                badge_y2 = y1
                text_y   = y1 - pad_y
            else:
                badge_y1 = y2
                badge_y2 = min(height - 1, y2 + text_h + pad_y * 2)
                text_y   = y2 + text_h + pad_y

            badge_x1 = max(0, x1)
            badge_x2 = min(width - 1, x1 + text_w + pad_x * 2)

            cv2.rectangle(annotated, (badge_x1, badge_y1),
                          (badge_x2, badge_y2), badge_bg, -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1),
                          (badge_x2, badge_y2), badge_col, 1)
            cv2.putText(annotated, label, (badge_x1 + pad_x, text_y),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # ================================================================
        # 2.  Industrial HUD — translucent top banner
        # ================================================================
        if show_hud:
            BANNER_H = 58
            margin   = 10

            # Dark translucent overlay (75% opacity)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, BANNER_H), (8, 10, 14), -1)
            cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

            # Thin accent line below banner
            cv2.line(annotated, (0, BANNER_H), (width, BANNER_H), (50, 55, 70), 1)

            display_fps = override_fps if override_fps is not None else result.fps

            # ---- Tier breakdown counts ----
            t1 = sum(1 for d in result.detections if d.tier == "Tier 1")
            t2 = sum(1 for d in result.detections if d.tier == "Tier 2")
            t3 = sum(1 for d in result.detections if d.tier == "Tier 3")

            # ---- Left section: active hazards + tier breakdown ----
            hazard_col = (40, 140, 255) if result.count > 0 else (80, 210, 80)
            cv2.putText(annotated,
                        f"Hazards: {result.count} active",
                        (margin, 22), font, 0.58, hazard_col, 2, cv2.LINE_AA)

            # Tier pips: small coloured dots + counts
            pip_x = margin
            pip_y = 46
            for tier_label, count, col in [
                ("T1", t1, TIER_COLORS["Tier 1"]),
                ("T2", t2, TIER_COLORS["Tier 2"]),
                ("T3", t3, TIER_COLORS["Tier 3"]),
            ]:
                cv2.circle(annotated, (pip_x + 5, pip_y - 4), 5, col, -1, cv2.LINE_AA)
                pip_txt = f"{tier_label}:{count}"
                cv2.putText(annotated, pip_txt, (pip_x + 14, pip_y),
                            font, 0.40, col, 1, cv2.LINE_AA)
                (pw, _), _ = cv2.getTextSize(pip_txt, font, 0.40, 1)
                pip_x += pw + 28

            # ---- Centre section: Logged + Total unique ----
            logged_total = total_count if total_count is not None else result.count
            log_text  = f"Logged: {total_logged}  Total: {logged_total}"
            (lw, _), _ = cv2.getTextSize(log_text, font, 0.52, 2)
            lx = max(pip_x + 20, (width - lw) // 2)
            cv2.putText(annotated, log_text, (lx, 22),
                        font, 0.52, (200, 210, 230), 2, cv2.LINE_AA)

            # ---- Right section: FPS + RHI ----
            fps_text = f"FPS {display_fps:.1f}"
            (fw, _), _ = cv2.getTextSize(fps_text, font, 0.55, 2)
            cv2.putText(annotated, fps_text,
                        (width - fw - margin, 22),
                        font, 0.55, (200, 210, 230), 2, cv2.LINE_AA)

            if rhi is not None:
                rhi_col  = rhi_color(rhi)
                rhi_text = f"RHI: {rhi}"
                # Small progress bar
                bar_w    = 80
                bar_h    = 8
                bar_x    = width - bar_w - margin
                bar_y    = 34
                # Background track
                cv2.rectangle(annotated,
                              (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                              (40, 42, 50), -1)
                # Filled portion
                filled_w = max(1, int(bar_w * rhi / 100))
                cv2.rectangle(annotated,
                              (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                              rhi_col, -1)
                # RHI label
                (rw, _), _ = cv2.getTextSize(rhi_text, font, 0.48, 1)
                cv2.putText(annotated, rhi_text,
                            (bar_x - rw - 6, bar_y + bar_h),
                            font, 0.48, rhi_col, 1, cv2.LINE_AA)

            # ---- WhatsApp alert bar (bottom) ----
            if whatsapp_msg:
                msg_scale = 0.50
                (mw, mh), _ = cv2.getTextSize(whatsapp_msg, font, msg_scale, 2)
                bar3_h = 34
                bar3_w = mw + 32
                bar3_x = max(margin, (width - bar3_w) // 2)
                bar3_y = height - margin - bar3_h
                cv2.rectangle(annotated,
                              (bar3_x, bar3_y),
                              (bar3_x + bar3_w, bar3_y + bar3_h),
                              (14, 38, 18), -1)
                cv2.rectangle(annotated,
                              (bar3_x, bar3_y),
                              (bar3_x + bar3_w, bar3_y + bar3_h),
                              (40, 210, 80), 2)
                cv2.putText(annotated, whatsapp_msg,
                            (bar3_x + 16, bar3_y + 22),
                            font, msg_scale, (210, 255, 210), 2, cv2.LINE_AA)

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

