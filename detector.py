"""Core YOLO inference and annotation engine for pothole detection."""
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import config


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
        """Qualitative severity class based on bbox area relative to frame.

        Thresholds:
            area_ratio < 2%   → "Low"
            area_ratio < 8%   → "Medium"
            area_ratio >= 8%  → "High"

        Returns:
            One of "Low", "Medium", "High", or "Unknown" when frame dims missing.
        """
        ratio = self.area_ratio
        if ratio == 0.0 and (self.frame_width == 0 or self.frame_height == 0):
            return "Unknown"
        if ratio < 0.02:
            return "Low"
        if ratio < 0.08:
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
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}")

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

        # Run YOLO inference
        raw_results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=config.IMAGE_SIZE,
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

                for box, score, class_id in zip(xyxy, confs, cls_ids):
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
    ) -> np.ndarray:
        """Draw bounding boxes, severity labels, corner accents, and HUD overlay.

        Args:
            frame: OpenCV BGR image.
            result: DetectionResult from :meth:`detect`.
            show_hud: Render the count/FPS status bar in the top corners.

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

        for det in result.detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            box_color = _SEVERITY_COLORS.get(det.severity, config.BOX_COLOR)

            # Main bounding box
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

            # Label badge: "Pothole 87% [High]"
            label = f"{det.label} [{det.severity}]"
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

            # FPS card (top-right) — shows smoothed EMA value
            fps_text = f"FPS: {result.fps:.1f}"
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

        return annotated
