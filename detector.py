"""Core YOLO inference and annotation engine for pothole detection."""
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import config


@dataclass
class PotholeDetection:
    """Represents a detected pothole."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    label: str

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectionResult:
    """Aggregated output from a single frame/image inference."""
    detections: List[PotholeDetection]
    count: int
    fps: float
    inference_time_ms: float
    max_confidence: float


class PotholeDetector:
    """Decoupled computer vision pothole detector powered by Ultralytics YOLO."""

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
        """Register a callback for future extension (e.g. WhatsApp, GPS, Hazard DB)."""
        self._alert_handlers.append(handler)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Execute inference on a single image frame (BGR format).

        Args:
            frame: OpenCV BGR image array.

        Returns:
            DetectionResult containing detected bounding boxes, confidence, count, and FPS.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame: frame is empty or None.")

        height, width = frame.shape[:2]
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
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

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
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    score_val = float(score)
                    confidences.append(score_val)

                    # Required format: "Pothole 87%"
                    conf_pct = int(round(score_val * 100))
                    label = f"Pothole {conf_pct}%"

                    class_name = "Pothole"
                    if hasattr(self.model, "names") and self.model.names:
                        class_name = self.model.names.get(class_id, "Pothole")

                    detections.append(
                        PotholeDetection(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=score_val,
                            class_id=int(class_id),
                            class_name=class_name,
                            label=label,
                        )
                    )

        detection_result = DetectionResult(
            detections=detections,
            count=len(detections),
            fps=fps,
            inference_time_ms=inference_time_ms,
            max_confidence=max(confidences) if confidences else 0.0,
        )

        # Trigger registered event hooks if potholes detected (extensibility hook)
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
        """Draw bounding boxes, labels, and HUD on the frame.

        Args:
            frame: OpenCV BGR image.
            result: DetectionResult containing detections.
            show_hud: Whether to render the count and FPS status bar.

        Returns:
            Annotated OpenCV BGR image.
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Draw each detected pothole
        for det in result.detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

            # Main bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                config.BOX_COLOR,
                config.BOX_THICKNESS,
                cv2.LINE_AA,
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

            # Label badge: "Pothole 87%"
            label = det.label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            font_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

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

            # Solid background for label
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                config.LABEL_BG_COLOR,
                -1,
            )

            # White label text
            cv2.putText(
                annotated,
                label,
                (badge_x1 + pad_x, text_y),
                font,
                font_scale,
                config.LABEL_TEXT_COLOR,
                font_thickness,
                cv2.LINE_AA,
            )

        # Draw HUD
        if show_hud:
            font = cv2.FONT_HERSHEY_SIMPLEX
            hud_bg = (18, 20, 24)
            card_h = 36
            margin = 12

            # 1. Pothole Count Card (Top-Left)
            count_text = f"Potholes: {result.count}"
            status_color = (40, 140, 255) if result.count > 0 else (80, 210, 80)
            (cw, ch), _ = cv2.getTextSize(count_text, font, 0.6, 2)
            card_w = cw + 46

            cv2.rectangle(annotated, (margin, margin), (margin + card_w, margin + card_h), hud_bg, -1)
            cv2.rectangle(annotated, (margin, margin), (margin + card_w, margin + card_h), (60, 60, 60), 1)
            cv2.circle(annotated, (margin + 16, margin + card_h // 2), 6, status_color, -1)
            cv2.putText(
                annotated,
                count_text,
                (margin + 32, margin + 24),
                font,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 2. FPS Card (Top-Right)
            fps_text = f"FPS: {result.fps:.1f}"
            (fw, fh), _ = cv2.getTextSize(fps_text, font, 0.6, 2)
            card2_w = fw + 32
            x2_card = width - margin - card2_w

            if x2_card > margin + card_w + 10:
                cv2.rectangle(annotated, (x2_card, margin), (x2_card + card2_w, margin + card_h), hud_bg, -1)
                cv2.rectangle(annotated, (x2_card, margin), (x2_card + card2_w, margin + card_h), (60, 60, 60), 1)
                cv2.putText(
                    annotated,
                    fps_text,
                    (x2_card + 16, margin + 24),
                    font,
                    0.6,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )

        return annotated
