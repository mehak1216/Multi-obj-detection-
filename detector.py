"""Object detection wrapper for the sports tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass(frozen=True)
class Detection:
    """Single object detection emitted by the detector."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO detection models."""

    def __init__(
        self,
        weights: str,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: int,
        classes: Sequence[int] | None = None,
        device: str | None = None,
    ) -> None:
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.classes = list(classes) if classes else None
        self.device = device

        self.model = YOLO(weights)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run plain object detection on a single frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        return self._to_detections(results[0])

    def class_name(self, class_id: int) -> str:
        """Resolve a numeric class id into a readable label."""

        return str(self.model.names.get(class_id, class_id))

    def _to_detections(self, result: Results) -> list[Detection]:
        """Convert an Ultralytics result object into project dataclasses."""

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for bbox, score, class_id in zip(xyxy, conf, cls):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=self.class_name(int(class_id)),
                    confidence=float(score),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                )
            )
        return detections
