"""Object detection wrapper for the sports tracking pipeline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.downloads import attempt_download_asset


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
        self.weights = self._resolve_weights(weights)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.classes = list(classes) if classes else None
        self.device = device

        self.model = YOLO(self.weights)

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

    @staticmethod
    def _resolve_weights(weights: str) -> str:
        """Resolve local weights, downloading known Ultralytics assets when needed."""

        project_root = Path(__file__).resolve().parent
        candidate = Path(weights).expanduser()
        if candidate.exists():
            YOLODetector._validate_weights_file(candidate)
            return str(candidate.resolve())

        if not candidate.is_absolute():
            project_candidate = (project_root / candidate).resolve()
            if project_candidate.exists():
                YOLODetector._validate_weights_file(project_candidate)
                return str(project_candidate)

        if candidate.parent != Path("."):
            return weights

        destination = project_root / candidate.name
        if destination.exists():
            YOLODetector._validate_weights_file(destination)
            return str(destination.resolve())

        try:
            downloaded = Path(attempt_download_asset(str(destination)))
        except Exception:
            try:
                downloaded = Path(attempt_download_asset(weights))
            except Exception:
                return weights

        if downloaded.exists():
            if downloaded.resolve() != destination.resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, destination)
            YOLODetector._validate_weights_file(destination)
            return str(destination.resolve())

        return weights

    @staticmethod
    def _validate_weights_file(weights_path: Path) -> None:
        """Raise a clearer error when a .pt weights path actually contains text or HTML."""

        if weights_path.suffix.lower() != ".pt" or not weights_path.is_file():
            return

        header = weights_path.read_bytes()[:512]
        if not header:
            raise RuntimeError(f"Model weights file is empty: {weights_path}")

        text_prefixes = (b"\n", b"\r", b"<", b"{", b"[", b"version ", b"<!DOCTYPE", b"<?xml")
        if any(header.startswith(prefix) for prefix in text_prefixes):
            raise RuntimeError(
                "Model weights file is not a valid PyTorch .pt binary. "
                f"Please replace or delete the bad file and use a real YOLO weights file: {weights_path}"
            )
