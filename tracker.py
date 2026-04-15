"""Persistent multi-object tracking wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics.engine.results import Results

from detector import YOLODetector


@dataclass(frozen=True)
class TrackedObject:
    """Single tracked object for one frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[int, int]:
        """Return the integer center point for trajectory rendering."""

        x1, y1, x2, y2 = self.bbox_xyxy
        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))


class PersistentTracker:
    """Run BoT-SORT tracking on top of YOLO detections."""

    def __init__(self, detector: YOLODetector, tracker_config: str) -> None:
        self.detector = detector
        self.tracker_config = str(Path(tracker_config).resolve())
        self.seen_track_ids: set[int] = set()

    def track(self, frame: np.ndarray) -> list[TrackedObject]:
        """Track all configured subjects in a single frame."""

        # BoT-SORT is used because it combines motion, camera-motion compensation,
        # and appearance matching (ReID), which helps preserve identities through
        # short occlusions, panning, zooming, and similar-looking subjects.
        results = self.detector.model.track(
            source=frame,
            conf=self.detector.confidence_threshold,
            iou=self.detector.iou_threshold,
            imgsz=self.detector.image_size,
            classes=self.detector.classes,
            device=self.detector.device,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )
        return self._to_tracks(results[0])

    @property
    def total_unique_ids(self) -> int:
        """Return the total number of unique tracks observed so far."""

        return len(self.seen_track_ids)

    def _to_tracks(self, result: Results) -> list[TrackedObject]:
        """Convert Ultralytics tracking results into project dataclasses."""

        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)

        tracks: list[TrackedObject] = []
        for bbox, score, class_id, track_id in zip(xyxy, conf, cls, track_ids):
            track = TrackedObject(
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=self.detector.class_name(int(class_id)),
                confidence=float(score),
                bbox_xyxy=tuple(float(value) for value in bbox),
            )
            self.seen_track_ids.add(track.track_id)
            tracks.append(track)
        return tracks
