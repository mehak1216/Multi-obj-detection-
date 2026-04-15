"""Visualization utilities for tracked detections."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict

import cv2
import numpy as np

from tracker import TrackedObject

MAX_TRAJECTORY_POINTS = 40
BOX_THICKNESS = 2
TEXT_SCALE = 0.55
TEXT_THICKNESS = 2


class TrackVisualizer:
    """Render tracked bounding boxes, IDs, and short trajectories."""

    def __init__(self, draw_trajectories: bool = True) -> None:
        self.draw_trajectories = draw_trajectories
        self.history: DefaultDict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=MAX_TRAJECTORY_POINTS)
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[TrackedObject],
        frame_index: int,
    ) -> np.ndarray:
        """Return a copy of the frame with boxes, labels, and trajectories."""

        annotated = frame.copy()
        for track in tracks:
            self.history[track.track_id].append(track.center)
            color = self._color_for_track(track.track_id)
            x1, y1, x2, y2 = (int(value) for value in track.bbox_xyxy)
            label = f"ID {track.track_id} {track.class_name} {track.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            text_origin = (x1, max(24, y1 - 10))
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                color,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

            if self.draw_trajectories:
                self._draw_trajectory(annotated, track.track_id, color)

        cv2.putText(
            annotated,
            f"Frame: {frame_index}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def export_trajectory_summary(self, canvas: np.ndarray) -> np.ndarray:
        """Overlay all accumulated trajectories on top of a canvas image."""

        summary = canvas.copy()
        for track_id, points in self.history.items():
            color = self._color_for_track(track_id)
            if len(points) > 1:
                cv2.polylines(
                    summary,
                    [np.array(points, dtype=np.int32)],
                    isClosed=False,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            if points:
                cv2.circle(summary, points[-1], 4, color, -1, lineType=cv2.LINE_AA)
        return summary

    def _draw_trajectory(self, frame: np.ndarray, track_id: int, color: tuple[int, int, int]) -> None:
        """Draw the short recent trail for a single track."""

        points = self.history[track_id]
        if len(points) > 1:
            cv2.polylines(
                frame,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    @staticmethod
    def _color_for_track(track_id: int) -> tuple[int, int, int]:
        """Create a deterministic BGR color for a track id."""

        return (
            int((37 * track_id) % 255),
            int((17 * track_id + 91) % 255),
            int((29 * track_id + 53) % 255),
        )
