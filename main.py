"""End-to-end multi-object detection and persistent ID tracking pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics.utils.downloads import attempt_download_asset

from detector import YOLODetector
from tracker import PersistentTracker, TrackedObject
from video_loader import VideoSource, resolve_video_source
from visualizer import TrackVisualizer

DEFAULT_VIDEO_SOURCE_URL = "https://pixabay.com/videos/running-people-sports-run-walk-294/"
DEFAULT_TRACKER_CONFIG = "configs/botsort_reid.yaml"
DEFAULT_MODEL_WEIGHTS = "yolo11n.pt"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_OUTPUT_VIDEO = "outputs/output_tracked.mp4"
DEFAULT_OUTPUT_CSV = "outputs/tracking_data.csv"
DEFAULT_TRAJECTORY_IMAGE = "outputs/trajectory_summary.png"
DEFAULT_PREVIEW_GIF = "outputs/output_preview.gif"
DEFAULT_SUMMARY_JSON = "outputs/run_summary.json"
DEFAULT_DOWNLOAD_DIR = "assets"

LOGGER = logging.getLogger("sports_tracking")


def resolve_model_weights(weights: str) -> str:
    """Resolve local model weights, downloading the default asset when missing."""

    candidate = Path(weights).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    # Only auto-download bare asset names like "yolo11n.pt".
    if candidate.parent != Path("."):
        return weights

    destination = Path.cwd() / candidate.name
    if destination.exists():
        return str(destination.resolve())

    downloaded = Path(attempt_download_asset(str(destination)))
    if downloaded.exists():
        return str(downloaded.resolve())
    return weights


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tracking pipeline."""

    parser = argparse.ArgumentParser(
        description="Detect and persistently track subjects in public sports footage."
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Local video path or public video URL. If omitted, the configured public source is used.",
    )
    parser.add_argument(
        "--download-default-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the configured public sample video when --video-path is not provided.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=DEFAULT_VIDEO_SOURCE_URL,
        help="Public page URL or direct video URL used when --download-default-video is enabled.",
    )
    parser.add_argument(
        "--video-source-link",
        type=str,
        default=None,
        help="Original public source link for the video, useful when you downloaded a public clip manually.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS,
        help="YOLO detection weights. Defaults to the faster nano model for quicker runs.",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default=DEFAULT_TRACKER_CONFIG,
        help="Path to the BoT-SORT tracker YAML config.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="YOLO inference image size. Lower defaults trade some accuracy for faster processing.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0",
        help="Comma-separated COCO class ids to track. Default tracks persons only.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device string such as cpu, cuda:0, or mps.")
    parser.add_argument("--output-video", type=str, default=DEFAULT_OUTPUT_VIDEO, help="Annotated output MP4 path.")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Per-frame tracking CSV path.")
    parser.add_argument(
        "--trajectory-image",
        type=str,
        default=DEFAULT_TRAJECTORY_IMAGE,
        help="Optional path for a trajectory summary image.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=DEFAULT_SUMMARY_JSON,
        help="Run summary JSON path, including the public source link used for submission.",
    )
    parser.add_argument("--download-dir", type=str, default=DEFAULT_DOWNLOAD_DIR, help="Download/cache directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick testing.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def parse_classes(classes_arg: str) -> list[int] | None:
    """Parse a comma-separated class-id list into integers."""

    cleaned = classes_arg.strip()
    if not cleaned:
        return None
    return [int(token.strip()) for token in cleaned.split(",") if token.strip()]


def configure_logging(log_level: str) -> None:
    """Configure structured logging for the script."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def build_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create an OpenCV writer for the annotated output video."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if fps and fps > 0 else 25.0
    # Prefer browser-friendly codecs first so Streamlit can preview the result inline.
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (width, height))
        if writer.isOpened():
            LOGGER.info("Opened video writer with codec %s", codec)
            return writer
        writer.release()

    raise RuntimeError(
        "Failed to create an output video writer. No supported codec was available for this OpenCV build."
    )


def open_csv_writer(csv_path: Path) -> tuple[csv.DictWriter, object]:
    """Create a CSV writer for per-frame tracking rows."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["frame_id", "subject_id", "class_id", "class_name", "x1", "y1", "x2", "y2", "confidence"],
    )
    writer.writeheader()
    return writer, csv_file


def write_tracking_rows(writer: csv.DictWriter, frame_id: int, tracks: list[TrackedObject]) -> None:
    """Persist the tracking outputs for one frame."""

    for track in tracks:
        x1, y1, x2, y2 = track.bbox_xyxy
        writer.writerow(
            {
                "frame_id": frame_id,
                "subject_id": track.track_id,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "x1": f"{x1:.2f}",
                "y1": f"{y1:.2f}",
                "x2": f"{x2:.2f}",
                "y2": f"{y2:.2f}",
                "confidence": f"{track.confidence:.4f}",
            }
        )


def write_summary(summary_path: Path, summary: dict[str, float | int | str | bool]) -> None:
    """Persist the run summary as JSON for easy submission packaging."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def export_preview_gif(preview_path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write a lightweight browser-safe animated preview for environments without H.264 encoding."""

    if not frames:
        return

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    frame_duration_ms = max(int(1000 / max(fps, 1.0)), 40)
    pil_frames[0].save(
        preview_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    """Run the full detection, tracking, annotation, and export pipeline."""

    source: VideoSource = resolve_video_source(
        video_path=args.video_path,
        default_source_url=args.source_url,
        download_dir=args.download_dir,
        auto_download=args.download_default_video,
    )
    LOGGER.info("Using input video: %s", source.source_path)
    if source.source_url:
        LOGGER.info("Public source URL: %s", source.source_url)

    model_weights = resolve_model_weights(args.model_weights)
    LOGGER.info("Model weights: %s", model_weights)

    detector = YOLODetector(
        weights=model_weights,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.image_size,
        classes=parse_classes(args.classes),
        device=args.device,
    )
    tracker = PersistentTracker(detector=detector, tracker_config=args.tracker_config)
    visualizer = TrackVisualizer(draw_trajectories=True)

    capture = cv2.VideoCapture(str(source.source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {source.source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid dimensions.")

    output_video_path = Path(args.output_video).expanduser().resolve()
    output_csv_path = Path(args.output_csv).expanduser().resolve()
    trajectory_image_path = Path(args.trajectory_image).expanduser().resolve()
    preview_gif_path = output_video_path.with_name("output_preview.gif")
    summary_json_path = Path(args.summary_json).expanduser().resolve()

    writer = build_writer(output_video_path, fps, width, height)
    csv_writer, csv_handle = open_csv_writer(output_csv_path)

    frame_index = 0
    last_annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
    preview_frames: list[np.ndarray] = []
    start_time = time.perf_counter()

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame or frame is None:
                break

            frame_index += 1
            tracks = tracker.track(frame)
            annotated = visualizer.annotate(frame, tracks, frame_index)
            last_annotated_frame = annotated

            writer.write(annotated)
            write_tracking_rows(csv_writer, frame_index, tracks)
            if len(preview_frames) < 90 and frame_index % 4 == 0:
                preview_frames.append(
                    cv2.resize(annotated, (min(width, 720), int(height * min(width, 720) / width)))
                )

            if args.max_frames is not None and frame_index >= args.max_frames:
                LOGGER.info("Reached --max-frames=%s, stopping early.", args.max_frames)
                break

            if frame_index % 25 == 0:
                LOGGER.info(
                    "Processed %d frames | active tracks=%d | unique ids=%d",
                    frame_index,
                    len(tracks),
                    tracker.total_unique_ids,
                )
    finally:
        capture.release()
        writer.release()
        csv_handle.close()

    if frame_index and frame_index % 25 != 0:
        LOGGER.info(
            "Processed %d frames | active tracks=%d | unique ids=%d",
            frame_index,
            len(tracks),
            tracker.total_unique_ids,
        )

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    average_fps = frame_index / elapsed
    trajectory_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(trajectory_image_path),
        visualizer.export_trajectory_summary(last_annotated_frame),
    )
    export_preview_gif(preview_gif_path, preview_frames, fps / 4 if fps > 0 else 6.0)

    summary = {
        "input_video": str(source.source_path),
        "source_url": args.video_source_link or source.source_url or "local file",
        "downloaded_from_public_source": source.downloaded,
        "model_weights": model_weights,
        "tracker_config": str(Path(args.tracker_config).expanduser().resolve()),
        "output_video": str(output_video_path),
        "preview_gif": str(preview_gif_path),
        "tracking_csv": str(output_csv_path),
        "trajectory_image": str(trajectory_image_path),
        "summary_json": str(summary_json_path),
        "frames_processed": frame_index,
        "total_unique_ids": tracker.total_unique_ids,
        "processing_time_seconds": round(elapsed, 2),
        "avg_fps": round(average_fps, 2),
    }
    write_summary(summary_json_path, summary)
    return summary


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    summary = run_pipeline(args)

    LOGGER.info("Run complete.")
    LOGGER.info("Frames processed: %s", summary["frames_processed"])
    LOGGER.info("Total unique IDs: %s", summary["total_unique_ids"])
    LOGGER.info("Processing time (s): %s", summary["processing_time_seconds"])
    LOGGER.info("Average FPS: %s", summary["avg_fps"])
    LOGGER.info("Annotated video: %s", summary["output_video"])
    LOGGER.info("Tracking CSV: %s", summary["tracking_csv"])
    LOGGER.info("Trajectory image: %s", summary["trajectory_image"])
    LOGGER.info("Run summary JSON: %s", summary["summary_json"])
    LOGGER.info("Source link: %s", summary["source_url"])


if __name__ == "__main__":
    main()
