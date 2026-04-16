# Multi-Object Detection and Persistent ID Tracking

This project detects subjects in public sports footage and assigns persistent IDs across the full video. The reference implementation uses Ultralytics YOLO for detection and BoT-SORT with ReID enabled for tracking.

## Assignment alignment

- Uses a publicly accessible sports/event video source.
- Detects multiple moving subjects and assigns persistent IDs across the full clip.
- Uses BoT-SORT with ReID to better handle occlusion, camera motion, scale changes, and similar-looking subjects.
- Writes a submission-friendly JSON summary that includes the video source link.

## Why this stack

- `YOLO` provides strong real-time detection performance with reliable confidence-scored bounding boxes.
- `BoT-SORT` was chosen over simpler IoU-only trackers because it combines motion cues, global motion compensation, and appearance matching. That makes it a better fit for sports footage with camera pans, short occlusions, scale changes, and similar-looking players.
- The pipeline writes both an annotated output video and a per-frame CSV export so the results are easy to inspect or grade.

## Public video source

- Default public source URL: `https://pixabay.com/videos/running-people-sports-run-walk-294/`
- Running `python main.py` downloads that public clip into `assets/` if it is not already cached locally.
- You can still pass your own publicly accessible source with `--video-path` or change the fallback with `--source-url`.
- If a hosting page blocks automated downloads, download the clip once manually and pass the local MP4 with `--video-path`.

## Project structure

- [main.py](</Users/riteshsingh/Desktop/Multi-Object Detection/main.py>)
- [detector.py](</Users/riteshsingh/Desktop/Multi-Object Detection/detector.py>)
- [tracker.py](</Users/riteshsingh/Desktop/Multi-Object Detection/tracker.py>)
- [visualizer.py](</Users/riteshsingh/Desktop/Multi-Object Detection/visualizer.py>)
- [video_loader.py](</Users/riteshsingh/Desktop/Multi-Object Detection/video_loader.py>)
- [configs/botsort_reid.yaml](</Users/riteshsingh/Desktop/Multi-Object Detection/configs/botsort_reid.yaml>)
- [requirements.txt](</Users/riteshsingh/Desktop/Multi-Object Detection/requirements.txt>)

## Installation

Use Python `3.10`, `3.11`, or `3.12` for this project. The pinned stack in `requirements.txt` is not intended for Python `3.13`, where packages such as NumPy may fall back to a source build on Windows and fail with errors like `NumPy requires GCC >= 8.4`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows with the Python launcher:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the UI

Launch the Streamlit interface:

```bash
source .venv/bin/activate
streamlit run app.py
```

If Streamlit crashes on startup with a `torch.classes` or file-watcher error, this repo now includes a local `.streamlit/config.toml` that disables Streamlit's file watcher to avoid that PyTorch compatibility issue.

The UI lets you:

- upload a video directly from your machine
- paste a local path or public video URL
- use the default public sample video
- tune thresholds, model weights, image size, classes, and device
- preview the annotated output video, trajectory image, CSV, and JSON summary

## Run the pipeline

Run with the default public source:

```bash
python main.py
 python main.py --video-path "/Users/riteshsingh/Downloads/294-135882925.mp4" --video-source-link "https://pixabay.com/videos/running-people-sports-run-walk-294/"
```

The default profile now favors speed by using `yolo11n.pt` and `--image-size 640`. If the nano weights are not present locally, Ultralytics will download them on the first run. Because `yolo11n.pt` requires a newer Ultralytics release, keep the installed package aligned with `requirements.txt` or switch to an older weight such as `yolov8n.pt`.

Run with your own public or local video:

```bash
python main.py --video-path /absolute/path/to/sports_video.mp4
```

Use a manually downloaded local copy while preserving the original public source link for submission:

```bash
python main.py \
  --video-path /absolute/path/to/downloaded_video.mp4 \
  --video-source-link "https://www.youtube.com/watch?v=ep2BlduD6XQ"
```

Use a different public source URL while keeping automatic download enabled:

```bash
python main.py --source-url "https://example.com/public-sports-video-page-or-file"
```

Use a different model, device, and a larger image size:

```bash
python main.py \
  --video-path /absolute/path/to/sports_video.mp4 \
  --model-weights yolo11x.pt \
  --device cuda:0 \
  --image-size 1280
```

## Outputs

- Annotated video: `outputs/output_tracked.mp4`
- Frame-wise tracking export: `outputs/tracking_data.csv`
- Trajectory overview image: `outputs/trajectory_summary.png`
- Run summary JSON with source link and metadata: `outputs/run_summary.json`
- UI runs are also saved under timestamped folders in `outputs/ui_runs/`

CSV columns:

- `frame_id`
- `subject_id`
- `class_id`
- `class_name`
- `x1`, `y1`, `x2`, `y2`
- `confidence`

## Key implementation notes

- The default `--classes 0` tracks persons, which is the most reliable setting for multi-player sports scenes. You can pass other COCO class ids if the assignment rubric expects more classes.
- `persist=True` keeps tracker state across frames so IDs remain stable through the full clip.
- The custom BoT-SORT YAML enables ReID and sparse optical flow global motion compensation to better handle occlusion and moving cameras.
- Trajectory rendering is intentionally short-term rather than full-length to avoid cluttering the output video.
- Public video downloads are cached in `assets/` so repeated runs do not need to re-download the same clip.
- The JSON run summary records the public source URL so it can be included directly in the submission.

## Known limitations

- Persistent IDs are strongest for short-to-medium occlusions. Very long disappearances or complete scene cuts can still create a new ID.
- The default COCO detector is tuned best for people. Small fast-moving objects like a ball may require a task-specific model.
- Runtime and memory usage depend heavily on the chosen YOLO weights and whether inference runs on CPU, CUDA, or Apple Metal.
- Public hosting pages can change over time or block automated downloads. If that happens, download the clip manually once and pass the local MP4 with `--video-path`.
