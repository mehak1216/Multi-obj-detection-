"""Streamlit UI for the multi-object detection and tracking pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from main import (
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_TRACKER_CONFIG,
    DEFAULT_VIDEO_SOURCE_URL,
    configure_logging,
    run_pipeline,
)

APP_TITLE = "Multi-Object Detection Studio"
UPLOAD_DIR = Path("assets/ui_uploads")
RUNS_DIR = Path("outputs/ui_runs")


def ensure_logging() -> None:
    """Configure logging once for the Streamlit process."""

    configure_logging("INFO")


def apply_theme() -> None:
    """Inject custom styling so the Streamlit app feels more product-like."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #16232d;
            --muted: #60707a;
            --accent: #d65a31;
            --accent-soft: #fff1e7;
            --teal: #2f7f86;
            --sand: #f6ead8;
            --cream: #fffaf2;
            --line: rgba(22, 35, 45, 0.09);
            --shadow: 0 24px 60px rgba(58, 43, 23, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(214, 90, 49, 0.18), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(47, 127, 134, 0.16), transparent 24%),
                linear-gradient(180deg, #fbf3e7 0%, #f2e7d7 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            letter-spacing: -0.02em;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(255, 251, 244, 0.95), rgba(246, 234, 216, 0.88)),
                linear-gradient(135deg, rgba(214, 90, 49, 0.1), transparent);
            border: 1px solid var(--line);
            border-radius: 30px;
            padding: 1.4rem 1.4rem 1.2rem;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
            margin-bottom: 1rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -60px -70px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(47, 127, 134, 0.14), transparent 70%);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-size: 0.72rem;
            color: var(--accent);
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .hero p {
            color: var(--muted);
            max-width: 54rem;
            margin-bottom: 0.9rem;
            font-size: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .hero-stat, .surface-card, .mini-card {
            background: rgba(255, 252, 247, 0.86);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }

        .hero-stat {
            border-radius: 22px;
            padding: 0.95rem 1rem;
        }

        .hero-stat-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }

        .hero-stat-value {
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--ink);
        }

        .surface-card {
            border-radius: 26px;
            padding: 1rem 1rem 0.2rem;
            margin-top: 0.75rem;
        }

        .mini-card {
            border-radius: 22px;
            padding: 0.9rem 1rem;
            height: 100%;
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.35rem;
        }

        .mini-value {
            color: var(--ink);
            font-size: 1.1rem;
            font-weight: 700;
        }

        .source-link {
            display: block;
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.45;
            word-break: break-word;
            overflow-wrap: anywhere;
            text-decoration: none;
        }

        .source-link:hover {
            color: var(--accent);
            text-decoration: underline;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 252, 247, 0.86);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.8rem 1rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stForm"] {
            background: rgba(255, 252, 247, 0.86);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1rem 1rem 0.4rem;
            box-shadow: var(--shadow);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 250, 244, 0.96);
            padding: 0.55rem 1rem;
            color: var(--ink) !important;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: var(--accent-soft);
            color: var(--accent) !important;
            border-color: rgba(214, 90, 49, 0.18);
        }

        .stDownloadButton > button {
            background: var(--accent) !important;
            color: #fffaf4 !important;
            border: 1px solid rgba(214, 90, 49, 0.28) !important;
            font-weight: 700 !important;
        }

        .stDownloadButton > button:hover {
            background: #bf4b25 !important;
            color: #ffffff !important;
        }

        .stDownloadButton > button p,
        .stButton > button p,
        .stFormSubmitButton > button p {
            color: inherit !important;
            font-weight: 700 !important;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            background: #16232d !important;
            color: #fffaf4 !important;
            border: 1px solid rgba(22, 35, 45, 0.18) !important;
            font-weight: 700 !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #223340 !important;
            color: #ffffff !important;
        }

        .stButton > button:disabled,
        .stDownloadButton > button:disabled,
        .stFormSubmitButton > button:disabled {
            background: #d8d2c8 !important;
            color: #5d676e !important;
            border-color: rgba(22, 35, 45, 0.12) !important;
            opacity: 1 !important;
        }

        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] label p,
        div[role="radiogroup"] label,
        div[role="radiogroup"] label p {
            color: var(--ink) !important;
            font-weight: 700 !important;
        }

        div[role="radiogroup"] {
            background: rgba(255, 252, 247, 0.9);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 0.85rem 1rem 0.35rem;
            box-shadow: var(--shadow);
        }

        .section-note {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: -0.1rem;
            margin-bottom: 0.6rem;
        }

        .json-box {
            background: #fffaf4;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.8rem;
        }

        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file) -> Path:
    """Persist the uploaded file and return its local path."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    target.write_bytes(uploaded_file.getbuffer())
    return target.resolve()


def infer_video_source_link(source_mode: str, uploaded_file, source_text: str) -> str | None:
    """Infer the public source link for known local copies of the default sample."""

    if source_mode == "Use default public sample":
        return DEFAULT_VIDEO_SOURCE_URL

    if source_mode == "Use local path or URL":
        cleaned = source_text.strip()
        if cleaned.startswith(("http://", "https://")):
            return None

    return None


def build_args(
    source_mode: str,
    uploaded_file,
    source_text: str,
    model_weights: str,
    tracker_config: str,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    classes: str,
    device: str,
    max_frames: int,
) -> SimpleNamespace:
    """Build a namespace compatible with the existing CLI pipeline."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (RUNS_DIR / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path: str | None = None
    auto_download = False
    source_url = DEFAULT_VIDEO_SOURCE_URL
    video_source_link = infer_video_source_link(source_mode, uploaded_file, source_text)

    if source_mode == "Use uploaded file":
        if uploaded_file is None:
            raise ValueError("Please upload a video file before starting the run.")
        video_path = str(save_uploaded_file(uploaded_file))
    elif source_mode == "Use local path or URL":
        cleaned = source_text.strip()
        if not cleaned:
            raise ValueError("Please provide a local video path or a public video URL.")
        video_path = cleaned
    else:
        auto_download = True
        source_url = source_text.strip() or DEFAULT_VIDEO_SOURCE_URL

    return SimpleNamespace(
        video_path=video_path,
        download_default_video=auto_download,
        source_url=source_url,
        video_source_link=video_source_link,
        model_weights=model_weights.strip() or DEFAULT_MODEL_WEIGHTS,
        tracker_config=tracker_config.strip() or DEFAULT_TRACKER_CONFIG,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        classes=classes.strip() or "0",
        device=device.strip() or None,
        output_video=str(run_dir / "output_tracked.mp4"),
        output_csv=str(run_dir / "tracking_data.csv"),
        trajectory_image=str(run_dir / "trajectory_summary.png"),
        summary_json=str(run_dir / "run_summary.json"),
        download_dir=DEFAULT_DOWNLOAD_DIR,
        max_frames=max_frames if max_frames > 0 else None,
        log_level="INFO",
    )


def render_header() -> None:
    """Render the top hero section."""

    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">Tracking Workspace</div>
            <h1>{APP_TITLE}</h1>
            <p>
                A cleaner front-end for your existing YOLO + BoT-SORT pipeline. The detection and
                tracking logic stays untouched; this interface just makes runs, outputs, and JSON artifacts
                easier to inspect and present.
            </p>
            <div class="hero-grid">
                <div class="hero-stat">
                    <div class="hero-stat-label">Pipeline</div>
                    <div class="hero-stat-value">YOLO Detection + Persistent IDs</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Outputs</div>
                    <div class="hero-stat-value">Video, CSV, Trajectory PNG, JSON</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Promise</div>
                    <div class="hero-stat-value">UI upgrade only, no logic changes</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_overview_cards(summary: dict[str, object]) -> None:
    """Render high-level run summary cards."""

    cards = st.columns([1, 1, 1, 1, 1.25])
    fps_text = f'{float(summary["avg_fps"]):.2f}'
    processing_time = summary.get("processing_time_seconds")
    processing_time_text = f"{float(processing_time):.2f} s" if processing_time is not None else "N/A"
    items = [
        ("Frames Processed", str(int(summary["frames_processed"]))),
        ("Unique IDs", str(int(summary["total_unique_ids"]))),
        ("Processing Time", processing_time_text),
        ("Average FPS", fps_text),
    ]

    for column, (label, value) in zip(cards[:4], items):
        with column:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-label">{label}</div>
                    <div class="mini-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    source_url = str(summary["source_url"])
    with cards[4]:
        if source_url.startswith(("http://", "https://")):
            source_markup = f'<a class="source-link" href="{source_url}" target="_blank">{source_url}</a>'
        else:
            source_markup = f'<div class="source-link">{source_url}</div>'

        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">Source</div>
                {source_markup}
            </div>
            """,
            unsafe_allow_html=True,
        )


def load_tracking_dataframe(csv_path: Path) -> pd.DataFrame | None:
    """Load the CSV export if present."""

    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def render_analytics(dataframe: pd.DataFrame | None) -> None:
    """Render lightweight analytics derived from the tracking CSV."""

    st.markdown("### Tracking Analytics")
    st.markdown(
        '<p class="section-note">These charts are derived from the existing CSV export and do not alter pipeline behavior.</p>',
        unsafe_allow_html=True,
    )

    if dataframe is None or dataframe.empty:
        st.info("Run analytics will appear here once a tracking CSV is available.")
        return

    analytics_col1, analytics_col2 = st.columns(2, gap="large")

    with analytics_col1:
        counts_per_id = (
            dataframe.groupby("subject_id")
            .size()
            .sort_values(ascending=False)
            .rename("tracked_frames")
            .head(12)
        )
        st.markdown("#### Most Active IDs")
        st.bar_chart(counts_per_id)

    with analytics_col2:
        detections_by_frame = (
            dataframe.groupby("frame_id")
            .size()
            .rename("detections")
        )
        st.markdown("#### Detections Across Frames")
        st.line_chart(detections_by_frame)

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("CSV Rows", len(dataframe))
    summary_col2.metric("Classes Seen", dataframe["class_name"].nunique())
    summary_col3.metric("Highest Frame", int(dataframe["frame_id"].max()))


def render_artifact_downloads(summary: dict[str, object]) -> None:
    """Render artifact download buttons."""

    st.markdown("### Export Bundle")
    st.markdown(
        '<p class="section-note">Download the exact files produced by the existing pipeline.</p>',
        unsafe_allow_html=True,
    )

    artifacts = [
        ("Annotated Video", Path(str(summary["output_video"])), "video/mp4"),
        ("Tracking CSV", Path(str(summary["tracking_csv"])), "text/csv"),
        ("Trajectory PNG", Path(str(summary["trajectory_image"])), "image/png"),
        ("Summary JSON", Path(str(summary["summary_json"])), "application/json"),
    ]
    columns = st.columns(len(artifacts))

    for column, (label, path, mime) in zip(columns, artifacts):
        with column:
            if path.exists():
                st.download_button(
                    f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    width="stretch",
                )
            else:
                st.button(f"{label} Missing", disabled=True, width="stretch")


def render_media_tab(summary: dict[str, object]) -> None:
    """Render video and image outputs."""

    video_path = Path(str(summary["output_video"]))
    image_path = Path(str(summary["trajectory_image"]))

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown("### Annotated Video")
        st.markdown(
            '<p class="section-note">Rendered output with persistent IDs and trajectories overlaid.</p>',
            unsafe_allow_html=True,
        )
        if video_path.exists():
            st.video(str(video_path))
            st.caption(
                "If the player stays at 0:00, the file was likely encoded with a codec your browser cannot preview inline. "
                "Use the download button below or rerun after the app writes an H.264-compatible MP4."
            )
        else:
            st.warning("The output video was not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown("### Trajectory Summary")
        st.markdown(
            '<p class="section-note">Snapshot of accumulated track paths from the run.</p>',
            unsafe_allow_html=True,
        )
        if image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.warning("The trajectory summary image was not found.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_data_tab(summary: dict[str, object], dataframe: pd.DataFrame | None) -> None:
    """Render CSV preview and metadata."""

    st.markdown("### Tracking Table")
    st.markdown(
        '<p class="section-note">Interactive preview of the exported CSV with the first 500 rows.</p>',
        unsafe_allow_html=True,
    )
    if dataframe is not None and not dataframe.empty:
        st.dataframe(dataframe.head(500), width="stretch", height=360)
    else:
        st.info("Tracking CSV data is not available for preview.")

    st.markdown("### Run Metadata")
    metadata_df = pd.DataFrame(
        [{"field": key, "value": str(value)} for key, value in summary.items()]
    )
    st.dataframe(metadata_df, width="stretch", height=260)


def render_json_tab(summary: dict[str, object]) -> None:
    """Render JSON output in multiple forms."""

    json_path = Path(str(summary["summary_json"]))
    payload = summary
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    formatted_json = json.dumps(payload, indent=2)

    st.markdown("### JSON Output")
    st.markdown(
        '<p class="section-note">Raw submission-friendly JSON produced by the pipeline.</p>',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.95, 1.05], gap="large")
    with top_left:
        st.json(payload, expanded=True)
    with top_right:
        st.code(formatted_json, language="json")

    st.download_button(
        "Download Raw JSON",
        data=formatted_json.encode("utf-8"),
        file_name=json_path.name if json_path.exists() else "run_summary.json",
        mime="application/json",
        width="stretch",
    )


def get_recent_run_options() -> dict[str, Path]:
    """Return recent saved UI runs keyed by display label."""

    summaries = sorted(RUNS_DIR.glob("*/run_summary.json"), reverse=True) if RUNS_DIR.exists() else []
    options: dict[str, Path] = {}
    for summary_path in summaries[:12]:
        run_name = summary_path.parent.name
        label = f"{run_name}  -  {summary_path.parent}"
        options[label] = summary_path
    return options


def load_summary(summary_path: Path) -> dict[str, object]:
    """Load a saved summary JSON."""

    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_results(summary: dict[str, object]) -> None:
    """Display outputs produced by the tracking run."""

    st.subheader("Run Results")
    render_overview_cards(summary)
    render_artifact_downloads(summary)

    csv_path = Path(str(summary["tracking_csv"]))
    dataframe = load_tracking_dataframe(csv_path)

    media_tab, analytics_tab, data_tab, json_tab = st.tabs(
        ["Media", "Analytics", "Data Table", "JSON Output"]
    )

    with media_tab:
        render_media_tab(summary)

    with analytics_tab:
        render_analytics(dataframe)

    with data_tab:
        render_data_tab(summary, dataframe)

    with json_tab:
        render_json_tab(summary)


def main() -> None:
    """Render the Streamlit application."""

    ensure_logging()
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    render_header()

    with st.sidebar:
        st.markdown("## Run Browser")
        st.caption("Open a previous UI run without rerunning the pipeline.")
        recent_runs = get_recent_run_options()
        selected_run_label = st.selectbox(
            "Recent runs",
            options=["None"] + list(recent_runs.keys()),
            index=0,
        )
        st.markdown("## What stays unchanged")
        st.caption("This UI still calls the same `run_pipeline(...)` function and uses the same outputs.")

    left_panel, right_panel = st.columns([1.05, 0.95], gap="large")

    with left_panel:
        st.markdown("## Configure Run")
        st.markdown(
            '<p class="section-note">Choose a source and pass the same settings your CLI already supports.</p>',
            unsafe_allow_html=True,
        )

        source_mode = st.radio(
            "Input source",
            ["Use default public sample", "Use local path or URL", "Use uploaded file"],
            horizontal=True,
        )

        uploaded_file = None
        source_text = ""

        if source_mode == "Use uploaded file":
            uploaded_file = st.file_uploader(
                "Upload a video",
                type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
            )
            st.caption("Good for testing local clips without leaving the app.")
        elif source_mode == "Use local path or URL":
            source_text = st.text_input(
                "Local path or public video URL",
                placeholder="/absolute/path/to/video.mp4 or https://example.com/video.mp4",
            )
            st.caption("Use a local file path or paste a public downloadable URL.")
        else:
            source_text = st.text_input(
                "Public sample URL",
                value=DEFAULT_VIDEO_SOURCE_URL,
                placeholder="https://pixabay.com/videos/...",
            )
            st.caption("This processes the Pixababy public source URL directly.")

        st.caption(
            "Source links are resolved from the backend. The default processing source is the Pixababy URL."
        )

    with right_panel:
        st.markdown("## Tracking Controls")
        st.markdown(
            '<p class="section-note">These only affect arguments passed into the existing pipeline.</p>',
            unsafe_allow_html=True,
        )

        with st.form("tracking_form"):
            top_left, top_right = st.columns(2, gap="large")

            with top_left:
                model_weights = st.text_input("Model weights", value=DEFAULT_MODEL_WEIGHTS)
                tracker_config = st.text_input("Tracker config", value=DEFAULT_TRACKER_CONFIG)
                classes = st.text_input("Classes", value="0", help="Comma-separated COCO class ids.")
                device = st.text_input("Device", value="", placeholder="cpu, cuda:0, mps")

            with top_right:
                confidence_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
                iou_threshold = st.slider("IoU threshold", 0.05, 0.95, 0.45, 0.05)
                image_size = st.select_slider(
                    "Image size",
                    options=[320, 480, 640, 960, 1280],
                    value=DEFAULT_IMAGE_SIZE,
                )
                max_frames = st.number_input(
                    "Max frames",
                    min_value=0,
                    value=0,
                    step=25,
                    help="0 means the full video.",
                )

            start_run = st.form_submit_button("Run Tracking", width="stretch")

    summary_to_render: dict[str, object] | None = None

    if start_run:
        try:
            args = build_args(
                source_mode=source_mode,
                uploaded_file=uploaded_file,
                source_text=source_text,
                model_weights=model_weights,
                tracker_config=tracker_config,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                image_size=int(image_size),
                classes=classes,
                device=device,
                max_frames=int(max_frames),
            )
        except Exception as exc:
            st.error(str(exc))
            return

        with st.spinner("Running detection and persistent tracking. Larger videos can take a while."):
            try:
                summary_to_render = run_pipeline(args)
            except Exception as exc:
                st.error(f"Run failed: {exc}")
                return

        st.success("Tracking run completed.")

    elif selected_run_label != "None":
        summary_to_render = load_summary(recent_runs[selected_run_label])
        st.info("Showing a saved UI run from the sidebar selection.")

    elif recent_runs:
        first_summary_path = next(iter(recent_runs.values()))
        summary_to_render = load_summary(first_summary_path)
        st.info("Showing the most recent saved run.")

    if summary_to_render is not None:
        render_results(summary_to_render)


if __name__ == "__main__":
    main()
