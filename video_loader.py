"""Video loading and optional public-video download helpers."""

from __future__ import annotations

import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class VideoSource:
    """Resolved video input metadata."""

    source_path: Path
    source_url: str | None
    downloaded: bool


def resolve_video_source(
    video_path: str | None,
    default_source_url: str | None,
    download_dir: str,
    auto_download: bool,
) -> VideoSource:
    """Resolve a local video path or download a public sample if needed."""

    if video_path:
        if _is_url(video_path):
            downloaded_path = download_video(video_path, download_dir)
            return VideoSource(downloaded_path, video_path, True)

        candidate = Path(video_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input video was not found: {candidate}")
        return VideoSource(candidate, None, False)

    if not auto_download or not default_source_url:
        raise ValueError(
            "No local video was supplied. Pass --video-path or enable --download-default-video."
        )

    downloaded_path = download_video(default_source_url, download_dir)
    return VideoSource(downloaded_path, default_source_url, True)


def download_video(video_url: str, download_dir: str) -> Path:
    """Download a public video and return the local file path."""

    target_dir = Path(download_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if _looks_like_direct_video_file(video_url):
        destination = target_dir / Path(urlparse(video_url).path).name
        if destination.exists():
            return destination.resolve()
        urlretrieve(video_url, destination)
        if not destination.exists():
            raise FileNotFoundError(f"Direct video download failed: {destination}")
        return destination.resolve()

    output_template = str(target_dir / "public_sports_video.%(ext)s")
    existing_downloads = sorted(target_dir.glob("public_sports_video.*"))
    for existing_file in existing_downloads:
        if existing_file.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}:
            return existing_file.resolve()

    # Pixabay page URLs are often easier to resolve by extracting the direct CDN MP4
    # ourselves instead of relying on a generic downloader path that may be blocked.
    pixabay_fallback = _download_pixabay_video(video_url, target_dir)
    if pixabay_fallback is not None:
        return pixabay_fallback

    ydl_opts = {
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "outtmpl": output_template,
        "quiet": True,
        "http_headers": _BROWSER_HEADERS,
        "extractor_args": {"generic": {"impersonate": ["chrome"]}},
    }

    try:
        downloaded_file = _download_with_ytdlp(video_url, ydl_opts)
    except DownloadError as exc:
        raise RuntimeError(
            "Automatic download failed for the provided page URL. "
            "Some hosting sites block scripted downloads. "
            "Download the public clip in your browser and pass it with --video-path, "
            "or use a direct downloadable video URL with --source-url."
        ) from exc

    if downloaded_file.suffix.lower() != ".mp4":
        mp4_candidate = downloaded_file.with_suffix(".mp4")
        if mp4_candidate.exists():
            downloaded_file = mp4_candidate

    if not downloaded_file.exists():
        raise FileNotFoundError(f"Video download finished but file was not found: {downloaded_file}")

    return downloaded_file.resolve()


def _is_url(candidate: str) -> bool:
    """Return True when the input string looks like a URL."""

    parsed = urlparse(candidate)
    return bool(parsed.scheme and parsed.netloc)


def _looks_like_direct_video_file(candidate: str) -> bool:
    """Return True when the URL appears to point directly to a video file."""

    path = urlparse(candidate).path.lower()
    return path.endswith((".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"))


def _download_pixabay_video(video_url: str, target_dir: Path) -> Path | None:
    """Try to extract and download a direct MP4 from a Pixabay video page."""

    parsed = urlparse(video_url)
    if "pixabay.com" not in parsed.netloc:
        return None

    request = Request(video_url, headers=_BROWSER_HEADERS)
    with urlopen(request) as response:
        page_html = response.read().decode("utf-8", errors="ignore")

    direct_url = _extract_direct_video_url(page_html)
    if direct_url is None:
        return None

    destination = target_dir / "public_sports_video.mp4"
    if destination.exists():
        return destination.resolve()

    _download_url_to_path(
        direct_url,
        destination,
        headers={
            **_BROWSER_HEADERS,
            "Accept": "video/webm,video/mp4,application/octet-stream,*/*;q=0.8",
            "Referer": video_url,
        },
    )
    return destination.resolve() if destination.exists() else None


def _extract_direct_video_url(page_html: str) -> str | None:
    """Extract a direct MP4 URL from provider HTML when available."""

    patterns = [
        r'<source[^>]+src="(https://cdn\.pixabay\.com/video/[^"]+?\.mp4)"',
        r'<video[^>]+src="(https://cdn\.pixabay\.com/video/[^"]+?\.mp4)"',
        r'https:\\/\\/cdn\.pixabay\.com\\/video\\/[^"]+?\.mp4',
        r'https://cdn\.pixabay\.com/video/[^"\']+?\.mp4',
        r'"contentUrl"\s*:\s*"([^"]+?\.mp4)"',
        r'"url"\s*:\s*"(https://cdn\.pixabay\.com/video/[^"]+?\.mp4)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, page_html)
        if not match:
            continue
        candidate = match.group(1) if match.groups() else match.group(0)
        return html.unescape(candidate.replace("\\/", "/"))

    return None


def _download_url_to_path(url: str, destination: Path, headers: dict[str, str] | None = None) -> None:
    """Download a URL to disk while allowing browser-like headers."""

    request = Request(url, headers=headers or {})
    with urlopen(request) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def _download_with_ytdlp(video_url: str, base_opts: dict) -> Path:
    """Try yt-dlp with increasingly capable options before giving up."""

    option_attempts = [
        base_opts,
        {**base_opts, "cookiesfrombrowser": ("chrome", None, None, None)},
        {**base_opts, "cookiesfrombrowser": ("safari", None, None, None)},
    ]

    last_error: Exception | None = None
    for attempt_opts in option_attempts:
        try:
            with YoutubeDL(attempt_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                return Path(ydl.prepare_filename(info))
        except Exception as exc:
            last_error = exc

    if isinstance(last_error, DownloadError):
        raise last_error
    raise DownloadError(str(last_error) if last_error else "yt-dlp download failed")
