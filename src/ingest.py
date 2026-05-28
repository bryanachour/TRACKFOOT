import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

import yt_dlp


URL_RE = re.compile(r"^https?://", re.IGNORECASE)
VEO_URL_RE = re.compile(r"^https?://(?:app\.)?veo\.co/", re.IGNORECASE)


def is_url(text: str) -> bool:
    return bool(URL_RE.match(text.strip()))


def is_veo_url(text: str) -> bool:
    return bool(VEO_URL_RE.match(text.strip()))


def download_url(url: str, dest_dir: Path, progress_hook: Optional[callable] = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / f"{uuid.uuid4().hex}.%(ext)s")
    opts = {
        "outtmpl": out_template,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    if progress_hook:
        opts["progress_hooks"] = [progress_hook]
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url.strip(), download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        for cand in dest_dir.glob(f"{path.stem}.*"):
            return cand
        raise FileNotFoundError(f"download failed for {url}")
    return path


def resolve_input(file_path: Optional[str], url: Optional[str], dest_dir: Path) -> Path:
    if file_path:
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"upload not found: {src}")
        dst = dest_dir / src.name
        if src.resolve() != dst.resolve():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return dst
    if url and is_url(url):
        return download_url(url.strip(), dest_dir)
    raise ValueError("provide either an uploaded file or a URL")


def hint_for_url(url: str) -> Optional[str]:
    if not url or not is_url(url):
        return None
    if is_veo_url(url):
        return "✅ URL Veo détectée — téléchargement direct via yt-dlp (pas de login requis pour les matchs publics)"
    return None
