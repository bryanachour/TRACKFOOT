"""Smart temporal sampler for football videos.

Goal
----
Run BEFORE the heavy TRACKFOOT pipeline to skip dead time on a Veo recording
(halftime, before kickoff, players off pitch, empty panorama). Returns a list
of ``(start_sec, end_sec)`` segments that deserve full processing.

Strategy (three cheap signals, fused)
-------------------------------------
1. **I-frame keyframe probe** (free).
   Decode only video keyframes (typically 1 every 2-5 s for Veo / x264 default
   ``-g 250``). On a 90 min match this is ~1800-2700 frames instead of 162 000.

2. **Dense optical flow magnitude** on a downscaled (320 px wide) gray version
   of each probe frame. ``cv2.calcOpticalFlowFarneback`` at 320x180 costs
   ~3-6 ms / frame on a modern CPU. Static panorama (empty pitch, halftime
   logo) gives mean magnitude < 0.5 px; live play gives 2-8 px.

3. **Light player count** with a small YOLO at ``imgsz=320``. We only invoke
   it on probe frames that already passed the motion gate, and only every
   N-th probe. A real match frame yields >= 8 detected persons; warmup,
   half-time and empty-pitch frames yield 0-3.

Each probe frame gets a binary label (active / inactive). Adjacent active
labels are merged into segments, padded by ``pad_sec`` on each side, and
gaps shorter than ``min_gap_sec`` are bridged. Segments shorter than
``min_segment_sec`` are dropped.

The sampler is **drop-in**: feed the resulting segments to
``pipeline.run`` via a thin wrapper (see :func:`iter_segment_frames`).

Expected gain on a 90 min Veo recording with 30-40% dead time:
- Pre-scan cost: ~30-60 s on CPU (no GPU needed for the gate).
- Saved pipeline time: 25-35% of total runtime.

Tuning knobs live in :class:`SamplerConfig` -- defaults are tuned for Veo
panoramic 1920x1080 @ 30 fps recordings of amateur matches.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    """All thresholds in one place. Defaults tuned for Veo amateur football."""

    # Probe strategy
    use_iframes: bool = True              # decode only keyframes if possible
    fallback_probe_sec: float = 2.0       # else sample one frame every N s
    probe_resize_width: int = 320         # downscale for motion + YOLO

    # Motion gate (optical flow)
    motion_min_magnitude: float = 0.8     # mean |flow| in pixels at 320 px wide
    motion_window: int = 3                # smoothing window over probe frames

    # Player-count gate (optional, requires a YOLO weight path)
    use_player_gate: bool = True
    player_gate_every: int = 1            # run YOLO on every Nth motion-positive probe
    player_gate_min_count: int = 8        # >= N persons => match in progress
    player_gate_conf: float = 0.25
    player_gate_imgsz: int = 320

    # Segment assembly
    pad_sec: float = 4.0                  # widen each active segment by +/- pad
    min_gap_sec: float = 8.0              # merge segments separated by < gap
    min_segment_sec: float = 6.0          # drop sub-second blips

    # Safety net
    max_skipped_fraction: float = 0.55    # never skip more than this much of the match


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    reason: str = "active"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class SampleReport:
    total_duration_sec: float
    kept_duration_sec: float
    n_probe_frames: int
    n_segments: int
    segments: List[Segment] = field(default_factory=list)

    @property
    def skipped_fraction(self) -> float:
        if self.total_duration_sec <= 0:
            return 0.0
        return 1.0 - (self.kept_duration_sec / self.total_duration_sec)

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "total_duration_sec": self.total_duration_sec,
                    "kept_duration_sec": self.kept_duration_sec,
                    "skipped_fraction": self.skipped_fraction,
                    "n_probe_frames": self.n_probe_frames,
                    "segments": [asdict(s) for s in self.segments],
                },
                indent=2,
            )
        )


# ---------------------------------------------------------------------------
# I-frame probing via ffprobe (fallback: cv2 stride)
# ---------------------------------------------------------------------------


def _probe_iframe_timestamps(video_path: Path) -> List[float]:
    """Return PTS (seconds) of every I-frame using ffprobe.

    Returns an empty list if ffprobe is unavailable or the stream is not
    H.264/H.265 -- caller must fall back to fixed-stride probing.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-loglevel", "error",
                "-select_streams", "v:0",
                "-skip_frame", "nokey",
                "-show_frames",
                "-show_entries", "frame=pts_time",
                "-of", "csv=p=0",
                str(video_path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffprobe iframe probe failed (%s); falling back to fixed stride", exc)
        return []

    timestamps: List[float] = []
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            timestamps.append(float(line))
        except ValueError:
            continue
    return timestamps


def _fixed_stride_timestamps(duration_sec: float, step_sec: float) -> List[float]:
    if duration_sec <= 0:
        return []
    n = int(duration_sec / step_sec)
    return [i * step_sec for i in range(n + 1)]


# ---------------------------------------------------------------------------
# Cheap player-count gate (optional)
# ---------------------------------------------------------------------------


class _LiteYoloCounter:
    """Wraps a YOLO model to return only a person count.

    Resolves the COCO 'person' class id when present, otherwise falls back
    to class 0 (works with both ultralytics COCO weights and the TRACKFOOT
    player-only weights, where the player class id is 2 but a small COCO
    model is preferred).
    """

    def __init__(self, weights_path: Path, conf: float, imgsz: int, device: Optional[str]):
        from ultralytics import YOLO  # local import keeps module light

        self.model = YOLO(str(weights_path))
        if device and device != "cpu":
            try:
                self.model.to(f"cuda:{device}" if device.isdigit() else device)
            except Exception:
                pass
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        names = getattr(self.model, "names", {}) or {}
        person_ids = [int(i) for i, n in names.items() if str(n).lower() in {"person", "player"}]
        self.person_ids = set(person_ids) if person_ids else {0}

    def __call__(self, frame: np.ndarray) -> int:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or result.boxes.cls is None:
            return 0
        cls = result.boxes.cls.cpu().numpy().astype(int)
        return int(np.isin(cls, list(self.person_ids)).sum())


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------


def _video_metadata(video_path: Path) -> Tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0
    return duration, fps, width, height


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float, fps: float) -> Optional[np.ndarray]:
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def _downscale_gray(frame: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_w:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = target_w / w
    small = cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _smoothed(values: Sequence[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return np.asarray(values, dtype=np.float32)
    k = np.ones(window, dtype=np.float32) / float(window)
    arr = np.asarray(values, dtype=np.float32)
    return np.convolve(arr, k, mode="same")


def _labels_to_segments(
    timestamps: Sequence[float],
    active_mask: Sequence[bool],
    duration: float,
    cfg: SamplerConfig,
) -> List[Segment]:
    if not timestamps:
        return [Segment(0.0, duration, reason="fallback-no-probe")]

    segments: List[Segment] = []
    cur_start: Optional[float] = None
    last_active_ts: Optional[float] = None

    for ts, active in zip(timestamps, active_mask):
        if active and cur_start is None:
            cur_start = ts
        if active:
            last_active_ts = ts
        if not active and cur_start is not None:
            segments.append(Segment(cur_start, last_active_ts or ts))
            cur_start = None
    if cur_start is not None:
        segments.append(Segment(cur_start, last_active_ts or duration))

    # Pad
    padded: List[Segment] = [
        Segment(max(0.0, s.start_sec - cfg.pad_sec), min(duration, s.end_sec + cfg.pad_sec))
        for s in segments
    ]

    # Merge gaps shorter than min_gap_sec
    merged: List[Segment] = []
    for seg in padded:
        if merged and seg.start_sec - merged[-1].end_sec <= cfg.min_gap_sec:
            merged[-1] = Segment(merged[-1].start_sec, max(merged[-1].end_sec, seg.end_sec))
        else:
            merged.append(seg)

    # Drop blips
    merged = [s for s in merged if s.duration >= cfg.min_segment_sec]
    return merged


def _safety_net(segments: List[Segment], duration: float, cfg: SamplerConfig) -> List[Segment]:
    """Never skip more than max_skipped_fraction of the match."""
    kept = sum(s.duration for s in segments)
    if duration <= 0:
        return segments
    skipped = 1.0 - (kept / duration)
    if skipped <= cfg.max_skipped_fraction:
        return segments
    logger.warning(
        "smart sampler would skip %.0f%% of video (> safety %.0f%%); returning full clip",
        skipped * 100,
        cfg.max_skipped_fraction * 100,
    )
    return [Segment(0.0, duration, reason="safety-fallback")]


def sample_segments(
    video_path: Path,
    cfg: Optional[SamplerConfig] = None,
    player_weights: Optional[Path] = None,
    device: Optional[str] = None,
) -> SampleReport:
    """Scan ``video_path`` and return the active segments to process.

    Parameters
    ----------
    video_path
        Path to the source video (mp4 / mkv).
    cfg
        Tuning knobs. Pass ``SamplerConfig()`` for defaults.
    player_weights
        Optional YOLO weights for the player-count gate. If ``None`` the
        gate is skipped and only the motion gate is used.
    device
        Torch device for the YOLO gate ("cpu", "0", "cuda:0", ...).
    """
    video_path = Path(video_path)
    cfg = cfg or SamplerConfig()
    duration, fps, _w, _h = _video_metadata(video_path)

    # 1) probe timestamps
    timestamps: List[float] = []
    if cfg.use_iframes:
        timestamps = _probe_iframe_timestamps(video_path)
    if not timestamps:
        timestamps = _fixed_stride_timestamps(duration, cfg.fallback_probe_sec)
    if not timestamps:
        return SampleReport(duration, duration, 0, 1, [Segment(0.0, duration, "no-probe")])

    # 2) optional player gate
    counter: Optional[_LiteYoloCounter] = None
    if cfg.use_player_gate and player_weights and Path(player_weights).exists():
        try:
            counter = _LiteYoloCounter(
                Path(player_weights),
                conf=cfg.player_gate_conf,
                imgsz=cfg.player_gate_imgsz,
                device=device,
            )
        except Exception as exc:
            logger.warning("player gate disabled (%s)", exc)
            counter = None

    # 3) sweep probes -> motion magnitude + optional player count
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    prev_gray: Optional[np.ndarray] = None
    magnitudes: List[float] = []
    motion_active: List[bool] = []
    player_active: List[Optional[bool]] = []
    valid_timestamps: List[float] = []

    flow_params = dict(pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

    try:
        for i, ts in enumerate(timestamps):
            frame = _read_frame_at(cap, ts, fps)
            if frame is None:
                continue
            gray = _downscale_gray(frame, cfg.probe_resize_width)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
                mag = float(np.linalg.norm(flow, axis=2).mean())
            else:
                mag = 0.0
            prev_gray = gray
            magnitudes.append(mag)
            valid_timestamps.append(ts)

            motion_ok = mag >= cfg.motion_min_magnitude
            motion_active.append(motion_ok)

            if counter is not None and motion_ok and (i % cfg.player_gate_every == 0):
                try:
                    n_persons = counter(frame)
                except Exception as exc:
                    logger.debug("YOLO counter error at %.1fs: %s", ts, exc)
                    n_persons = -1
                player_active.append(n_persons >= cfg.player_gate_min_count if n_persons >= 0 else None)
            else:
                player_active.append(None)
    finally:
        cap.release()

    # smooth motion to absorb single-frame noise
    smoothed = _smoothed(magnitudes, cfg.motion_window)
    smoothed_active = smoothed >= cfg.motion_min_magnitude

    # fuse: a probe is active if motion AND (no player vote OR player vote True)
    fused: List[bool] = []
    for m_ok, p_vote in zip(smoothed_active, player_active):
        if not m_ok:
            fused.append(False)
        elif p_vote is False:
            fused.append(False)
        else:
            fused.append(True)

    segments = _labels_to_segments(valid_timestamps, fused, duration, cfg)
    segments = _safety_net(segments, duration, cfg)

    kept = sum(s.duration for s in segments)
    return SampleReport(
        total_duration_sec=duration,
        kept_duration_sec=kept,
        n_probe_frames=len(valid_timestamps),
        n_segments=len(segments),
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Drop-in frame iterator for the existing pipeline
# ---------------------------------------------------------------------------


def iter_segment_frames(
    video_path: Path,
    segments: Sequence[Segment],
    stride: int = 1,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(absolute_frame_idx, frame)`` for every requested segment.

    Designed to replace ``supervision.get_video_frames_generator`` in
    :func:`src.pipeline.run`. The absolute frame index is preserved so the
    tracker, stats aggregator and trajectory store keep working with
    real timestamps.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        for seg in segments:
            start_idx = int(round(seg.start_sec * fps))
            end_idx = int(round(seg.end_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            idx = start_idx
            while idx < end_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                if (idx - start_idx) % max(stride, 1) == 0:
                    yield idx, frame
                idx += 1
    finally:
        cap.release()
