from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

import config as C
from .detection import YoloDetector, BallDetector, split_by_class
from .homography import PitchTransformerSmoother, bottom_centre_points
from .pitch import SoccerPitchConfiguration
from .stats import StatsAggregator
from .teams import TeamClassifier
from .tracking import build_tracker
from .trajectories import TrajectoryStore, export_heatmaps
from .visualization import (
    annotate_frame,
    build_annotators,
    render_tactical_view,
    stack_views,
)


ProgressCb = Callable[[int, int, str], None]


@dataclass
class PipelineOptions:
    source: Path
    output_dir: Path
    player_weights: Path = C.PLAYER_DETECTION_WEIGHTS
    pitch_weights: Path = C.PITCH_DETECTION_WEIGHTS
    ball_weights: Optional[Path] = C.BALL_DETECTION_WEIGHTS
    device: Optional[str] = None
    stride: int = 1
    save_annotated: bool = True
    save_tactical: bool = True
    save_stacked: bool = True
    enable_team_classification: bool = True
    enable_stats: bool = True


@dataclass
class PipelineResult:
    annotated: Optional[Path] = None
    tactical: Optional[Path] = None
    stacked: Optional[Path] = None
    trajectories_json: Optional[Path] = None
    stats_json: Optional[Path] = None
    heatmaps_dir: Optional[Path] = None
    heatmap_files: list = field(default_factory=list)
    n_players: int = 0
    n_frames: int = 0
    n_ball_points: int = 0
    fps: float = 0.0
    stats_summary: Optional[dict] = None


def _load_pitch_model(weights: Path, device: Optional[str], imgsz: int = 640, half: bool = True) -> YOLO:
    weights = Path(weights)
    engine = weights.with_suffix(".engine")
    use_gpu = device is not None and device != "cpu"

    if use_gpu and engine.exists():
        try:
            return YOLO(str(engine), task="pose")
        except Exception:
            pass
    if use_gpu:
        try:
            import tensorrt  # noqa: F401
            pt_model = YOLO(str(weights))
            pt_model.export(format="engine", half=half, imgsz=imgsz, device=device, dynamic=False, batch=1, verbose=False)
            if engine.exists():
                return YOLO(str(engine), task="pose")
        except Exception:
            pass

    model = YOLO(str(weights))
    if use_gpu:
        try:
            model.to(f"cuda:{device}" if str(device).isdigit() else device)
        except Exception:
            pass
    return model


def _pitch_keypoints(model: YOLO, frame: np.ndarray, conf: float, device: Optional[str], imgsz: int = 640, half: bool = True) -> sv.KeyPoints:
    result = model.predict(frame, conf=conf, device=device, imgsz=imgsz, half=half, verbose=False)[0]
    return sv.KeyPoints.from_ultralytics(result)


def run(opts: PipelineOptions, progress_cb: Optional[ProgressCb] = None) -> PipelineResult:
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    result = PipelineResult()

    if progress_cb:
        progress_cb(0, 1, "loading models")

    video_info = sv.VideoInfo.from_video_path(str(opts.source))
    cfg = SoccerPitchConfiguration()

    player_det = YoloDetector(opts.player_weights, conf=C.PLAYER_CONF, iou=C.NMS_IOU, device=opts.device, imgsz=C.INFERENCE_IMGSZ)
    pitch_model = _load_pitch_model(opts.pitch_weights, opts.device, imgsz=C.INFERENCE_IMGSZ)
    ball_det = BallDetector(opts.ball_weights, conf=C.BALL_CONF, device=opts.device, imgsz=C.INFERENCE_IMGSZ) if opts.ball_weights and opts.ball_weights.exists() else None

    tracker = build_tracker(
        frame_rate=int(video_info.fps),
        track_buffer=C.TRACKER_TRACK_BUFFER,
        match_thresh=C.TRACKER_MATCH_THRESH,
        high_thresh=C.TRACKER_HIGH_THRESH,
        low_thresh=C.TRACKER_LOW_THRESH,
    )
    smoother = PitchTransformerSmoother(cfg, min_inliers=C.HOMOGRAPHY_MIN_INLIERS, alpha=C.HOMOGRAPHY_SMOOTHING)
    annotators = build_annotators()
    store = TrajectoryStore()
    team_classifier: Optional[TeamClassifier] = TeamClassifier() if opts.enable_team_classification else None
    stats_agg: Optional[StatsAggregator] = StatsAggregator(
        fps=float(video_info.fps),
        pitch_length_cm=cfg.length,
        pitch_width_cm=cfg.width,
    ) if opts.enable_stats else None

    annotated_path = opts.output_dir / "annotated.mp4"
    tactical_path = opts.output_dir / "tactical.mp4"
    stacked_path = opts.output_dir / "stacked.mp4"

    sinks: Dict[str, sv.VideoSink] = {}
    if opts.save_annotated:
        sinks["annotated"] = sv.VideoSink(str(annotated_path), video_info)
    if opts.save_tactical:
        tac_info = sv.VideoInfo(width=C.PITCH_OUT_WIDTH_PX, height=C.PITCH_OUT_HEIGHT_PX, fps=video_info.fps)
        sinks["tactical"] = sv.VideoSink(str(tactical_path), tac_info)
    if opts.save_stacked:
        scale = video_info.width / C.PITCH_OUT_WIDTH_PX
        stacked_h = video_info.height + int(round(C.PITCH_OUT_HEIGHT_PX * scale))
        st_info = sv.VideoInfo(width=video_info.width, height=stacked_h, fps=video_info.fps)
        sinks["stacked"] = sv.VideoSink(str(stacked_path), st_info)

    frames = sv.get_video_frames_generator(str(opts.source), stride=opts.stride)
    total = (video_info.total_frames or 0) // max(opts.stride, 1)

    contexts = [s.__enter__() for s in sinks.values()]
    processed = 0
    try:
        for frame_idx, frame in enumerate(tqdm(frames, total=total, desc="processing")):
            dets = player_det(frame)

            players = split_by_class(dets, C.CLASS_PLAYER)
            keepers = split_by_class(dets, C.CLASS_GOALKEEPER)
            players = sv.Detections.merge([players, keepers]) if len(keepers) > 0 else players
            referees = split_by_class(dets, C.CLASS_REFEREE)

            players = tracker.update_with_detections(players)

            if ball_det is not None:
                ball = ball_det(frame)
            else:
                ball = split_by_class(dets, C.CLASS_BALL)

            kpts = _pitch_keypoints(pitch_model, frame, C.PITCH_CONF, opts.device, imgsz=C.INFERENCE_IMGSZ)
            transformer = smoother.update(kpts)

            players_xy_cm = np.zeros((0, 2), dtype=np.float32)
            ball_xy_cm = np.zeros((0, 2), dtype=np.float32)
            if transformer is not None:
                if len(players) > 0:
                    players_xy_cm = transformer.transform_points(bottom_centre_points(players))
                if len(ball) > 0:
                    ball_xy_cm = transformer.transform_points(bottom_centre_points(ball))

            team_ids: Optional[np.ndarray] = None
            if team_classifier is not None and len(players) > 0:
                team_classifier.observe(frame, players)
                if team_classifier.calibrated and players.tracker_id is not None:
                    team_ids = team_classifier.team_ids_for(players.tracker_id)

            possessor: Optional[int] = None
            if stats_agg is not None and len(players_xy_cm) > 0 and players.tracker_id is not None:
                stats_agg.update(frame_idx, players.tracker_id, players_xy_cm, ball_xy_cm, team_ids)
                possessor = stats_agg.current_possessor

            if len(players_xy_cm) > 0 and players.tracker_id is not None:
                store.add_players(frame_idx, players.tracker_id, players_xy_cm)
            if len(ball_xy_cm) > 0:
                store.add_ball(frame_idx, ball_xy_cm)

            anno = None
            if "annotated" in sinks or "stacked" in sinks:
                anno = annotate_frame(frame, players, referees, ball, annotators, team_ids=team_ids, possessor_id=possessor)
            if "annotated" in sinks:
                sinks["annotated"].write_frame(anno)

            tac = None
            if "tactical" in sinks or "stacked" in sinks:
                tac = render_tactical_view(
                    cfg,
                    players_xy_cm,
                    players.tracker_id if len(players_xy_cm) > 0 else None,
                    ball_xy_cm,
                    out_w=C.PITCH_OUT_WIDTH_PX,
                    out_h=C.PITCH_OUT_HEIGHT_PX,
                    margin=C.PITCH_MARGIN_PX,
                    team_ids=team_ids,
                    possessor_id=possessor,
                )

            if "tactical" in sinks:
                sinks["tactical"].write_frame(tac)

            if "stacked" in sinks:
                sinks["stacked"].write_frame(stack_views(anno, tac))

            processed += 1
            if progress_cb and total > 0:
                progress_cb(processed, total, f"frame {processed}/{total}")
    finally:
        for s in sinks.values():
            s.__exit__(None, None, None)

    if progress_cb:
        progress_cb(processed, max(processed, 1), "exporting trajectories + heatmaps")

    json_path = opts.output_dir / "trajectories.json"
    store.save_json(json_path)
    heatmap_dir = opts.output_dir / "heatmaps"
    export_heatmaps(store, cfg, heatmap_dir)

    stats_json = None
    summary = None
    if stats_agg is not None:
        stats_json = opts.output_dir / "stats.json"
        stats_agg.save_json(stats_json)
        summary = stats_agg.summary()

    result.annotated = annotated_path if opts.save_annotated else None
    result.tactical = tactical_path if opts.save_tactical else None
    result.stacked = stacked_path if opts.save_stacked else None
    result.trajectories_json = json_path
    result.stats_json = stats_json
    result.stats_summary = summary
    result.heatmaps_dir = heatmap_dir
    result.heatmap_files = sorted(heatmap_dir.glob("*.png")) if heatmap_dir.exists() else []
    result.n_players = len(store.tracks)
    result.n_frames = processed
    result.n_ball_points = len(store.ball)
    result.fps = float(video_info.fps)

    if progress_cb:
        progress_cb(processed, max(processed, 1), "done")

    return result
