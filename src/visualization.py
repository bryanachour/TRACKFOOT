from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from .pitch import SoccerPitchConfiguration, draw_pitch, pitch_to_image


PALETTE = sv.ColorPalette.from_hex([
    "#FFA500", "#1E90FF", "#FF1493", "#7CFC00",
    "#FFD700", "#9370DB", "#00CED1", "#FF6347",
])

BALL_COLOR = sv.Color.from_hex("#FFD700")
REFEREE_COLOR = sv.Color.from_hex("#FF00FF")


def build_annotators() -> Dict[str, object]:
    box = sv.BoxAnnotator(color=PALETTE, color_lookup=sv.ColorLookup.TRACK, thickness=2)
    ellipse = sv.EllipseAnnotator(color=PALETTE, color_lookup=sv.ColorLookup.TRACK, thickness=2)
    label = sv.LabelAnnotator(
        color=PALETTE,
        color_lookup=sv.ColorLookup.TRACK,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.BOTTOM_CENTER,
        text_scale=0.5,
        text_padding=4,
    )
    trace = sv.TraceAnnotator(color=PALETTE, color_lookup=sv.ColorLookup.TRACK, thickness=2, trace_length=30)
    triangle = sv.TriangleAnnotator(color=BALL_COLOR, base=18, height=14, outline_thickness=1)
    return {
        "box": box,
        "ellipse": ellipse,
        "label": label,
        "trace": trace,
        "triangle": triangle,
    }


def annotate_frame(
    frame: np.ndarray,
    players: sv.Detections,
    referees: sv.Detections,
    ball: sv.Detections,
    annotators: Dict[str, object],
) -> np.ndarray:
    out = frame.copy()
    if len(players) > 0:
        out = annotators["ellipse"].annotate(out, players)
        labels = [f"#{tid}" for tid in players.tracker_id] if players.tracker_id is not None else None
        if labels:
            out = annotators["label"].annotate(out, players, labels=labels)
        out = annotators["trace"].annotate(out, players)
    if len(referees) > 0:
        ref_anno = sv.EllipseAnnotator(color=REFEREE_COLOR, thickness=2)
        out = ref_anno.annotate(out, referees)
    if len(ball) > 0:
        out = annotators["triangle"].annotate(out, ball)
    return out


def render_tactical_view(
    cfg: SoccerPitchConfiguration,
    players_xy_cm: np.ndarray,
    player_ids: Optional[np.ndarray],
    ball_xy_cm: np.ndarray,
    out_w: int = 1050,
    out_h: int = 680,
    margin: int = 40,
) -> np.ndarray:
    pitch, scale, offset = draw_pitch(cfg, out_w=out_w, out_h=out_h, margin=margin)

    if len(players_xy_cm) > 0:
        for i, (x, y) in enumerate(players_xy_cm):
            if not (0 <= x <= cfg.length and 0 <= y <= cfg.width):
                continue
            px, py = pitch_to_image((x, y), scale, offset)
            tid = int(player_ids[i]) if player_ids is not None else 0
            color = PALETTE.by_idx(tid)
            cv2.circle(pitch, (px, py), 7, color.as_bgr(), -1, cv2.LINE_AA)
            cv2.circle(pitch, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)
            if player_ids is not None:
                cv2.putText(pitch, str(tid), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if len(ball_xy_cm) > 0:
        for x, y in ball_xy_cm:
            if not (0 <= x <= cfg.length and 0 <= y <= cfg.width):
                continue
            px, py = pitch_to_image((x, y), scale, offset)
            cv2.circle(pitch, (px, py), 5, BALL_COLOR.as_bgr(), -1, cv2.LINE_AA)
            cv2.circle(pitch, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

    return pitch


def stack_views(camera: np.ndarray, tactical: np.ndarray) -> np.ndarray:
    h_cam, w_cam = camera.shape[:2]
    h_tac, w_tac = tactical.shape[:2]
    scale = w_cam / w_tac
    new_h = int(round(h_tac * scale))
    tac_resized = cv2.resize(tactical, (w_cam, new_h), interpolation=cv2.INTER_LINEAR)
    return np.vstack([camera, tac_resized])
