from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class SoccerPitchConfiguration:
    """Mirrors roboflow/sports SoccerPitchConfiguration (commit main).

    Source: https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py
    The roboflow `football-field-detection-f07vi/v14` model outputs 32 keypoints
    in the EXACT positional order of `vertices` (index i ↔ vertices[i]).
    Coordinates are in centimetres. Origin (0,0) = top-left corner; X = length, Y = width.
    """
    width: int = 7000
    length: int = 12000
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return [
            (0, 0),
            (0, 1450.0),
            (0, 2584.0),
            (0, 4416.0),
            (0, 5550.0),
            (0, 7000),
            (550, 2584.0),
            (550, 4416.0),
            (1100, 3500.0),
            (2015, 1450.0),
            (2015, 2584.0),
            (2015, 4416.0),
            (2015, 5550.0),
            (6000, 0),
            (6000, 2585.0),
            (6000, 4415.0),
            (6000, 7000),
            (9985, 1450.0),
            (9985, 2584.0),
            (9985, 4416.0),
            (9985, 5550.0),
            (10900, 3500.0),
            (11450, 2584.0),
            (11450, 4416.0),
            (12000, 0),
            (12000, 1450.0),
            (12000, 2584.0),
            (12000, 4416.0),
            (12000, 5550.0),
            (12000, 7000),
            (5085.0, 3500.0),
            (6915.0, 3500.0),
        ]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [
            (1, 6), (6, 30), (30, 25), (25, 1),
            (14, 17),
            (2, 10), (10, 13), (13, 5),
            (3, 7), (7, 8), (8, 4),
            (26, 18), (18, 21), (21, 29),
            (27, 23), (23, 24), (24, 28),
        ]


def draw_pitch(
    cfg: SoccerPitchConfiguration,
    out_w: int = 1050,
    out_h: int = 680,
    margin: int = 40,
    pitch_color: Tuple[int, int, int] = (34, 139, 34),
    line_color: Tuple[int, int, int] = (245, 245, 245),
    line_thickness: int = 2,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    img = np.full((out_h, out_w, 3), pitch_color, dtype=np.uint8)

    scale_x = (out_w - 2 * margin) / cfg.length
    scale_y = (out_h - 2 * margin) / cfg.width
    scale = min(scale_x, scale_y)
    off_x = margin + ((out_w - 2 * margin) - cfg.length * scale) / 2
    off_y = margin + ((out_h - 2 * margin) - cfg.width * scale) / 2

    def project(p: Tuple[float, float]) -> Tuple[int, int]:
        x, y = p
        return int(round(off_x + x * scale)), int(round(off_y + y * scale))

    verts = cfg.vertices
    for a, b in cfg.edges:
        pa = project(verts[a - 1])
        pb = project(verts[b - 1])
        cv2.line(img, pa, pb, line_color, line_thickness, cv2.LINE_AA)

    centre = project((cfg.length / 2, cfg.width / 2))
    radius_px = int(round(cfg.centre_circle_radius * scale))
    cv2.circle(img, centre, radius_px, line_color, line_thickness, cv2.LINE_AA)
    cv2.circle(img, centre, 4, line_color, -1, cv2.LINE_AA)

    left_spot = project((cfg.penalty_spot_distance, cfg.width / 2))
    right_spot = project((cfg.length - cfg.penalty_spot_distance, cfg.width / 2))
    cv2.circle(img, left_spot, 4, line_color, -1, cv2.LINE_AA)
    cv2.circle(img, right_spot, 4, line_color, -1, cv2.LINE_AA)

    arc_r = int(round(cfg.centre_circle_radius * scale))
    cv2.ellipse(img, left_spot, (arc_r, arc_r), 0, -53, 53, line_color, line_thickness, cv2.LINE_AA)
    cv2.ellipse(img, right_spot, (arc_r, arc_r), 180, -53, 53, line_color, line_thickness, cv2.LINE_AA)

    return img, scale, (int(round(off_x)), int(round(off_y)))


def pitch_to_image(point_cm: Tuple[float, float], scale: float, offset: Tuple[int, int]) -> Tuple[int, int]:
    return int(round(offset[0] + point_cm[0] * scale)), int(round(offset[1] + point_cm[1] * scale))
