import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .pitch import SoccerPitchConfiguration, draw_pitch, pitch_to_image


class TrajectoryStore:
    def __init__(self):
        self.tracks: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self.ball: List[Tuple[int, float, float]] = []

    def add_players(self, frame_idx: int, ids: np.ndarray, xy_cm: np.ndarray) -> None:
        for tid, (x, y) in zip(ids, xy_cm):
            self.tracks[int(tid)].append((int(frame_idx), float(x), float(y)))

    def add_ball(self, frame_idx: int, xy_cm: np.ndarray) -> None:
        for x, y in xy_cm:
            self.ball.append((int(frame_idx), float(x), float(y)))

    def to_dict(self) -> dict:
        return {
            "players": {str(tid): [{"frame": f, "x": x, "y": y} for f, x, y in pts]
                        for tid, pts in self.tracks.items()},
            "ball": [{"frame": f, "x": x, "y": y} for f, x, y in self.ball],
        }

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


def render_heatmap(
    cfg: SoccerPitchConfiguration,
    points_cm: List[Tuple[float, float]],
    out_w: int = 1050,
    out_h: int = 680,
    margin: int = 40,
    sigma: float = 12.0,
    cmap: str = "hot",
) -> np.ndarray:
    pitch, scale, offset = draw_pitch(cfg, out_w=out_w, out_h=out_h, margin=margin)
    h, w = pitch.shape[:2]
    grid = np.zeros((h, w), dtype=np.float32)
    for x, y in points_cm:
        if not (0 <= x <= cfg.length and 0 <= y <= cfg.width):
            continue
        px, py = pitch_to_image((x, y), scale, offset)
        if 0 <= px < w and 0 <= py < h:
            grid[py, px] += 1.0
    if grid.sum() == 0:
        return pitch
    grid = gaussian_filter(grid, sigma=sigma)
    grid /= grid.max()
    cmap_obj = plt.get_cmap(cmap)
    coloured = (cmap_obj(grid)[..., :3] * 255).astype(np.uint8)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
    alpha = (grid * 0.85).astype(np.float32)[..., None]
    blended = (pitch.astype(np.float32) * (1 - alpha) + coloured.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def export_heatmaps(
    store: TrajectoryStore,
    cfg: SoccerPitchConfiguration,
    out_dir: Path,
    min_points: int = 30,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, pts in store.tracks.items():
        if len(pts) < min_points:
            continue
        xy = [(x, y) for _, x, y in pts]
        img = render_heatmap(cfg, xy)
        cv2.imwrite(str(out_dir / f"player_{tid:03d}.png"), img)
    if store.ball:
        xy = [(x, y) for _, x, y in store.ball]
        img = render_heatmap(cfg, xy, cmap="cool")
        cv2.imwrite(str(out_dir / "ball.png"), img)
