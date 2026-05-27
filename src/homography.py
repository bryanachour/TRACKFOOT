from typing import Optional

import cv2
import numpy as np
import supervision as sv

from .pitch import SoccerPitchConfiguration


class ViewTransformer:
    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray):
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("need >= 4 points to estimate homography")
        src = src_points.astype(np.float32)
        dst = dst_points.astype(np.float32)
        self.m, self.mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.m is None:
            raise ValueError("homography estimation failed")

    @property
    def inliers(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts.reshape(-1, 2)
        pts = pts.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.m)
        return out.reshape(-1, 2)


class PitchTransformerSmoother:
    def __init__(self, cfg: SoccerPitchConfiguration, min_inliers: int = 6, alpha: float = 0.85):
        self.cfg = cfg
        self.min_inliers = min_inliers
        self.alpha = alpha
        self.m_ema: Optional[np.ndarray] = None

    def update(self, keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
        if keypoints is None or len(keypoints.xy) == 0:
            return self._current()

        xy = keypoints.xy[0]
        pitch_vertices = np.array(self.cfg.vertices, dtype=np.float32)
        n = min(len(xy), len(pitch_vertices))
        if n < 4:
            return self._current()
        xy = xy[:n]
        pitch_vertices = pitch_vertices[:n]

        mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)
        if keypoints.confidence is not None:
            conf = keypoints.confidence[0][:n]
            mask &= conf > 0.5
        if mask.sum() < 4:
            return self._current()

        src = xy[mask]
        dst = pitch_vertices[mask]
        try:
            vt = ViewTransformer(src, dst)
        except ValueError:
            return self._current()
        if vt.inliers < self.min_inliers:
            return self._current()

        if self.m_ema is None:
            self.m_ema = vt.m
        else:
            self.m_ema = self.alpha * self.m_ema + (1 - self.alpha) * vt.m

        smoothed = ViewTransformer.__new__(ViewTransformer)
        smoothed.m = self.m_ema
        smoothed.mask = vt.mask
        return smoothed

    def _current(self) -> Optional[ViewTransformer]:
        if self.m_ema is None:
            return None
        vt = ViewTransformer.__new__(ViewTransformer)
        vt.m = self.m_ema
        vt.mask = None
        return vt


def bottom_centre_points(detections: sv.Detections) -> np.ndarray:
    if len(detections) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xyxy = detections.xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = xyxy[:, 3]
    return np.stack([cx, cy], axis=1)
