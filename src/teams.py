from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans


@dataclass
class TeamClassifier:
    n_calibration_frames: int = 30
    samples_per_frame_max: int = 60
    crop_top_ratio: float = 0.15
    crop_bot_ratio: float = 0.55
    crop_side_ratio: float = 0.15
    sat_min: int = 40
    val_min: int = 40
    val_max: int = 230

    _calibrated: bool = False
    _centers_hsv: Optional[np.ndarray] = None
    _samples: List[np.ndarray] = field(default_factory=list)
    _samples_count: int = 0
    _player_team: Dict[int, int] = field(default_factory=dict)
    _player_history: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def _jersey_crop(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = xyxy.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 - x1 < 6 or y2 - y1 < 12:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        h = y2 - y1; w = x2 - x1
        yt = y1 + int(h * self.crop_top_ratio)
        yb = y1 + int(h * self.crop_bot_ratio)
        xl = x1 + int(w * self.crop_side_ratio)
        xr = x2 - int(w * self.crop_side_ratio)
        return frame[yt:yb, xl:xr]

    def _dominant_hsv(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr.size == 0:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 1] >= self.sat_min) & (hsv[..., 2] >= self.val_min) & (hsv[..., 2] <= self.val_max)
        if mask.sum() < 20:
            return None
        pixels = hsv[mask].astype(np.float32)
        return pixels.mean(axis=0)

    def observe(self, frame: np.ndarray, players: sv.Detections) -> None:
        if self._calibrated or len(players) == 0:
            if self._calibrated:
                self._assign(frame, players)
            return
        count_in_frame = 0
        for i in range(len(players)):
            if count_in_frame >= self.samples_per_frame_max:
                break
            crop = self._jersey_crop(frame, players.xyxy[i])
            sig = self._dominant_hsv(crop)
            if sig is not None:
                self._samples.append(sig)
                count_in_frame += 1
        self._samples_count += 1
        if self._samples_count >= self.n_calibration_frames and len(self._samples) >= 12:
            self._fit()

    def _fit(self) -> None:
        X = np.stack(self._samples, axis=0)
        hue = X[:, 0:1] * (2 * np.pi / 180.0)
        feat = np.concatenate([np.cos(hue) * 0.7, np.sin(hue) * 0.7, X[:, 1:2] / 255.0, X[:, 2:3] / 255.0], axis=1)
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feat)
        centers = []
        for k in range(2):
            members = X[km.labels_ == k]
            if len(members):
                centers.append(members.mean(axis=0))
            else:
                centers.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self._centers_hsv = np.stack(centers, axis=0)
        self._calibrated = True

    def _team_for_sig(self, sig: np.ndarray) -> int:
        if self._centers_hsv is None:
            return 0
        d = []
        for c in self._centers_hsv:
            dh = min(abs(sig[0] - c[0]), 180 - abs(sig[0] - c[0])) / 90.0
            ds = abs(sig[1] - c[1]) / 255.0
            dv = abs(sig[2] - c[2]) / 255.0
            d.append(dh * 1.5 + ds * 0.5 + dv * 0.3)
        return int(np.argmin(d))

    def _assign(self, frame: np.ndarray, players: sv.Detections) -> None:
        if players.tracker_id is None:
            return
        for i, tid in enumerate(players.tracker_id):
            tid = int(tid)
            crop = self._jersey_crop(frame, players.xyxy[i])
            sig = self._dominant_hsv(crop)
            if sig is None:
                continue
            t = self._team_for_sig(sig)
            self._player_history.setdefault(tid, []).append(t)
            if len(self._player_history[tid]) >= 3:
                hist = self._player_history[tid][-12:]
                self._player_team[tid] = int(round(sum(hist) / len(hist)))

    def team_of(self, tracker_id: int) -> Optional[int]:
        return self._player_team.get(int(tracker_id))

    def team_ids_for(self, tracker_ids: np.ndarray) -> np.ndarray:
        out = np.full(len(tracker_ids), -1, dtype=np.int32)
        for i, tid in enumerate(tracker_ids):
            t = self.team_of(int(tid))
            if t is not None:
                out[i] = t
        return out

    def centers_bgr(self) -> Optional[List[Tuple[int, int, int]]]:
        if self._centers_hsv is None:
            return None
        out = []
        for c in self._centers_hsv:
            hsv_px = np.uint8([[[c[0], c[1], c[2]]]])
            bgr = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
            out.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return out
