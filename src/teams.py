"""SigLIP-based team classifier — drop-in replacement for HSV k-means.

Pattern from roboflow/sports/examples/soccer/team.py:
  1. Crop player bbox jersey region from each frame
  2. Encode each crop to a vision embedding (SigLIP)
  3. After N calibration frames, fit UMAP(3D) + KMeans(k=2)
  4. For subsequent frames, embed → reduce → predict cluster
  5. Per-tracker_id rolling mode → stable team assignment

Falls back to a no-op classifier if transformers/umap-learn aren't installed.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv


SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"


def _try_imports():
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        import umap
        from sklearn.cluster import KMeans
        return torch, AutoModel, AutoProcessor, umap, KMeans
    except ImportError:
        return None


@dataclass
class TeamClassifier:
    n_calibration_frames: int = 30
    crops_per_frame_max: int = 30
    crop_top_ratio: float = 0.10
    crop_bot_ratio: float = 0.55
    crop_side_ratio: float = 0.10
    rolling_window: int = 12
    device: Optional[str] = None

    _embed_model: Optional[object] = None
    _processor: Optional[object] = None
    _reducer: Optional[object] = None
    _kmeans: Optional[object] = None
    _torch_device: Optional[str] = None
    _crop_buffer: List[np.ndarray] = field(default_factory=list)
    _frames_observed: int = 0
    _calibrated: bool = False
    _disabled: bool = False
    _player_history: Dict[int, Deque[int]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=12)))
    _player_team: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        imports = _try_imports()
        if imports is None:
            self._disabled = True
            return
        torch, AutoModel, AutoProcessor, umap, KMeans = imports
        try:
            self._torch_device = "cuda" if (torch.cuda.is_available() and self.device != "cpu") else "cpu"
            self._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
            self._embed_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(self._torch_device).eval()
            self._UMAP = umap.UMAP
            self._KMeans = KMeans
            self._torch = torch
        except Exception:
            self._disabled = True

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def disabled(self) -> bool:
        return self._disabled

    def _crop_jersey(self, frame: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = xyxy.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 18:
            return None
        h = y2 - y1; w = x2 - x1
        yt = y1 + int(h * self.crop_top_ratio)
        yb = y1 + int(h * self.crop_bot_ratio)
        xl = x1 + int(w * self.crop_side_ratio)
        xr = x2 - int(w * self.crop_side_ratio)
        crop = frame[yt:yb, xl:xr]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def _embed(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops or self._disabled:
            return np.zeros((0, 768), dtype=np.float32)
        inputs = self._processor(images=crops, return_tensors="pt").to(self._torch_device)
        with self._torch.no_grad():
            feats = self._embed_model.get_image_features(**inputs)
        return feats.cpu().numpy()

    def observe(self, frame: np.ndarray, players: sv.Detections) -> None:
        if self._disabled or len(players) == 0:
            return
        if self._calibrated:
            self._assign(frame, players)
            return

        added = 0
        for i in range(len(players)):
            if added >= self.crops_per_frame_max:
                break
            crop = self._crop_jersey(frame, players.xyxy[i])
            if crop is not None:
                self._crop_buffer.append(crop)
                added += 1
        self._frames_observed += 1

        if self._frames_observed >= self.n_calibration_frames and len(self._crop_buffer) >= 20:
            self._fit()

    def _fit(self) -> None:
        try:
            features = self._embed(self._crop_buffer)
            if features.shape[0] < 12:
                self._disabled = True
                return
            n_neighbors = min(15, max(2, features.shape[0] - 1))
            self._reducer = self._UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
            reduced = self._reducer.fit_transform(features)
            self._kmeans = self._KMeans(n_clusters=2, n_init=10, random_state=42).fit(reduced)
            self._calibrated = True
            self._crop_buffer = []
        except Exception:
            self._disabled = True

    def _assign(self, frame: np.ndarray, players: sv.Detections) -> None:
        if players.tracker_id is None:
            return
        valid_crops: List[np.ndarray] = []
        valid_tids: List[int] = []
        for i, tid in enumerate(players.tracker_id):
            crop = self._crop_jersey(frame, players.xyxy[i])
            if crop is not None:
                valid_crops.append(crop)
                valid_tids.append(int(tid))
        if not valid_crops:
            return
        try:
            feats = self._embed(valid_crops)
            reduced = self._reducer.transform(feats)
            labels = self._kmeans.predict(reduced)
            for tid, lbl in zip(valid_tids, labels):
                self._player_history[tid].append(int(lbl))
                if len(self._player_history[tid]) >= 3:
                    hist = list(self._player_history[tid])
                    self._player_team[tid] = int(round(sum(hist) / len(hist)))
        except Exception:
            pass

    def team_of(self, tracker_id: int) -> Optional[int]:
        return self._player_team.get(int(tracker_id))

    def team_ids_for(self, tracker_ids: np.ndarray) -> np.ndarray:
        out = np.full(len(tracker_ids), -1, dtype=np.int32)
        for i, tid in enumerate(tracker_ids):
            t = self.team_of(int(tid))
            if t is not None:
                out[i] = t
        return out
