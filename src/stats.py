from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PlayerStats:
    tracker_id: int
    team: Optional[int] = None
    n_frames: int = 0
    distance_m: float = 0.0
    speed_avg_kmh: float = 0.0
    speed_max_kmh: float = 0.0
    possession_frames: int = 0
    possession_ratio: float = 0.0


@dataclass
class StatsAggregator:
    fps: float
    pitch_length_cm: int = 12000
    pitch_width_cm: int = 7000
    possession_radius_cm: int = 250
    max_speed_kmh: float = 40.0
    speed_window_frames: int = 5

    _last_pos: Dict[int, np.ndarray] = field(default_factory=dict)
    _last_frame: Dict[int, int] = field(default_factory=dict)
    _distance: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    _frames_seen: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _speed_window: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    _speed_max: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    _team_for: Dict[int, int] = field(default_factory=dict)
    _possession: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _total_ball_frames: int = 0
    _current_possessor: Optional[int] = None

    def update(
        self,
        frame_idx: int,
        player_ids: np.ndarray,
        players_xy_cm: np.ndarray,
        ball_xy_cm: np.ndarray,
        team_ids: Optional[np.ndarray] = None,
    ) -> None:
        max_step_cm = (self.max_speed_kmh * 1000 / 3600) * 100 / max(self.fps, 1.0)

        for i, tid in enumerate(player_ids):
            tid = int(tid)
            pos = players_xy_cm[i]
            if not (0 <= pos[0] <= self.pitch_length_cm and 0 <= pos[1] <= self.pitch_width_cm):
                continue
            self._frames_seen[tid] += 1
            if team_ids is not None and team_ids[i] >= 0:
                self._team_for[tid] = int(team_ids[i])

            if tid in self._last_pos:
                dt_frames = frame_idx - self._last_frame[tid]
                if 0 < dt_frames <= int(self.fps * 2):
                    step_cm = float(np.linalg.norm(pos - self._last_pos[tid]))
                    if step_cm <= max_step_cm * max(dt_frames, 1):
                        self._distance[tid] += step_cm / 100.0
                        v_mps = (step_cm / 100.0) / (dt_frames / max(self.fps, 1.0))
                        v_kmh = v_mps * 3.6
                        window = self._speed_window[tid]
                        window.append(v_kmh)
                        if len(window) > self.speed_window_frames:
                            window.pop(0)
                        avg_v = sum(window) / len(window)
                        if avg_v > self._speed_max[tid]:
                            self._speed_max[tid] = avg_v
            self._last_pos[tid] = pos.copy()
            self._last_frame[tid] = frame_idx

        if len(ball_xy_cm) > 0 and len(player_ids) > 0:
            self._total_ball_frames += 1
            bx, by = ball_xy_cm[0]
            if 0 <= bx <= self.pitch_length_cm and 0 <= by <= self.pitch_width_cm:
                dists = np.linalg.norm(players_xy_cm - np.array([bx, by]), axis=1)
                nearest = int(np.argmin(dists))
                if dists[nearest] <= self.possession_radius_cm:
                    tid = int(player_ids[nearest])
                    self._possession[tid] += 1
                    self._current_possessor = tid
                else:
                    self._current_possessor = None

    @property
    def current_possessor(self) -> Optional[int]:
        return self._current_possessor

    def per_player(self) -> List[PlayerStats]:
        out = []
        for tid in sorted(self._frames_seen.keys()):
            n = self._frames_seen[tid]
            dist = self._distance[tid]
            duration_s = n / max(self.fps, 1.0)
            avg_kmh = (dist / duration_s) * 3.6 if duration_s > 0 else 0.0
            poss = self._possession.get(tid, 0)
            ratio = poss / max(self._total_ball_frames, 1) if self._total_ball_frames else 0.0
            out.append(PlayerStats(
                tracker_id=tid,
                team=self._team_for.get(tid),
                n_frames=n,
                distance_m=round(dist, 1),
                speed_avg_kmh=round(avg_kmh, 2),
                speed_max_kmh=round(self._speed_max[tid], 2),
                possession_frames=poss,
                possession_ratio=round(ratio, 4),
            ))
        return out

    def summary(self) -> dict:
        per = [p.__dict__ for p in self.per_player()]
        per.sort(key=lambda p: -p["distance_m"])
        return {
            "fps": self.fps,
            "ball_frames_total": self._total_ball_frames,
            "players": per,
        }

    def save_json(self, path: Path) -> None:
        import json
        path.write_text(json.dumps(self.summary(), indent=2))
