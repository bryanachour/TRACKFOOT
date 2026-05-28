"""Tracker abstraction — uses BoT-SORT via boxmot if available, else falls back
to supervision ByteTrack. Both expose `update_with_detections(sv.Detections)`.

BoT-SORT advantages on amateur football (Veo tribune cam):
- Camera-motion compensation (helps even if camera is fixed: AI-stitched panorama
  has micro-warps that read as camera motion)
- Better Kalman state for sustained motion
- Optional appearance ReID (we disable by default — heavy and we're amateur res)
"""
from typing import Optional

import numpy as np
import supervision as sv


class _BoxmotAdapter:
    """Wraps boxmot's BotSort to look like sv.ByteTrack."""

    def __init__(self, frame_rate: int = 30, track_buffer: int = 60, match_thresh: float = 0.85):
        from boxmot import BotSort
        self._tracker = BotSort(
            reid_weights=None,
            device="cpu",
            half=False,
            with_reid=False,
            track_high_thresh=0.4,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        dets_array = np.concatenate(
            [detections.xyxy, detections.confidence.reshape(-1, 1), detections.class_id.reshape(-1, 1)],
            axis=1,
        )
        tracked = self._tracker.update(dets_array, np.zeros((1, 1, 3), dtype=np.uint8))
        if tracked is None or len(tracked) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=tracked[:, 0:4],
            confidence=tracked[:, 5],
            class_id=tracked[:, 6].astype(int),
            tracker_id=tracked[:, 4].astype(int),
        )


def build_tracker(
    frame_rate: int = 30,
    track_buffer: int = 60,
    match_thresh: float = 0.85,
    high_thresh: float = 0.30,
    low_thresh: float = 0.10,
    prefer: str = "auto",
):
    """Returns a tracker with .update_with_detections(sv.Detections) -> sv.Detections.

    prefer='botsort' forces BoT-SORT (errors if boxmot missing).
    prefer='bytetrack' forces supervision ByteTrack.
    prefer='auto' tries BoT-SORT, falls back to ByteTrack.
    """
    if prefer in ("botsort", "auto"):
        try:
            return _BoxmotAdapter(frame_rate=frame_rate, track_buffer=track_buffer, match_thresh=match_thresh)
        except Exception:
            if prefer == "botsort":
                raise

    return sv.ByteTrack(
        track_activation_threshold=high_thresh,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=match_thresh,
        frame_rate=frame_rate,
        minimum_consecutive_frames=1,
    )
