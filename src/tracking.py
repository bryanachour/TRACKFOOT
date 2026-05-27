import supervision as sv


def build_tracker(
    frame_rate: int = 30,
    track_buffer: int = 60,
    match_thresh: float = 0.85,
    high_thresh: float = 0.30,
    low_thresh: float = 0.10,
) -> sv.ByteTrack:
    return sv.ByteTrack(
        track_activation_threshold=high_thresh,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=match_thresh,
        frame_rate=frame_rate,
        minimum_consecutive_frames=1,
    )
