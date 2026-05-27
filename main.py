import argparse
from datetime import datetime
from pathlib import Path

import config as C
from src.pipeline import PipelineOptions, run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRACKFOOT — football detection + tracking + tactical view")
    p.add_argument("--source", type=Path, required=True, help="input video path (mp4)")
    p.add_argument("--output", type=Path, default=None, help="output directory (default: output/run_<timestamp>)")
    p.add_argument("--player-weights", type=Path, default=C.PLAYER_DETECTION_WEIGHTS)
    p.add_argument("--pitch-weights", type=Path, default=C.PITCH_DETECTION_WEIGHTS)
    p.add_argument("--ball-weights", type=Path, default=C.BALL_DETECTION_WEIGHTS)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, mps, 0, ...")
    p.add_argument("--stride", type=int, default=1, help="process 1 in N frames")
    p.add_argument("--no-annotated", action="store_true")
    p.add_argument("--no-tactical", action="store_true")
    p.add_argument("--no-stacked", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output or (C.OUTPUT_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    opts = PipelineOptions(
        source=args.source,
        output_dir=out_dir,
        player_weights=args.player_weights,
        pitch_weights=args.pitch_weights,
        ball_weights=args.ball_weights if args.ball_weights and args.ball_weights.exists() else None,
        device=args.device,
        stride=args.stride,
        save_annotated=not args.no_annotated,
        save_tactical=not args.no_tactical,
        save_stacked=not args.no_stacked,
    )
    run(opts)


if __name__ == "__main__":
    main()
