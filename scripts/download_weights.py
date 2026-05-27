"""Download pre-trained weights from Roboflow.

Set ROBOFLOW_API_KEY env var, or pass --api-key. Free tier works.

Models used (Roboflow Universe):
  - football-players-detection-3zvbc / v11   (player, goalkeeper, referee, ball)
  - football-field-detection-f07vi / v14     (32 pitch keypoints)
  - football-ball-detection-rejhg / v2       (small-ball specialised, optional)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roboflow import Roboflow

import config as C


MODELS = [
    ("roboflow-jvuqo", "football-players-detection-3zvbc", 11, C.PLAYER_DETECTION_WEIGHTS),
    ("roboflow-jvuqo", "football-field-detection-f07vi", 14, C.PITCH_DETECTION_WEIGHTS),
    ("roboflow-jvuqo", "football-ball-detection-rejhg", 2, C.BALL_DETECTION_WEIGHTS),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("set ROBOFLOW_API_KEY or pass --api-key")

    rf = Roboflow(api_key=args.api_key)
    C.WEIGHTS_DIR.mkdir(exist_ok=True)

    for workspace, project_slug, version, target in MODELS:
        if target.exists():
            print(f"skip {target.name} (exists)")
            continue
        print(f"downloading {project_slug} v{version} -> {target.name}")
        project = rf.workspace(workspace).project(project_slug)
        version_obj = project.version(version)
        dataset = version_obj.download("yolov8", location=str(C.WEIGHTS_DIR / project_slug))
        weights_src = Path(dataset.location) / "weights" / "best.pt"
        if not weights_src.exists():
            for cand in Path(dataset.location).rglob("best.pt"):
                weights_src = cand
                break
        if not weights_src.exists():
            print(f"  WARN: no best.pt under {dataset.location}; download dataset only")
            continue
        shutil.copy(weights_src, target)
        print(f"  saved -> {target}")


if __name__ == "__main__":
    main()
