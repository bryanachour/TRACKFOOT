"""Download pre-trained weights used by roboflow/sports football pipeline.

Hosted on Google Drive (public, no auth needed). Source URLs taken from:
https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh

Models:
  - football-player-detection.pt  (3zvbc/v11 — player/keeper/ref/ball)
  - football-pitch-detection.pt   (f07vi/v14 — 32 pitch keypoints)
  - football-ball-detection.pt    (rejhg/v2  — small-ball specialised)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gdown

import config as C


WEIGHTS = [
    ("17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q", C.PLAYER_DETECTION_WEIGHTS),
    ("1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf", C.PITCH_DETECTION_WEIGHTS),
    ("1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V", C.BALL_DETECTION_WEIGHTS),
]


def main() -> None:
    C.WEIGHTS_DIR.mkdir(exist_ok=True)
    for gdrive_id, target in WEIGHTS:
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"skip {target.name} (exists, {target.stat().st_size // 1024} KB)")
            continue
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        print(f"downloading {target.name}")
        gdown.download(url, str(target), quiet=False)
        if not target.exists() or target.stat().st_size < 1_000_000:
            raise RuntimeError(f"download failed for {target.name}")
        print(f"  saved -> {target} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
