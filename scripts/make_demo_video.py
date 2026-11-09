"""Build a demo video simulating a conveyor: mostly good parts, with defective parts mixed in.

The video is served over RTSP by docker-compose (mediamtx + ffmpeg) to emulate an industrial camera.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector.dataset import list_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="demo/line1.mp4")
    ap.add_argument("--parts", type=int, default=60)
    ap.add_argument("--defect-ratio", type=float, default=0.15)
    ap.add_argument("--frames-per-part", type=int, default=12)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    test = Path(a.data) / "test"
    good = list_images(test / "good")
    bad = [p for d in test.iterdir() if d.is_dir() and d.name != "good" for p in list_images(d)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    size = cv2.imread(str(good[0])).shape[0]
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (size, size))
    for _ in range(a.parts):
        pool = bad if rng.random() < a.defect_ratio else good
        img = cv2.imread(str(pool[rng.integers(len(pool))]))
        for _ in range(a.frames_per_part):  # part stays under the camera for a few frames
            writer.write(img)
    writer.release()
    print(f"Wrote {a.out} ({a.parts} parts, {a.parts * a.frames_per_part / a.fps:.0f}s)")


if __name__ == "__main__":
    main()
