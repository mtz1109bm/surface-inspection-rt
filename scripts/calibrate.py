"""(Re)calibrate scale norms and decision threshold on val/good, without retraining.

Use it when the camera, lighting or the false-reject target changes:
    python scripts/calibrate.py --percentile 99.9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector.backends import make_backend  # noqa: E402
from inspector.config import load_config  # noqa: E402
from inspector.dataset import list_images  # noqa: E402
from inspector.scoring import (anomaly_map, calibrate_threshold, image_score, preprocess,  # noqa: E402
                               save_calibration, scale_norms)


def calibrate(backend, cfg, data: str = "data", percentile: float = 99.5) -> tuple[float, list[float]]:
    tile, size = cfg.model.image_size, cfg.model.input_size
    pairs = []
    for p in list_images(Path(data) / "val" / "good"):
        x = preprocess(cv2.imread(str(p)), size, tile)
        pairs.append((x, backend(x)))
    half = len(pairs) // 2  # first half: per-scale norms, second half: threshold (no reuse)
    norms = scale_norms(pairs[:half])
    scores = np.array([image_score(anomaly_map(x, r, norms)) for x, r in pairs[half:]])
    thr = calibrate_threshold(scores, percentile)
    save_calibration(cfg.model.threshold_file, thr, norms, percentile=percentile, n_val=len(pairs))
    return thr, norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--percentile", type=float, default=99.5)
    a = ap.parse_args()
    cfg = load_config()
    thr, norms = calibrate(make_backend(a.backend, cfg.model), cfg, a.data, a.percentile)
    print(f"threshold={thr:.4f} (p{a.percentile}), norms={np.round(norms, 7).tolist()}")


if __name__ == "__main__":
    main()
