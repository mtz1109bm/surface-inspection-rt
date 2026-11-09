"""Evaluate on data/test: image AUROC (global and per defect type), pixel AUROC,
detection rate and false-reject rate at the calibrated threshold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector.backends import make_backend  # noqa: E402
from inspector.config import load_config  # noqa: E402
from inspector.dataset import list_images  # noqa: E402
from inspector.scoring import anomaly_map, image_score, load_calibration, preprocess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--backend", default="onnx", choices=["torch", "onnx"])
    ap.add_argument("--out", default="results/metrics.json")
    a = ap.parse_args()

    cfg = load_config()
    tile, size = cfg.model.image_size, cfg.model.input_size
    model = make_backend(a.backend, cfg.model)
    thr, norms = load_calibration(cfg.model.threshold_file)
    test = Path(a.data) / "test"

    scores: dict[str, list[float]] = {}
    pix_true, pix_score = [], []
    for kind_dir in sorted(p for p in test.iterdir() if p.is_dir()):
        kind = kind_dir.name
        scores[kind] = []
        for path in list_images(kind_dir):
            x = preprocess(cv2.imread(str(path)), size, tile)
            amap = anomaly_map(x, model(x), norms)
            scores[kind].append(image_score(amap))
            if kind != "good":
                m = cv2.imread(str(Path(a.data) / "ground_truth" / kind / f"{path.stem}_mask.png"), 0)
                m = cv2.resize(m, amap.shape[::-1], interpolation=cv2.INTER_NEAREST)
                pix_true.append((m > 0).ravel())
                pix_score.append(amap.ravel())

    good = np.array(scores.pop("good"))
    defects = np.concatenate([np.array(v) for v in scores.values()])
    labels = np.r_[np.zeros(len(good)), np.ones(len(defects))]
    metrics = {
        "backend": a.backend,
        "n_good": len(good),
        "n_defective": len(defects),
        "threshold": round(thr, 4),
        "image_auroc": round(float(roc_auc_score(labels, np.r_[good, defects])), 4),
        "pixel_auroc": round(float(roc_auc_score(np.concatenate(pix_true), np.concatenate(pix_score))), 4),
        "detection_rate": round(float((defects > thr).mean()), 4),
        "false_reject_rate": round(float((good > thr).mean()), 4),
        "per_type": {
            k: {"n": len(v),
                "image_auroc": round(float(roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(v))],
                                                         np.r_[good, v])), 4),
                "detection_rate": round(float((np.array(v) > thr).mean()), 4)}
            for k, v in scores.items()
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
