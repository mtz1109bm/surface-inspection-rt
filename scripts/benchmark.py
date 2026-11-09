"""Measure end-to-end per-frame latency (preprocess + inference + scoring) for each backend.

Reports p50 / p95 in milliseconds and throughput, with warm-up, on a fixed set of test images.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector.backends import make_backend  # noqa: E402
from inspector.config import load_config  # noqa: E402
from inspector.dataset import list_images  # noqa: E402
from inspector.scoring import anomaly_map, image_score, load_calibration, preprocess  # noqa: E402


def run(backend, frames, size: int, tile: int, norms, warmup: int, iters: int) -> dict:
    for f in frames[:warmup]:
        backend(preprocess(f, size, tile))
    times = []
    for i in range(iters):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        x = preprocess(f, size, tile)
        image_score(anomaly_map(x, backend(x), norms))
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2),
            "fps": round(1000 / float(t.mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (1 = conservative, embedded-like)")
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    cfg = load_config()
    paths = list_images(Path(a.data) / "test" / "good")[:50] + list_images(Path(a.data) / "test" / "scratch")[:50]
    frames = [cv2.imread(str(p)) for p in paths]

    _, norms = load_calibration(cfg.model.threshold_file)
    n_tiles = (cfg.model.input_size // cfg.model.image_size) ** 2
    results = {"cpu": platform.processor() or platform.machine(), "threads": a.threads,
               "input": f"{cfg.model.input_size}x{cfg.model.input_size} frame -> "
                        f"{n_tiles} tiles of {cfg.model.image_size}x{cfg.model.image_size}"}
    for kind in ("torch", "onnx"):
        if kind == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = a.threads
            backend = make_backend("onnx", cfg.model)
            backend.session = ort.InferenceSession(cfg.model.onnx, so, providers=["CPUExecutionProvider"])
        else:
            backend = make_backend("torch", cfg.model)
        results[kind] = run(backend, frames, cfg.model.input_size, cfg.model.image_size, norms,
                            a.warmup, a.iters)
        print(kind, results[kind])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
