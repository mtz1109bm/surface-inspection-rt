"""Tiling, anomaly maps, image scores and threshold calibration (NumPy / OpenCV, backend-agnostic).

Design choices (see docs/DESIGN.md):
* Full-resolution tiling: thin defects (1-2 px scratches, cracks) vanish if the frame is simply
  down-scaled to the network size, so the frame is split into tiles at native resolution.
* Signed residual, blurred *before* squaring: random texture residual is zero-mean and averages
  out, while a defect has a coherent sign and survives. Squaring first would keep texture energy.
* Multi-scale (sigma 1, 2, 4), each scale normalized by its level on defect-free images, so thin
  scratches (small sigma) and diffuse stains (large sigma) are detected with the same threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SIGMAS = (1.0, 2.0, 4.0)


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def preprocess(frame: np.ndarray, input_size: int, tile: int) -> np.ndarray:
    """uint8 frame -> float32 tiles (n_tiles, 1, tile, tile) in [0, 1], row-major order."""
    g = cv2.resize(to_gray(frame), (input_size, input_size), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32) / 255.0
    n = input_size // tile
    tiles = g.reshape(n, tile, n, tile).transpose(0, 2, 1, 3).reshape(n * n, 1, tile, tile)
    return np.ascontiguousarray(tiles)


def stitch(tiles: np.ndarray) -> np.ndarray:
    """(n*n, 1, t, t) -> (n*t, n*t)."""
    k, _, t, _ = tiles.shape
    n = int(round(k ** 0.5))
    return tiles.reshape(n, n, t, t).transpose(0, 2, 1, 3).reshape(n * t, n * t)


def residual_maps(inp: np.ndarray, recon: np.ndarray) -> list[np.ndarray]:
    """Per-scale squared blurred signed residual on the stitched full-resolution image."""
    d = stitch(inp) - stitch(recon) if inp.ndim == 4 else inp - recon
    return [cv2.GaussianBlur(d, (0, 0), s) ** 2 for s in SIGMAS]


def top_mean(a: np.ndarray, fraction: float = 0.002) -> float:
    flat = np.partition(a.ravel(), -max(1, int(a.size * fraction)))
    return float(flat[-max(1, int(a.size * fraction)):].mean())


def anomaly_map(inp: np.ndarray, recon: np.ndarray, norms: list[float]) -> np.ndarray:
    """Max over scales of normalized residual energy. 1.0 ~ typical top level on good parts."""
    return np.max([m / n for m, n in zip(residual_maps(inp, recon), norms)], axis=0)


def image_score(amap: np.ndarray) -> float:
    """Mean of the top 0.2% pixels: robust to single noisy pixels, sensitive to small defects."""
    return top_mean(amap)


def scale_norms(pairs) -> list[float]:
    """Per-scale normalization constants from (input, reconstruction) pairs of defect-free images."""
    acc = np.zeros(len(SIGMAS))
    n = 0
    for inp, recon in pairs:
        acc += [top_mean(m) for m in residual_maps(inp, recon)]
        n += 1
    return (acc / n).tolist()


def calibrate_threshold(good_scores: np.ndarray, percentile: float = 99.5) -> float:
    """Threshold from defect-free validation images only: bounds the false-reject rate by design."""
    return float(np.percentile(good_scores, percentile))


def save_calibration(path: str | Path, threshold: float, norms: list[float], **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"threshold": threshold, "norms": norms, **meta}, indent=2))


def load_calibration(path: str | Path) -> tuple[float, list[float]]:
    d = json.loads(Path(path).read_text())
    return float(d["threshold"]), list(d["norms"])


def overlay(frame: np.ndarray, amap: np.ndarray, threshold: float) -> np.ndarray:
    """Heat-map overlay + contour of regions above threshold, for the web view."""
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    amap_r = cv2.resize(amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(amap_r / (threshold * 2 + 1e-12), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 0.65, heat, 0.35, 0)
    mask = (amap_r > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out
