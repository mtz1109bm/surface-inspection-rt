"""Generate a reproducible synthetic surface-inspection dataset (brushed metal-like textures).

Layout (same convention as MVTec AD, so the real dataset can be swapped in):
    data/train/good/*.png         defect-free, for training
    data/val/good/*.png           defect-free, for threshold calibration
    data/test/good/*.png
    data/test/<defect>/*.png      scratch, pit, crack, stain
    data/ground_truth/<defect>/*_mask.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFECTS = ("scratch", "pit", "crack", "stain")


def surface(rng: np.random.Generator, size: int) -> np.ndarray:
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    k = int(rng.integers(9, 21))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k  # horizontal "brushed" texture
    angle = rng.uniform(-8, 8)
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    tex = cv2.filter2D(noise, -1, kernel / max(kernel.sum(), 1e-6))
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gx, gy = rng.uniform(-0.15, 0.15, 2)  # slow illumination gradient
    base = rng.uniform(0.45, 0.6)
    img = base + gx * (xx - 0.5) + gy * (yy - 0.5) + 0.06 * tex
    img += rng.normal(0, 0.01, img.shape)  # sensor noise
    return np.clip(img, 0, 1)


def add_defect(rng: np.random.Generator, img: np.ndarray, kind: str, strength: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """strength scales defect contrast: 1.0 = clearly visible to an operator, 0.5 = hard."""
    size = img.shape[0]
    mask = np.zeros_like(img, np.uint8)
    if kind == "scratch":
        p1 = rng.integers(20, size - 20, 2)
        length = rng.integers(size // 6, size // 3)
        ang = rng.uniform(0, np.pi)
        p2 = np.clip(p1 + (length * np.array([np.cos(ang), np.sin(ang)])).astype(int), 0, size - 1)
        cv2.line(mask, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 1, int(rng.integers(2, 4)))
        delta = rng.choice([-1, 1]) * rng.uniform(0.15, 0.3)
    elif kind == "pit":
        c = tuple(int(v) for v in rng.integers(20, size - 20, 2))
        axes = tuple(int(v) for v in rng.integers(3, 8, 2))
        cv2.ellipse(mask, c, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        delta = -rng.uniform(0.2, 0.35)
    elif kind == "crack":
        p = rng.integers(30, size - 30, 2).astype(float)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(25, 50))):
            ang += rng.normal(0, 0.45)
            q = np.clip(p + 2.5 * np.array([np.cos(ang), np.sin(ang)]), 0, size - 1)
            cv2.line(mask, tuple(int(v) for v in p), tuple(int(v) for v in q), 1, 2)
            p = q
        delta = -rng.uniform(0.2, 0.3)
    elif kind == "stain":
        blob = np.zeros_like(img, np.float32)
        c = rng.integers(30, size - 30, 2)
        for _ in range(4):
            off = tuple(int(v) for v in c + rng.integers(-8, 8, 2))
            cv2.circle(blob, off, int(rng.integers(5, 12)), 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 3)
        mask = (blob > 0.3).astype(np.uint8)
        delta = rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)
        return np.clip(img + strength * delta * blob, 0, 1), mask
    else:
        raise ValueError(kind)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    return np.clip(img + strength * delta * soft, 0, 1), mask


def save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (img * 255).round().astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test-good", type=int, default=200)
    ap.add_argument("--n-test-defect", type=int, default=50, help="per defect type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defect-strength", type=float, default=1.0, help="defect contrast multiplier")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    for split, n in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test_good)):
        for i in range(n):
            save(out / split / "good" / f"{i:04d}.png", surface(rng, a.size))
    for kind in DEFECTS:
        for i in range(a.n_test_defect):
            img, mask = add_defect(rng, surface(rng, a.size), kind, a.defect_strength)
            save(out / "test" / kind / f"{i:04d}.png", img)
            save(out / "ground_truth" / kind / f"{i:04d}_mask.png", mask.astype(np.float32))
    print(f"Dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
