"""Image folder dataset (MVTec AD layout): random full-resolution tiles from defect-free images."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def load_gray(path: str | Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


class GoodTiles:
    """Torch-compatible dataset (__len__/__getitem__) of random tile crops, without importing torch.

    Random crops at native resolution give the network many more distinct training samples than
    whole images, and match exactly what it sees at inference (tiles of the full-resolution frame).
    """

    def __init__(self, folder: str | Path, input_size: int, tile: int, augment: bool = True, seed: int = 0):
        self.paths = list_images(folder)
        if not self.paths:
            raise FileNotFoundError(f"No images in {folder}")
        self.images = [load_gray(p, input_size) for p in self.paths]  # small dataset: keep in RAM
        self.input_size, self.tile, self.augment = input_size, tile, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> np.ndarray:
        img = self.images[i]
        y, x = self.rng.integers(0, self.input_size - self.tile + 1, 2)
        t = img[y : y + self.tile, x : x + self.tile]
        if self.augment:
            if self.rng.random() < 0.5:
                t = t[:, ::-1]
            if self.rng.random() < 0.5:
                t = t[::-1, :]
            t = np.clip(t * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05), 0, 1)
        return np.ascontiguousarray(t[None], dtype=np.float32)
