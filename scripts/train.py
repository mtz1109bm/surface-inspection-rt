"""Train the auto-encoder on defect-free tiles, then calibrate scale norms and threshold on val/good."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector.config import load_config  # noqa: E402
from inspector.backends import TorchBackend  # noqa: E402
from inspector.dataset import GoodTiles  # noqa: E402
from inspector.model import ConvAutoencoder, count_parameters  # noqa: E402
from scripts.calibrate import calibrate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--percentile", type=float, default=99.5, help="threshold percentile on val/good scores")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config()
    torch.manual_seed(a.seed)
    tile, size = cfg.model.image_size, cfg.model.input_size
    train = GoodTiles(Path(a.data) / "train" / "good", size, tile, augment=True, seed=a.seed)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0)

    model = ConvAutoencoder(cfg.model.latent_channels)
    print(f"Parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        if epoch % 5 == 0 or epoch == a.epochs:
            print(f"epoch {epoch:3d}/{a.epochs}  loss {total / len(train):.6f}  ({time.time() - t0:.0f}s)")

    Path(cfg.model.weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.model.weights)

    # Calibration uses defect-free validation images only (no defect example is ever needed).
    thr, norms = calibrate(TorchBackend(cfg.model.weights, cfg.model.latent_channels), cfg, a.data, a.percentile)
    print(f"Saved {cfg.model.weights}; threshold={thr:.4f}, norms={np.round(norms, 7).tolist()}")


if __name__ == "__main__":
    main()
