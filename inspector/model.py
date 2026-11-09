"""Convolutional auto-encoder trained on defect-free images only.

At inference time, regions the model cannot reconstruct well are flagged as anomalies.
Grayscale input, 128x128 by default -> 4 down-sampling stages -> 8x8 latent map.
"""
from __future__ import annotations

import torch
from torch import nn


def _down(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _down(1, 32),
            _down(32, 64),
            _down(64, 128),
            _down(128, 128),
            nn.Conv2d(128, latent_channels, 1),  # bottleneck limits what can be copied through
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 1),
            nn.ReLU(inplace=True),
            _up(128, 128),
            _up(128, 64),
            _up(64, 32),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
