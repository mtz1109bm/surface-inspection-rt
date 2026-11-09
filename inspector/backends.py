"""Inference backends with a common interface: PyTorch (reference) and ONNX Runtime (deployment)."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Backend(Protocol):
    name: str

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class TorchBackend:
    name = "torch"

    def __init__(self, weights: str, latent_channels: int = 64) -> None:
        import torch

        from .model import ConvAutoencoder

        self._torch = torch
        self.model = ConvAutoencoder(latent_channels)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            return self.model(self._torch.from_numpy(x)).numpy()


class OnnxBackend:
    name = "onnx"

    def __init__(self, onnx_path: str, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        # On a Jetson, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", ...]
        providers = providers or ort.get_available_providers()
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]


def make_backend(kind: str, model_cfg) -> Backend:
    if kind == "torch":
        return TorchBackend(model_cfg.weights, model_cfg.latent_channels)
    if kind == "onnx":
        return OnnxBackend(model_cfg.onnx)
    raise ValueError(f"Unknown backend: {kind}")
