"""Export the trained model to ONNX and check numerical parity with PyTorch.

On an NVIDIA Jetson, the ONNX file can then be compiled into a TensorRT engine:
    trtexec --onnx=models/autoencoder.onnx --saveEngine=models/autoencoder_fp16.engine --fp16
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector.backends import OnnxBackend  # noqa: E402
from inspector.config import load_config  # noqa: E402
from inspector.model import ConvAutoencoder  # noqa: E402


def main() -> None:
    cfg = load_config()
    model = ConvAutoencoder(cfg.model.latent_channels)
    model.load_state_dict(torch.load(cfg.model.weights, map_location="cpu"))
    model.eval()
    s = cfg.model.image_size
    n_tiles = (cfg.model.input_size // s) ** 2
    dummy = torch.rand(n_tiles, 1, s, s)  # all tiles of a frame in one batch
    torch.onnx.export(model, dummy, cfg.model.onnx, input_names=["tiles"], output_names=["reconstruction"],
                      opset_version=17, dynamo=False, dynamic_axes={"tiles": {0: "n"}, "reconstruction": {0: "n"}})
    ref = model(dummy).detach().numpy()
    out = OnnxBackend(cfg.model.onnx, providers=["CPUExecutionProvider"])(dummy.numpy())
    err = float(np.abs(ref - out).max())
    print(f"Exported {cfg.model.onnx}; max |torch - onnx| = {err:.2e}")
    assert err < 1e-4, "ONNX output diverges from PyTorch"


if __name__ == "__main__":
    main()
