"""Typed configuration loaded from YAML, overridable by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    image_size: int = 128  # tile size seen by the network
    input_size: int = 128  # frame is resized to input_size, then split into (input_size/image_size)^2 tiles
    latent_channels: int = 64
    weights: str = "models/autoencoder.pt"
    onnx: str = "models/autoencoder.onnx"
    threshold_file: str = "models/threshold.json"


@dataclass
class StreamConfig:
    source: str = "rtsp://localhost:8554/line1"
    reconnect_delay_s: float = 2.0
    roi: list[int] | None = None  # [x, y, w, h] in source pixels, None = full frame


@dataclass
class AlertConfig:
    consecutive_frames: int = 3  # debounce: N anomalous frames in a row before alerting
    cooldown_s: float = 5.0  # minimum time between two alerts
    history_size: int = 100


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    backend: str = "onnx"  # "onnx" or "torch"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _merge(section, values: dict) -> None:
    names = {f.name for f in fields(section)}
    for key, value in (values or {}).items():
        if key not in names:
            raise KeyError(f"Unknown config key: {type(section).__name__}.{key}")
        setattr(section, key, value)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    path = path or os.environ.get("INSPECTOR_CONFIG", "config.yaml")
    if Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for name in ("model", "stream", "alerts", "server"):
            _merge(getattr(cfg, name), raw.get(name, {}))
    # Most useful overrides for Docker deployments
    if src := os.environ.get("STREAM_SOURCE"):
        cfg.stream.source = src
    if backend := os.environ.get("INFERENCE_BACKEND"):
        cfg.server.backend = backend
    return cfg
