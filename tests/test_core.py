import numpy as np
import pytest

from inspector.alerts import AlertManager
from inspector.config import load_config
from inspector.scoring import (anomaly_map, calibrate_threshold, image_score, load_calibration, overlay,
                               preprocess, save_calibration, scale_norms, stitch)


def test_preprocess_tiles_and_stitch_roundtrip():
    frame = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    tiles = preprocess(frame, 256, 128)
    assert tiles.shape == (4, 1, 128, 128) and tiles.dtype == np.float32
    assert 0.0 <= tiles.min() and tiles.max() <= 1.0
    np.testing.assert_allclose(stitch(tiles), frame.astype(np.float32) / 255.0)


def test_preprocess_accepts_color_frames_of_any_size():
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    assert preprocess(frame, 256, 128).shape == (4, 1, 128, 128)


def _pair(rng, defect: bool):
    x = rng.random((4, 1, 128, 128), dtype=np.float32)
    recon = x + rng.normal(0, 0.05, x.shape).astype(np.float32)  # texture-like zero-mean residual
    if defect:
        x = x.copy()
        x[0, 0, 60:62, 20:100] += 0.3  # thin "scratch" with coherent sign
    return x, recon


def test_thin_coherent_defect_scores_higher_than_texture_noise():
    rng = np.random.default_rng(0)
    norms = scale_norms(_pair(rng, False) for _ in range(5))
    good = [image_score(anomaly_map(*_pair(rng, False), norms)) for _ in range(5)]
    bad = image_score(anomaly_map(*_pair(rng, True), norms))
    assert bad > 3 * max(good)


def test_threshold_bounds_false_rejects(tmp_path):
    good = np.random.default_rng(1).gamma(2.0, 1.0, 10_000)
    thr = calibrate_threshold(good, 99.5)
    assert (good > thr).mean() == pytest.approx(0.005, abs=1e-3)
    save_calibration(tmp_path / "c.json", thr, [1.0, 2.0, 3.0], percentile=99.5)
    t2, norms = load_calibration(tmp_path / "c.json")
    assert t2 == pytest.approx(thr) and norms == [1.0, 2.0, 3.0]


def test_alert_debounce_and_cooldown():
    m = AlertManager(consecutive_frames=3, cooldown_s=5.0)
    assert m.update(1, 2.0, 1.0, now=0.0) is None
    assert m.update(2, 2.0, 1.0, now=0.1) is None
    assert m.update(3, 2.0, 1.0, now=0.2) is not None  # 3rd consecutive anomalous frame
    for i in range(4, 10):  # still anomalous but within cooldown
        assert m.update(i, 2.0, 1.0, now=0.2 + i * 0.1) is None
    assert m.update(10, 0.5, 1.0, now=10.0) is None  # streak reset by a good frame
    assert len(m.history) == 1


def test_isolated_spike_does_not_alert():
    m = AlertManager(consecutive_frames=3, cooldown_s=0)
    scores = [0.1, 2.0, 0.1, 2.0, 2.0, 0.1]
    assert all(m.update(i, s, 1.0, now=float(i)) is None for i, s in enumerate(scores))


def test_overlay_keeps_frame_size():
    frame = np.zeros((240, 320, 3), np.uint8)
    amap = np.zeros((256, 256), np.float32)
    amap[100:120, 100:120] = 1.0
    out = overlay(frame, amap, 0.5)
    assert out.shape == frame.shape and out[..., 2].max() == 255  # red contour drawn


def test_config_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STREAM_SOURCE", "rtsp://cam:8554/x")
    monkeypatch.setenv("INFERENCE_BACKEND", "torch")
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.stream.source == "rtsp://cam:8554/x" and cfg.server.backend == "torch"
