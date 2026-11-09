"""Inference loop: latest frame -> preprocess -> model -> anomaly score -> alert + annotated frame."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

from .alerts import Alert, AlertManager
from .backends import make_backend
from .config import Config
from .scoring import anomaly_map, image_score, load_calibration, overlay, preprocess
from .stream import LatestFrameReader

log = logging.getLogger(__name__)


class InspectionPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.backend = make_backend(cfg.server.backend, cfg.model)
        self.threshold, self.norms = load_calibration(cfg.model.threshold_file)
        self.reader = LatestFrameReader(cfg.stream.source, cfg.stream.reconnect_delay_s)
        self.alerts = AlertManager(cfg.alerts.consecutive_frames, cfg.alerts.cooldown_s, cfg.alerts.history_size)
        self.subscribers: list = []  # callables receiving each new Alert (e.g. WebSocket broadcaster)
        self._jpeg: bytes | None = None
        self._latencies_ms: deque[float] = deque(maxlen=200)
        self._frame_times: deque[float] = deque(maxlen=200)
        self.last_score = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)

    # --- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        self.reader.start()
        self._thread.start()
        log.info("Pipeline started (backend=%s, threshold=%.3f)", self.backend.name, self.threshold)

    def stop(self) -> None:
        self._stop.set()
        self.reader.stop()
        self._thread.join(timeout=5)

    # --- core ------------------------------------------------------------------------------
    def _crop(self, frame: np.ndarray) -> np.ndarray:
        if self.cfg.stream.roi:
            x, y, w, h = self.cfg.stream.roi
            return frame[y : y + h, x : x + w]
        return frame

    def process(self, frame: np.ndarray, frame_id: int = 0) -> tuple[float, np.ndarray, Alert | None]:
        t0 = time.perf_counter()
        roi = self._crop(frame)
        x = preprocess(roi, self.cfg.model.input_size, self.cfg.model.image_size)
        recon = self.backend(x)
        amap = anomaly_map(x, recon, self.norms)
        score = image_score(amap)
        self._latencies_ms.append((time.perf_counter() - t0) * 1000)
        alert = self.alerts.update(frame_id, score, self.threshold)
        return score, overlay(roi, amap, self.threshold), alert

    def _run(self) -> None:
        last_id = -1
        while not self._stop.is_set():
            frame_id, frame = self.reader.read()
            if frame is None or frame_id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame_id
            score, vis, alert = self.process(frame, frame_id)
            self.last_score = score
            self._frame_times.append(time.time())
            color = (0, 0, 255) if score > self.threshold else (0, 200, 0)
            cv2.putText(vis, f"score {score:.2f} / seuil {self.threshold:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._jpeg = buf.tobytes()
            if alert:
                log.warning("ANOMALY frame=%d score=%.3f", alert.frame_id, alert.score)
                for callback in list(self.subscribers):
                    callback(alert)

    # --- read-only views for the web layer --------------------------------------------------
    @property
    def jpeg(self) -> bytes | None:
        return self._jpeg

    def stats(self) -> dict:
        lat = np.array(self._latencies_ms) if self._latencies_ms else np.array([0.0])
        ft = list(self._frame_times)
        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else 0.0
        return {
            "backend": self.backend.name,
            "connected": self.reader.connected,
            "fps": round(fps, 1),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "last_score": round(self.last_score, 3),
            "threshold": round(self.threshold, 3),
            "alerts_total": len(self.alerts.history),
        }
