"""Threaded video reader that always exposes the latest frame and reconnects on failure.

Reading in a dedicated thread prevents RTSP buffer build-up: if inference is slower than the
camera, old frames are dropped instead of accumulating latency.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class LatestFrameReader:
    def __init__(self, source: str, reconnect_delay_s: float = 2.0) -> None:
        self.source = source
        self.reconnect_delay_s = reconnect_delay_s
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stream-reader", daemon=True)
        self.connected = False

    def start(self) -> "LatestFrameReader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def read(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, self._frame

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        is_file = not self.source.startswith(("rtsp://", "http://", "https://"))
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                log.warning("Cannot open %s, retrying in %.1fs", self.source, self.reconnect_delay_s)
                time.sleep(self.reconnect_delay_s)
                continue
            self.connected = True
            log.info("Connected to %s", self.source)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # loop local video files (demo mode)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning("Stream lost, reconnecting")
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                if is_file:
                    time.sleep(1.0 / fps)  # emulate real-time pacing
            cap.release()
            self.connected = False
            time.sleep(self.reconnect_delay_s)
