"""Alert logic with debounce and cooldown, so one defect does not trigger an alert storm."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class Alert:
    timestamp: float
    frame_id: int
    score: float
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    def __init__(self, consecutive_frames: int = 3, cooldown_s: float = 5.0, history_size: int = 100) -> None:
        self.consecutive_frames = consecutive_frames
        self.cooldown_s = cooldown_s
        self.history: deque[Alert] = deque(maxlen=history_size)
        self._streak = 0
        self._last_alert_t = float("-inf")

    def update(self, frame_id: int, score: float, threshold: float, now: float | None = None) -> Alert | None:
        now = time.time() if now is None else now
        self._streak = self._streak + 1 if score > threshold else 0
        if self._streak >= self.consecutive_frames and now - self._last_alert_t >= self.cooldown_s:
            alert = Alert(now, frame_id, score, threshold)
            self.history.append(alert)
            self._last_alert_t = now
            self._streak = 0
            return alert
        return None
