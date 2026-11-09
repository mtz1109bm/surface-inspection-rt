from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from inspector.alerts import Alert, AlertManager  # noqa: E402
from inspector.web import create_app  # noqa: E402


class FakePipeline:
    """Stands in for InspectionPipeline: no model, no stream."""

    def __init__(self):
        self.subscribers = []
        self.reader = SimpleNamespace(connected=True)
        self.alerts = AlertManager()
        self.alerts.history.append(Alert(1.0, 7, 3.2, 2.4))
        self.jpeg = None
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def stats(self):
        return {"backend": "fake", "fps": 25.0}


def test_endpoints_and_websocket_push():
    p = FakePipeline()
    with TestClient(create_app(p)) as client:
        assert p.started
        assert client.get("/health").json() == {"status": "ok", "stream_connected": True}
        assert client.get("/api/stats").json()["backend"] == "fake"
        assert client.get("/api/alerts").json()[0]["frame_id"] == 7
        assert "<html" in client.get("/").text
        with client.websocket_connect("/ws") as ws:
            p.subscribers[0](Alert(2.0, 42, 5.0, 2.4))  # what the inference thread does
            assert ws.receive_json()["frame_id"] == 42
    assert p.stopped
