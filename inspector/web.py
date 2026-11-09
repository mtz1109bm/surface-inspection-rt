"""FastAPI app: live annotated MJPEG stream, JSON stats, alert history and WebSocket push."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from .config import load_config
from .pipeline import InspectionPipeline

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Broadcaster:
    """Bridges alerts from the inference thread to async WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, alert) -> None:  # called from the inference thread
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._send(alert.to_dict()), self.loop)

    async def _send(self, payload: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.clients.discard(ws)


def create_app(pipeline: InspectionPipeline | None = None) -> FastAPI:
    broadcaster = Broadcaster()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        p = pipeline or InspectionPipeline(load_config())
        broadcaster.loop = asyncio.get_running_loop()
        p.subscribers.append(broadcaster.publish)
        p.start()
        state["p"] = p
        yield
        p.stop()

    app = FastAPI(title="Surface Inspection RT", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "stream_connected": state["p"].reader.connected}

    @app.get("/api/stats")
    def stats():
        return state["p"].stats()

    @app.get("/api/alerts")
    def alerts():
        return [a.to_dict() for a in reversed(state["p"].alerts.history)]

    @app.get("/video.mjpg")
    def video():
        async def gen():
            last = None
            while True:
                jpg = state["p"].jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.03)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        broadcaster.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()  # keep-alive; client messages are ignored
        except WebSocketDisconnect:
            broadcaster.clients.discard(websocket)

    app.state.started = time.time()
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    uvicorn.run(create_app(), host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
