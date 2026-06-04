"""
api.websocket.stream
=====================

WebSocket endpoint for real-time streaming inference.

Clients open a persistent WebSocket connection to ``/ws/live`` and push
raw camera frames as binary messages.  The server decodes each frame,
runs the full RoadSage pipeline, and pushes back a JSON prediction.

Protocol
--------
Client → Server:
    Binary message containing a JPEG- or PNG-encoded image frame.

Server → Client (message types):

``connected``::

    {"type": "connected", "session_id": "<uuid>",
     "message": "RoadSage streaming ready. Send image frames as binary messages."}

``skipped``::

    {"type": "skipped", "frame": <int>}

``prediction``::

    {
        "type": "prediction",
        "frame_id": <int>,
        "session_fps": <float>,
        "result": { ...PredictionResult fields... }
    }

``error``::

    {"type": "error", "message": "<str>"}

Frame-rate control
------------------
* Hard cap: ``MIN_FRAME_INTERVAL = 1/30 s`` — binary messages that arrive
  faster than 30 FPS are silently dropped before any decoding work is done.
* Processing cadence: only every ``PROCESS_EVERY_N = 2``-nd accepted frame
  is actually run through the pipeline (≈15 FPS output).  The skipped
  frames receive a lightweight ``{"type":"skipped"}`` acknowledgement.

Typical usage (Python ``websockets`` client)::

    import asyncio, websockets, cv2

    async def stream():
        async with websockets.connect("ws://localhost:8000/ws/live") as ws:
            print(await ws.recv())          # "connected"
            for frame in frames:
                _, buf = cv2.imencode(".jpg", frame)
                await ws.send(buf.tobytes())
                print(await ws.recv())      # "prediction" or "skipped"

    asyncio.run(stream())
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from api.metrics import record_prediction_metrics

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class StreamSession:
    """Per-connection state for a single WebSocket client.

    Attributes:
        session_id: Unique identifier assigned at connection time (UUID4).
        connected_at: UTC timestamp of the ``accept()`` call.
        frames_received: Total binary messages received (including dropped /
            skipped frames).
        frames_processed: Frames that completed the full inference pipeline.
        last_frame_time: ``time.time()`` of the most recently processed
            binary message, or ``None`` before the first message arrives.
        fps: Instantaneous receive rate estimated from consecutive
            inter-frame intervals (updated by :meth:`update_fps`).
    """

    session_id: str
    connected_at: datetime
    frames_received: int = 0
    frames_processed: int = 0
    last_frame_time: Optional[float] = None
    fps: float = 0.0

    def update_fps(self) -> None:
        """Estimate instantaneous FPS from the last two frame arrival times.

        Uses the reciprocal of the inter-frame interval.  The interval is
        clamped to a minimum of 1 ms so ``fps`` never overflows to infinity
        on back-to-back messages.
        """
        now = time.time()
        if self.last_frame_time is not None:
            elapsed = now - self.last_frame_time
            self.fps = 1.0 / max(elapsed, 0.001)
        self.last_frame_time = now


# Module-level registry of all currently connected sessions
active_sessions: Dict[str, StreamSession] = {}

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

_MIN_FRAME_INTERVAL: float = 1.0 / 30   # hard cap: 30 FPS input
_PROCESS_EVERY_N: int = 2               # process every 2nd frame → ~15 FPS output


@router.websocket("/live")
async def websocket_stream(websocket: WebSocket) -> None:
    """Handle a persistent WebSocket streaming session.

    Accepts binary camera frames, runs the RoadSage pipeline on every
    second frame, and pushes JSON predictions back to the client.

    The engine is retrieved from ``websocket.app.state.engine``.  If the
    engine is ``None`` (still initialising) the connection is accepted but
    immediately closed with a 1013 (Try Again Later) status.

    Args:
        websocket: Starlette WebSocket connection provided by FastAPI.
    """
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session = StreamSession(
        session_id=session_id,
        connected_at=datetime.now(timezone.utc),
    )
    active_sessions[session_id] = session

    logger.info("WebSocket connected: %s", session_id)

    # Resolve engine — reject immediately if not ready
    engine = getattr(websocket.app.state, "engine", None)
    if engine is None:
        await websocket.send_json({
            "type": "error",
            "message": "Inference engine not initialised. Reconnect in a moment.",
        })
        await websocket.close(code=1013)
        active_sessions.pop(session_id, None)
        return

    # Welcome handshake
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "message": "RoadSage streaming ready. Send image frames as binary messages.",
    })

    frame_count: int = 0
    last_process_time: float = 0.0

    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            session.frames_received += 1
            session.update_fps()

            # Hard rate-cap: drop frames that arrive faster than 30 FPS
            now = time.time()
            if (now - last_process_time) < _MIN_FRAME_INTERVAL:
                continue

            # Process every Nth frame only
            frame_count += 1
            if frame_count % _PROCESS_EVERY_N != 0:
                await websocket.send_json({"type": "skipped", "frame": frame_count})
                continue

            # Decode binary payload to BGR image
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                logger.debug(
                    "Session %s frame %d: could not decode binary payload.",
                    session_id,
                    frame_count,
                )
                continue

            # Run inference in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda img=image: engine.predict(img, include_viz=True),
            )

            session.frames_processed += 1
            last_process_time = time.time()

            record_prediction_metrics(result)

            await websocket.send_json({
                "type": "prediction",
                "frame_id": frame_count,
                "session_fps": round(session.fps, 1),
                "result": result.to_dict(),
            })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket session %s error: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        active_sessions.pop(session_id, None)
        logger.info(
            "WebSocket disconnected: %s (%d frames)",
            session_id,
            session.frames_processed,
        )


# ---------------------------------------------------------------------------
# Status endpoint (plain HTTP)
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    summary="Active WebSocket session summary",
    response_description="Connection count and per-session stats",
    tags=["streaming"],
)
async def ws_status() -> JSONResponse:
    """Return a snapshot of all currently connected WebSocket sessions.

    Useful for monitoring dashboards and smoke-testing that the WebSocket
    layer is healthy without opening a WebSocket connection.

    Returns:
        ``200 OK`` with JSON body::

            {
                "active_connections": <int>,
                "sessions": [
                    {
                        "session_id": "<uuid>",
                        "connected_seconds": <int>,
                        "frames_processed": <int>,
                        "fps": <float>
                    },
                    ...
                ]
            }
    """
    now = datetime.now(timezone.utc)
    sessions_snapshot = [
        {
            "session_id": s.session_id,
            "connected_seconds": int(
                (now - s.connected_at.replace(tzinfo=timezone.utc)
                 if s.connected_at.tzinfo is None else now - s.connected_at
                 ).total_seconds()
            ),
            "frames_processed": s.frames_processed,
            "fps": round(s.fps, 1),
        }
        for s in active_sessions.values()
    ]

    return JSONResponse(
        content={
            "active_connections": len(active_sessions),
            "sessions": sessions_snapshot,
        },
        status_code=200,
    )
