"""
api.routes.health
=================

GET /health — liveness / readiness probe with full engine diagnostics.

Returns model readiness, rolling latency percentiles, a warmup prediction
result, API version, and uptime.  Designed to be called by load balancers,
Docker healthchecks, and monitoring dashboards.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_API_VERSION = "1.0.0"


@router.get(
    "/health",
    summary="Liveness and readiness probe",
    response_description="Engine health JSON",
    tags=["meta"],
)
async def health_check(request: Request) -> JSONResponse:
    """Return a comprehensive health summary for the RoadSage API.

    Runs a warmup prediction on a blank 480×640 frame so the response
    includes end-to-end pipeline latency even when no real traffic has
    been processed yet.

    The endpoint always returns HTTP 200.  Callers should inspect the
    ``models`` sub-object and ``warmup_command`` to determine whether
    inference is actually functional.

    Args:
        request: FastAPI request — provides ``app.state.engine``,
            ``app.state.start_time``, and ``app.state.config``.

    Returns:
        ``200 OK`` with a JSON body containing:

        - ``status`` — ``"healthy"``
        - ``api_version`` — ``"1.0.0"``
        - ``environment`` — value from the loaded config
        - ``uptime_seconds`` — seconds since API startup
        - ``models`` — per-module readiness flags
        - ``latency_p50_ms`` / ``latency_p95_ms`` — rolling percentiles
        - ``frames_processed`` — total frames seen by the engine
        - ``warmup_command`` — command returned for a blank frame
        - ``warmup_latency_ms`` — total pipeline time for the warmup frame
    """
    engine = getattr(request.app.state, "engine", None)
    start_time: datetime.datetime = getattr(
        request.app.state,
        "start_time",
        datetime.datetime.now(datetime.timezone.utc),
    )
    config: dict = getattr(request.app.state, "config", {})

    now = datetime.datetime.now(datetime.timezone.utc)
    # Ensure start_time is timezone-aware for subtraction
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
    uptime_seconds = (now - start_time).total_seconds()
    environment = config.get("project", {}).get("environment", "production")

    if engine is None:
        return JSONResponse(
            content={
                "status": "starting",
                "api_version": _API_VERSION,
                "environment": environment,
                "uptime_seconds": round(uptime_seconds, 2),
                "models": {},
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "frames_processed": 0,
                "warmup_command": None,
                "warmup_latency_ms": 0.0,
            },
            status_code=200,
        )

    health: dict = engine.get_health()
    health["api_version"] = _API_VERSION
    health["environment"] = environment
    health["uptime_seconds"] = round(uptime_seconds, 2)

    # Warmup prediction on a blank frame to verify the full pipeline
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    warmup_result = engine.predict(blank)
    health["warmup_command"] = warmup_result.command
    health["warmup_latency_ms"] = round(warmup_result.latency_ms["total"], 2)

    logger.debug(
        "Health check: status=%s warmup=%s %.1fms",
        health["status"],
        warmup_result.command,
        warmup_result.latency_ms["total"],
    )

    return JSONResponse(content=health, status_code=200)
