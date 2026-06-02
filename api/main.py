"""
api.main
=========

RoadSage FastAPI application — Phase 5 (full inference stack).

Startup sequence
----------------
1. Configure structured JSON logging.
2. Load and validate ``configs/production.yaml`` (or ``$CONFIG_PATH``).
3. Initialise :class:`~roadsage.engine.RoadSageEngine` (loads all model weights).
4. Mount Prometheus metrics sub-app at ``/metrics``.
5. Add CORS, request-logging, and rate-limiting middleware.
6. Include route groups for inference, health, batch, and WebSocket streaming.

Start with::

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from app.utils.config_validator import load_and_validate_config
from app.engine import RoadSageEngine

# Route routers
from api.routes.predict import router as predict_router
from api.routes.health import router as health_router
from api.routes.batch import router as batch_router
from api.websocket.stream import router as ws_router

# Middleware
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.rate_limit import (
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
    limiter,
)

# ---------------------------------------------------------------------------
# Prometheus metrics (module-level singletons — must be created once)
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "roadsage_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "roadsage_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
)
COMMAND_COUNT = Counter(
    "roadsage_command_total",
    "Driving commands predicted",
    ["command"],
)
SAFETY_GATE_COUNT = Counter(
    "roadsage_safety_gate_triggers_total",
    "Safety gate trigger count",
)
LANE_DETECTION_FAILURES = Counter(
    "roadsage_lane_detection_failures_total",
    "Frames with no lanes detected",
)
ML_FALLBACK_COUNT = Counter(
    "roadsage_ml_fallback_activations_total",
    "ML fallback activations",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("roadsage")

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup / shutdown tasks around the server lifetime.

    Startup:

    1. Load and validate ``configs/production.yaml``.
    2. Initialise the :class:`~roadsage.engine.RoadSageEngine`; store on
       ``app.state.engine``.
    3. Record ``app.state.start_time`` (float, seconds since epoch) for
       uptime calculations.

    Shutdown:

    * Logs a clean shutdown message.

    Any exception during engine initialisation is caught and logged —
    the API still starts so health and metadata endpoints remain reachable.
    """
    logger.info("RoadSage API starting up...")

    config_path = os.getenv("CONFIG_PATH", "configs/production.yaml")

    # Config
    try:
        config = load_and_validate_config(config_path)
        app.state.config = config
        env = config.get("project", {}).get("environment", "production")
        logger.info("Config loaded from '%s' (env=%s).", config_path, env)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Config load failed: %s — using empty config.", exc)
        app.state.config = {}

    app.state.start_time = datetime.now(timezone.utc)

    # Engine
    try:
        app.state.engine = RoadSageEngine(config_path)
        app.state.models_loaded = True
        logger.info("Engine initialized. API ready.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialise RoadSageEngine: %s", exc)
        app.state.engine = None
        app.state.models_loaded = False

    yield

    logger.info("RoadSage API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RoadSage API",
    description=(
        "Vision-Based Lane Understanding & Intelligent Driving Decision Engine. "
        "POST a camera frame to ``/api/v1/predict`` to receive a driving command "
        "with confidence score, lane geometry, hazard status, and optional "
        "GradCAM / lane-overlay visualizations.  Connect to ``/ws/live`` for "
        "low-latency streaming inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Safe defaults so unit tests that skip the lifespan never raise AttributeError
app.state.engine = None
app.state.models_loaded = False
app.state.config = {}
app.state.start_time = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Prometheus metrics sub-app
#
# Mounted before middleware so the /metrics ASGI sub-app is not
# double-counted by the Prometheus middleware itself.
# ---------------------------------------------------------------------------

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ---------------------------------------------------------------------------
# Middleware  (added innermost-first — last added = outermost wrapper)
# CORSMiddleware must be outermost so pre-flight OPTIONS requests are
# handled before any other middleware runs, including logging.
# ---------------------------------------------------------------------------

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

_API_PREFIX = "/api/v1"

app.include_router(predict_router, prefix=_API_PREFIX, tags=["inference"])
app.include_router(health_router,  prefix=_API_PREFIX, tags=["health"])
app.include_router(batch_router,   prefix=_API_PREFIX, tags=["batch"])
app.include_router(ws_router,      prefix="/ws",       tags=["streaming"])

# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


@app.get("/", tags=["meta"])
async def root():
    """Return service metadata and quick-start endpoint links.

    Returns:
        JSON with ``service``, ``version``, and links to ``/docs``,
        ``/api/v1/health``, ``/api/v1/predict``, and ``/ws/live``.
    """
    return {
        "service": "RoadSage API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
        "stream": "/ws/live",
    }


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return a structured 500 response.

    Args:
        request: The originating FastAPI request.
        exc: The unhandled exception.

    Returns:
        ``500 Internal Server Error`` JSON with ``error`` and ``type`` fields.
    """
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )
