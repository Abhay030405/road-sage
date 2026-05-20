"""RoadSage API — Phase 1 skeleton.

Phase 1 scope: config validation, health endpoint, Prometheus metrics,
structured logging.  Inference endpoints are Phase 5.

Start with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import time
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.config_validator import ConfigValidationError, load_and_validate_config

# ---------------------------------------------------------------------------
# Prometheus metrics (module-level singletons)
# ---------------------------------------------------------------------------

REQUEST_COUNTER = Counter(
    "roadsage_http_requests_total",
    "Total number of HTTP requests received",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "roadsage_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _setup_logging() -> None:
    """Configure root logger with JSON output at the level set by LOG_LEVEL."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App-wide state (mutated during lifespan startup)
# ---------------------------------------------------------------------------

_START_TIME: datetime.datetime = datetime.datetime.utcnow()
_CONFIG_VALID: bool = False
_ENVIRONMENT: str = "production"

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks, yield control to the server, then clean up."""
    global _CONFIG_VALID, _ENVIRONMENT

    _setup_logging()
    log.info("RoadSage API starting up...")

    config_path = os.getenv("CONFIG_PATH", "configs/production.yaml")
    try:
        config = load_and_validate_config(config_path)
        _CONFIG_VALID = True
        _ENVIRONMENT = config.get("project", {}).get("environment", "production")
        log.info("Config loaded and validated from '%s'.", config_path)
    except ConfigValidationError as exc:
        log.warning("Config validation failed: %s", exc)
        _CONFIG_VALID = False
    except Exception as exc:  # noqa: BLE001
        log.warning("Unexpected error loading config: %s", exc)
        _CONFIG_VALID = False

    # Phase 1: models are not loaded yet.
    app.state.models_loaded = False
    log.info("Phase 1: model loading deferred to Phase 5.")

    yield

    log.info("RoadSage API shut down cleanly.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RoadSage API",
    description="Vision-Based Lane Understanding and Intelligent Driving Decision Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Default state so the health endpoint never raises AttributeError when the
# lifespan hasn't run (e.g. during unit tests without a context manager).
app.state.models_loaded = False

# Mount Prometheus metrics scrape endpoint before middleware so the metrics
# ASGI app is a proper sub-mount and not double-counted.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ---------------------------------------------------------------------------
# Middleware  (added in reverse wrapping order: last added = outermost)
# ---------------------------------------------------------------------------


class _PrometheusMiddleware(BaseHTTPMiddleware):
    """Record per-request counters and latency histograms."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)

        REQUEST_COUNTER.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response


# Inner middleware first, outer last.
app.add_middleware(_PrometheusMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["meta"])
async def root():
    """Landing redirect — points clients to /docs."""
    return {"message": "RoadSage API is running. See /docs for API documentation."}


@app.get("/api/v1/health", tags=["meta"])
async def health():
    """Liveness / readiness probe used by Docker healthcheck and load balancers.

    Always returns HTTP 200 in Phase 1 even when models are not loaded,
    so the container is considered healthy as soon as the server is up.
    """
    uptime = (datetime.datetime.utcnow() - _START_TIME).total_seconds()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": app.state.models_loaded,
        "config_valid": _CONFIG_VALID,
        "uptime_seconds": round(uptime, 2),
        "environment": _ENVIRONMENT,
    }
