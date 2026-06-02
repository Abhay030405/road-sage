"""
api.middleware.logging
=======================

Structured JSON request/response logging middleware.

Each HTTP transaction is logged as a single-line JSON object containing
the method, path, status code, duration, and client IP.  Log records are
emitted at INFO level on the ``roadsage.api`` logger so they can be
aggregated by any standard log collector (Loki, CloudWatch, ELK, etc.).

Usage in ``api.main``::

    from api.middleware.logging import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)
"""

from __future__ import annotations

import json
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_api_logger = logging.getLogger("roadsage.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit a structured JSON log line for every HTTP request.

    The log entry is written **after** the response has been sent so that
    the ``status_code`` and ``duration_ms`` are both available.  It is
    intentionally lightweight — no body buffering is performed.

    Log format (one JSON object per line)::

        {
            "method": "POST",
            "path": "/api/v1/predict",
            "status_code": 200,
            "duration_ms": 142.37,
            "client_ip": "127.0.0.1"
        }
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request, then emit a structured log entry.

        Args:
            request: Incoming Starlette request.
            call_next: ASGI callable for the next middleware or route handler.

        Returns:
            The downstream response, unmodified.
        """
        start = time.time()
        response: Response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        log_entry = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": request.client.host if request.client else "unknown",
        }
        _api_logger.info(json.dumps(log_entry))

        return response
