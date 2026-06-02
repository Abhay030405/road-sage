"""
api.middleware.rate_limit
==========================

Rate-limiting primitives powered by ``slowapi`` (a Starlette-compatible
wrapper around ``limits``).

This module is intentionally thin — it only instantiates the shared
:data:`limiter` and re-exports the symbols needed by ``api.main`` to wire
everything together.  Actual per-endpoint limits are applied via the
``@limiter.limit(...)`` decorator directly on each route.

Usage in ``api.main``::

    from api.middleware.rate_limit import (
        limiter,
        RateLimitExceeded,
        _rate_limit_exceeded_handler,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Usage on a route::

    from api.middleware.rate_limit import limiter

    @router.post("/predict")
    @limiter.limit("30/minute")
    async def predict(request: Request, ...):
        ...

The key function is ``get_remote_address``, which uses the ``X-Forwarded-For``
header when present (so clients behind a reverse-proxy are rate-limited by
their real IP rather than the proxy IP).
"""

from __future__ import annotations

from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore[import]
from slowapi.errors import RateLimitExceeded  # type: ignore[import]
from slowapi.util import get_remote_address  # type: ignore[import]

limiter: Limiter = Limiter(key_func=get_remote_address)
"""Shared ``slowapi`` limiter instance.

Register it on the FastAPI app with::

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
"""

__all__ = [
    "limiter",
    "RateLimitExceeded",
    "_rate_limit_exceeded_handler",
]
