"""
api.routes.batch
================

POST /batch — concurrent multi-image inference endpoint.

Accepts up to 20 images per request and runs predictions concurrently
behind an asyncio semaphore (max 4 in-flight at once) to balance
throughput against per-request latency.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_BATCH_SIZE = 20
_MAX_CONCURRENCY = 4
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post(
    "/batch",
    summary="Batch inference on multiple frames",
    response_description="List of PredictionResult JSON objects",
    tags=["inference"],
)
async def batch_predict(
    request: Request,
    files: List[UploadFile] = File(..., description="Up to 20 JPEG or PNG frames"),
    include_viz: bool = Query(False, description="Include lane-overlay base64 for each frame"),
) -> JSONResponse:
    """Run RoadSage inference on a batch of uploaded frames concurrently.

    Predictions are executed in a thread-pool executor behind a semaphore
    capped at ``_MAX_CONCURRENCY`` (4) simultaneous jobs, preventing
    memory spikes on large batches while still fully utilising available
    CPU cores.

    Args:
        request: FastAPI request — provides ``app.state.engine``.
        files: List of uploaded image files (max 20).
        include_viz: When ``True``, each result includes
            ``lane_viz_base64``.

    Returns:
        ``200 OK`` with JSON body::

            {
                "total": <int>,
                "results": [{"filename": str, "result": PredictionResult}, ...],
                "command_distribution": {"FORWARD": n, ...}
            }

    Raises:
        HTTPException 400: More than 20 files submitted, or an image cannot
            be decoded.
        HTTPException 503: Engine not yet initialised.
    """
    # Guard: batch size cap
    if len(files) > _MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Max {_MAX_BATCH_SIZE} images per batch request.",
        )

    engine = _get_engine(request)

    # Decode all uploads eagerly (fast; avoids holding upload streams open)
    images: List[np.ndarray] = []
    filenames: List[str] = []

    for upload in files:
        contents = await upload.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        name = upload.filename or f"file_{len(filenames)}"

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode image: {name}",
            )

        images.append(image)
        filenames.append(name)

    # Concurrent inference behind semaphore
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def predict_one(img: np.ndarray, filename: str) -> dict:
        """Run a single prediction inside the semaphore guard."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: engine.predict(img, include_viz=include_viz),
            )
            return {"filename": filename, "result": result.to_dict()}

    tasks = [predict_one(img, name) for img, name in zip(images, filenames)]
    results = await asyncio.gather(*tasks)

    commands = [r["result"]["command"] for r in results]
    distribution = dict(Counter(commands))

    logger.info(
        "Batch complete: %d frames | distribution=%s",
        len(results),
        distribution,
    )

    return JSONResponse(
        content={
            "total": len(results),
            "results": list(results),
            "command_distribution": distribution,
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _get_engine(request: Request):
    """Retrieve the engine from app state, raising 503 if not yet loaded.

    Args:
        request: Active FastAPI request.

    Returns:
        The :class:`~roadsage.engine.RoadSageEngine` instance.

    Raises:
        HTTPException 503: When ``app.state.engine`` is not set.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialised. The server may still be starting up.",
        )
    return engine
