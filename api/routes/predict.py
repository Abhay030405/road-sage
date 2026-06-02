"""
api.routes.predict
==================

POST /predict — single-frame inference endpoint.
GET  /predict/sample — quick test using a random frame from disk.

The :class:`~roadsage.engine.RoadSageEngine` is accessed via
``request.app.state.engine``, which is populated during application
startup in ``api.main``.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import random

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_upload(contents: bytes, filename: str = "") -> np.ndarray:
    """Decode raw upload bytes to a BGR numpy image.

    Args:
        contents: Raw bytes from the uploaded file.
        filename: Original filename (used only in error messages).

    Returns:
        BGR uint8 array of shape ``(H, W, 3)``.

    Raises:
        HTTPException 400: If the bytes cannot be decoded or the resulting
            array has an unexpected shape.
    """
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image{': ' + filename if filename else ''}.",
        )

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format: expected a non-empty 3-channel (BGR) image.",
        )

    return image


def _get_engine(request: Request):
    """Retrieve the :class:`~roadsage.engine.RoadSageEngine` from app state.

    Args:
        request: Active FastAPI request.

    Returns:
        The engine instance stored in ``app.state.engine``.

    Raises:
        HTTPException 503: When the engine has not been initialised yet.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialised. The server may still be starting up.",
        )
    return engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    summary="Single-frame driving-command inference",
    response_description="PredictionResult as JSON",
    tags=["inference"],
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG camera frame"),
    include_viz: bool = Query(False, description="Include base64 lane-overlay JPEG in response"),
    include_gradcam: bool = Query(False, description="Include base64 GradCAM JPEG (throttled)"),
) -> JSONResponse:
    """Run the full RoadSage pipeline on a single uploaded camera frame.

    The CPU-bound inference call is dispatched to a thread-pool executor so
    the async event loop is never blocked.

    Content-type must be ``image/jpeg`` or ``image/png``.

    Args:
        request: FastAPI request — provides ``app.state.engine``.
        file: Uploaded image file.
        include_viz: When ``True``, the ``lane_viz_base64`` field of the
            response is populated with a base64-encoded annotated frame.
        include_gradcam: When ``True``, the ``gradcam_base64`` field is
            populated every *N* frames (set by ``GradCAMManager``).

    Returns:
        ``200 OK`` with a JSON body matching
        :meth:`~roadsage.engine.PredictionResult.to_dict`.

    Raises:
        HTTPException 400: Invalid content-type, undecodable image, or wrong
            channel count.
        HTTPException 503: Engine not yet initialised.
    """
    # Validate content type
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG or PNG).",
        )

    contents = await file.read()
    image = _decode_upload(contents, filename=file.filename or "")

    engine = _get_engine(request)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.predict(image, include_viz=(include_viz or include_gradcam)),
    )

    logger.info(
        "Predict: %s conf=%.2f %dms",
        result.command,
        result.confidence,
        result.latency_ms["total"],
    )

    return JSONResponse(content=result.to_dict(), status_code=200)


@router.get(
    "/predict/sample",
    summary="Test inference on a random disk frame",
    response_description="PredictionResult as JSON",
    tags=["inference"],
)
async def predict_sample(
    request: Request,
    include_viz: bool = Query(True, description="Include lane-overlay visualization"),
) -> JSONResponse:
    """Run inference on a random frame from the ``rgb/`` folder.

    Useful for quickly validating the API without uploading a file.
    Visualization is enabled by default so the response is immediately
    inspectable in the Swagger UI.

    Args:
        request: FastAPI request — provides ``app.state.engine``.
        include_viz: Include annotated lane-overlay in the response.

    Returns:
        ``200 OK`` with a JSON body matching
        :meth:`~roadsage.engine.PredictionResult.to_dict`.

    Raises:
        HTTPException 404: No sample images found in ``rgb/``.
        HTTPException 503: Engine not yet initialised.
    """
    engine = _get_engine(request)

    candidates = glob.glob("rgb/rgb_image_*.png")
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No sample images found in rgb/. Upload an image to /predict instead.",
        )

    img_path = random.choice(candidates)
    image = cv2.imread(img_path)
    if image is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read sample image: {img_path}",
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.predict(image, include_viz=include_viz),
    )

    logger.info(
        "Sample predict (%s): %s conf=%.2f %dms",
        img_path,
        result.command,
        result.confidence,
        result.latency_ms["total"],
    )

    return JSONResponse(content=result.to_dict(), status_code=200)
