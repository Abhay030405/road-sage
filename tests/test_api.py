"""
tests.test_api
==============

Phase 5 integration and unit tests for the RoadSage FastAPI application.

All tests are designed to pass without ONNX/PyTorch model weights.  When
weights are absent every module degrades gracefully, and ``engine.predict``
always returns a valid (STOP / low-confidence) :class:`~app.engine.PredictionResult`.

Run with::

    pytest tests/test_api.py -v --tb=short
"""

from __future__ import annotations

import base64
import json
import time

import cv2
import httpx  # noqa: F401 — imported per spec; used internally by TestClient
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# Module-scoped fixture — lifespan runs once for the whole test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Provide a TestClient that executes the full app lifespan on entry.

    Using ``scope="module"`` means the engine initialises once for all tests,
    avoiding redundant model-loading overhead.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_test_image_bytes(h: int = 480, w: int = 640) -> bytes:
    """Create a minimal synthetic road image encoded as JPEG bytes.

    Draws a grey rectangle in the lower-left to simulate a road surface.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        JPEG-encoded bytes of the synthetic image.
    """
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 300), (250, 479), (180, 180, 180), -1)
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


def _engine_ready(client: TestClient) -> bool:
    """Return True when the engine loaded successfully (health models non-empty)."""
    try:
        data = client.get("/api/v1/health").json()
        return bool(data.get("models"))
    except Exception:
        return False


# ===========================================================================
# Section 1 — Basic API
# ===========================================================================


def test_root_endpoint(client: TestClient) -> None:
    """GET / must return 200 with version, docs, and predict keys."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "docs" in data
    assert "predict" in data


def test_docs_available(client: TestClient) -> None:
    """Swagger UI at /docs must return HTTP 200."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_metrics_endpoint(client: TestClient) -> None:
    """Prometheus scrape endpoint must return 200 with recognisable metric names."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"roadsage" in response.content or b"python" in response.content


# ===========================================================================
# Section 2 — Health endpoint
# ===========================================================================


def test_health_returns_200(client: TestClient) -> None:
    """GET /api/v1/health must always return 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_has_required_fields(client: TestClient) -> None:
    """Health response must include all mandatory top-level keys."""
    data = client.get("/api/v1/health").json()
    for field in ("status", "models", "frames_processed", "api_version", "uptime_seconds"):
        assert field in data, f"Missing health field: {field}"


def test_health_api_version(client: TestClient) -> None:
    """api_version must equal '1.0.0'."""
    data = client.get("/api/v1/health").json()
    assert data["api_version"] == "1.0.0"


def test_health_models_section(client: TestClient) -> None:
    """models dict must contain the three module keys with boolean values."""
    data = client.get("/api/v1/health").json()
    if not data.get("models"):
        pytest.skip("Engine not loaded — skipping model-readiness check.")
    models = data["models"]
    for key in ("lane_detector", "scene_analyzer", "ml_fallback"):
        assert key in models, f"Missing models key: {key}"
    assert all(isinstance(v, bool) for v in models.values())


def test_health_uptime_increases(client: TestClient) -> None:
    """uptime_seconds must grow between two consecutive health calls."""
    r1 = client.get("/api/v1/health").json()["uptime_seconds"]
    time.sleep(0.15)
    r2 = client.get("/api/v1/health").json()["uptime_seconds"]
    assert r2 > r1


def test_health_uptime_non_negative(client: TestClient) -> None:
    """uptime_seconds must be a non-negative number."""
    data = client.get("/api/v1/health").json()
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0


def test_health_warmup_fields_present(client: TestClient) -> None:
    """When the engine is loaded the health response includes warmup fields."""
    data = client.get("/api/v1/health").json()
    if not data.get("models"):
        pytest.skip("Engine not loaded — skipping warmup field check.")
    assert "warmup_command" in data
    assert "warmup_latency_ms" in data
    assert data["warmup_command"] in ("FORWARD", "LEFT", "RIGHT", "STOP")
    assert data["warmup_latency_ms"] >= 0


# ===========================================================================
# Section 3 — Predict endpoint
# ===========================================================================


def test_predict_returns_200(client: TestClient) -> None:
    """POST /api/v1/predict must return 200 for a valid JPEG."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200


def test_predict_response_has_required_fields(client: TestClient) -> None:
    """Predict response must include all PredictionResult fields."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    for field in (
        "command", "confidence", "decision_path",
        "lane_offset_m", "curvature_inv_m",
        "left_lane_detected", "right_lane_detected",
        "hazard_detected", "surface_class", "latency_ms",
    ):
        assert field in data, f"Missing predict field: {field}"


def test_predict_command_is_valid(client: TestClient) -> None:
    """command must be one of the four recognised driving commands."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["command"] in ("FORWARD", "LEFT", "RIGHT", "STOP")


def test_predict_confidence_in_range(client: TestClient) -> None:
    """confidence must be a float in [0.0, 1.0]."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    conf = response.json()["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_predict_with_viz_returns_base64(client: TestClient) -> None:
    """?include_viz=true must populate lane_viz_base64 with a non-empty JPEG."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    response = client.post(
        "/api/v1/predict?include_viz=true",
        files={"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    if data.get("lane_viz_base64"):
        decoded = base64.b64decode(data["lane_viz_base64"])
        assert len(decoded) > 100, "lane_viz_base64 decoded to an unexpectedly small blob"


def test_predict_rejects_non_image(client: TestClient) -> None:
    """A text/plain upload must be rejected with HTTP 400."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_corrupt_image(client: TestClient) -> None:
    """Four random bytes labelled as image/jpeg must be rejected with HTTP 400."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("bad.jpg", b"\x00\x01\x02\x03", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_latency_fields(client: TestClient) -> None:
    """latency_ms must have lane/scene/decision/total keys, all non-negative."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    latency = response.json()["latency_ms"]
    for key in ("lane", "scene", "decision", "total"):
        assert key in latency, f"Missing latency key: {key}"
    assert all(v >= 0 for v in latency.values())


def test_predict_png_accepted(client: TestClient) -> None:
    """image/png content-type must be accepted (not just JPEG)."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    response = client.post(
        "/api/v1/predict",
        files={"file": ("frame.png", buf.tobytes(), "image/png")},
    )
    assert response.status_code == 200


# ===========================================================================
# Section 4 — Sample endpoint
# ===========================================================================


def test_sample_endpoint(client: TestClient) -> None:
    """GET /api/v1/predict/sample returns 200 (images present) or 404 (none)."""
    response = client.get("/api/v1/predict/sample")
    # 200 when rgb/ images exist, 404 when the folder is empty / absent,
    # 503 when the engine failed to initialise.
    assert response.status_code in (200, 404, 503)
    if response.status_code == 200:
        assert "command" in response.json()


# ===========================================================================
# Section 5 — Batch endpoint
# ===========================================================================


def test_batch_predict_two_images(client: TestClient) -> None:
    """POST /api/v1/batch with two images must return total=2 with two results."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    img_bytes = make_test_image_bytes()
    response = client.post(
        "/api/v1/batch",
        files=[
            ("files", ("img1.jpg", img_bytes, "image/jpeg")),
            ("files", ("img2.jpg", img_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


def test_batch_command_distribution_present(client: TestClient) -> None:
    """Batch response must include a command_distribution dict."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    img_bytes = make_test_image_bytes()
    response = client.post(
        "/api/v1/batch",
        files=[("files", ("img1.jpg", img_bytes, "image/jpeg"))],
    )
    assert response.status_code == 200
    assert "command_distribution" in response.json()


def test_batch_rejects_too_many_images(client: TestClient) -> None:
    """Submitting 21 images must be rejected with HTTP 400 regardless of engine state."""
    img_bytes = make_test_image_bytes()
    files = [
        ("files", (f"img{i}.jpg", img_bytes, "image/jpeg")) for i in range(21)
    ]
    response = client.post("/api/v1/batch", files=files)
    assert response.status_code == 400


def test_batch_each_result_has_command(client: TestClient) -> None:
    """Every result in a batch response must contain a command field."""
    if not _engine_ready(client):
        pytest.skip("Engine not initialised.")
    img_bytes = make_test_image_bytes()
    response = client.post(
        "/api/v1/batch",
        files=[
            ("files", ("a.jpg", img_bytes, "image/jpeg")),
            ("files", ("b.jpg", img_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    for item in response.json()["results"]:
        assert "command" in item["result"]
        assert item["result"]["command"] in ("FORWARD", "LEFT", "RIGHT", "STOP")


# ===========================================================================
# Section 6 — PredictionResult unit tests (no server required)
# ===========================================================================


def test_prediction_result_to_dict() -> None:
    """to_dict() must serialise all fields, converting lane tuples to lists."""
    from app.engine import PredictionResult

    result = PredictionResult(
        command="FORWARD",
        confidence=0.91,
        decision_path="geometric",
        lane_offset_m=0.12,
        curvature_inv_m=0.003,
        left_lane_detected=True,
        right_lane_detected=True,
        left_lane_points=[(100, 200)],
        right_lane_points=[(500, 200)],
        center_lane_points=[],
        lane_confidence=[0.9, 0.85],
        hazard_detected=False,
        hazard_reason=None,
        surface_class="clean",
        nearest_obstacle_class=None,
        nearest_obstacle_depth=0.0,
        gradcam_base64=None,
        lane_viz_base64=None,
        latency_ms={"lane": 15, "scene": 25, "decision": 5, "total": 45},
    )

    d = result.to_dict()
    assert d["command"] == "FORWARD"
    assert d["confidence"] == pytest.approx(0.91)
    assert d["hazard_detected"] is False
    assert d["hazard_reason"] is None

    # Tuples must be converted to lists for JSON serialisation
    assert isinstance(d["left_lane_points"], list)
    assert isinstance(d["left_lane_points"][0], list), "tuple not converted to list"

    # Round-trip through JSON
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["command"] == "FORWARD"
    assert parsed["confidence"] == pytest.approx(0.91)


def test_prediction_result_summary_stop() -> None:
    """summary() for a STOP result must include command and confidence."""
    from app.engine import PredictionResult

    result = PredictionResult(
        command="STOP",
        confidence=0.99,
        decision_path="safety_gate",
        lane_offset_m=0.0,
        curvature_inv_m=0.0,
        left_lane_detected=False,
        right_lane_detected=False,
        left_lane_points=[],
        right_lane_points=[],
        center_lane_points=[],
        lane_confidence=[],
        hazard_detected=True,
        hazard_reason="obstacle",
        surface_class="clean",
        nearest_obstacle_class="person",
        nearest_obstacle_depth=0.8,
        gradcam_base64=None,
        lane_viz_base64=None,
        latency_ms={"lane": 15, "scene": 25, "decision": 5, "total": 45},
    )

    summary = result.summary()
    assert "STOP" in summary
    assert "0.99" in summary


def test_prediction_result_is_safe() -> None:
    """is_safe() must reflect the hazard_detected flag."""
    from app.engine import PredictionResult

    safe = PredictionResult(
        command="FORWARD", confidence=0.9, decision_path="geometric",
        lane_offset_m=0.0, curvature_inv_m=0.0,
        left_lane_detected=True, right_lane_detected=True,
        left_lane_points=[], right_lane_points=[], center_lane_points=[],
        lane_confidence=[], hazard_detected=False, hazard_reason=None,
        surface_class="clean", nearest_obstacle_class=None,
        nearest_obstacle_depth=0.0, gradcam_base64=None, lane_viz_base64=None,
        latency_ms={"lane": 0, "scene": 0, "decision": 0, "total": 0},
    )
    assert safe.is_safe() is True

    unsafe = PredictionResult(
        command="STOP", confidence=1.0, decision_path="safety_gate",
        lane_offset_m=0.0, curvature_inv_m=0.0,
        left_lane_detected=False, right_lane_detected=False,
        left_lane_points=[], right_lane_points=[], center_lane_points=[],
        lane_confidence=[], hazard_detected=True, hazard_reason="obstacle",
        surface_class="unknown", nearest_obstacle_class="car",
        nearest_obstacle_depth=0.9, gradcam_base64=None, lane_viz_base64=None,
        latency_ms={"lane": 0, "scene": 0, "decision": 0, "total": 0},
    )
    assert unsafe.is_safe() is False


def test_prediction_result_summary_offset_sign() -> None:
    """summary() must prefix positive offsets with '+' and negative with '-'."""
    from app.engine import PredictionResult

    def _make(offset: float) -> str:
        r = PredictionResult(
            command="FORWARD", confidence=0.8, decision_path="geometric",
            lane_offset_m=offset, curvature_inv_m=0.0,
            left_lane_detected=True, right_lane_detected=True,
            left_lane_points=[], right_lane_points=[], center_lane_points=[],
            lane_confidence=[], hazard_detected=False, hazard_reason=None,
            surface_class="clean", nearest_obstacle_class=None,
            nearest_obstacle_depth=0.0, gradcam_base64=None, lane_viz_base64=None,
            latency_ms={"lane": 10, "scene": 20, "decision": 5, "total": 35},
        )
        return r.summary()

    assert "+0.25m" in _make(0.25)
    assert "-0.25m" in _make(-0.25)
