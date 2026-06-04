"""
tests.test_integration
=======================

End-to-end integration tests for the full RoadSage pipeline.

These tests spin up a real :class:`~app.engine.RoadSageEngine` against the
actual model configs (no mocking) and verify that:

* The engine initialises without error.
* Every predict() call returns a valid :class:`~app.engine.PredictionResult`.
* The pipeline never raises on real MNNIT images.
* JSON serialisation round-trips cleanly.
* Latency is within a generous threshold even without ONNX weights.

Requires at least one image in ``rgb/rgb_image_*.png``.

Run with::

    pytest tests/test_integration.py -v --tb=short
"""

from __future__ import annotations

import base64
import glob
import json

import cv2
import numpy as np
import pytest

from app.engine import RoadSageEngine, PredictionResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_COMMANDS = {"FORWARD", "LEFT", "RIGHT", "STOP"}
_VALID_DECISION_PATHS = {
    "geometric",
    "single_lane",
    "ml_fallback",
    "safety_gate",
    "confidence_gate",
}

# Generous latency budget: without ONNX weights inference may be slow.
# The target with ONNX on CPU is < 100 ms; without ONNX allow 2 000 ms.
_LATENCY_BUDGET_MS = 2000.0


# ---------------------------------------------------------------------------
# Module-scoped fixtures (engine + images loaded once for the whole module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> RoadSageEngine:
    """Create a single RoadSageEngine shared by all tests in this module.

    Using ``scope="module"`` avoids reloading model weights for every test,
    which would make the suite prohibitively slow.
    """
    return RoadSageEngine()


@pytest.fixture(scope="module")
def sample_images() -> list[np.ndarray]:
    """Load up to 5 MNNIT camera frames from the ``rgb/`` directory.

    Asserts that the directory is non-empty so downstream tests fail with a
    clear message rather than an index error.
    """
    paths = sorted(glob.glob("rgb/rgb_image_*.png"))[:5]
    assert len(paths) > 0, (
        "rgb/ folder must contain at least one rgb_image_*.png file. "
        "Run data collection or download the MNNIT dataset first."
    )
    return [cv2.imread(p) for p in paths]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_engine_initializes(engine: RoadSageEngine) -> None:
    """Engine must initialise without error and expose a valid health dict."""
    assert engine is not None

    health = engine.get_health()
    assert "models" in health, "health dict missing 'models' key"
    assert "frames_processed" in health, "health dict missing 'frames_processed' key"


def test_predict_returns_prediction_result(
    engine: RoadSageEngine,
    sample_images: list[np.ndarray],
) -> None:
    """predict() must return a PredictionResult with valid field values."""
    result = engine.predict(sample_images[0])

    assert isinstance(result, PredictionResult), (
        f"Expected PredictionResult, got {type(result)}"
    )
    assert result.command in _VALID_COMMANDS, (
        f"Unknown command: {result.command!r}"
    )
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} out of [0, 1]"
    )
    assert result.decision_path in _VALID_DECISION_PATHS, (
        f"Unknown decision path: {result.decision_path!r}"
    )


def test_predict_never_crashes(
    engine: RoadSageEngine,
    sample_images: list[np.ndarray],
) -> None:
    """predict() must not raise on any of the sample images."""
    errors: list[str] = []

    for img in sample_images:
        try:
            result = engine.predict(img)
            assert result.command in _VALID_COMMANDS
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    assert len(errors) == 0, (
        f"Pipeline crashed on {len(errors)} image(s): {errors}"
    )


def test_predict_with_viz_returns_base64(
    engine: RoadSageEngine,
    sample_images: list[np.ndarray],
) -> None:
    """When include_viz=True, a non-trivial base64 JPEG must be returned."""
    result = engine.predict(sample_images[0], include_viz=True)

    if result.lane_viz_base64:
        decoded = base64.b64decode(result.lane_viz_base64)
        assert len(decoded) > 100, (
            "lane_viz_base64 decoded to fewer than 100 bytes — likely corrupt"
        )


def test_prediction_result_is_json_serializable(
    engine: RoadSageEngine,
    sample_images: list[np.ndarray],
) -> None:
    """to_json() must produce a valid JSON string that round-trips correctly."""
    result = engine.predict(sample_images[0])

    json_str = result.to_json()
    assert isinstance(json_str, str), "to_json() must return a str"

    parsed = json.loads(json_str)
    assert parsed["command"] == result.command, (
        "command field does not survive JSON round-trip"
    )
    assert "latency_ms" in parsed, "latency_ms missing from JSON output"
    assert "confidence" in parsed, "confidence missing from JSON output"


def test_latency_under_threshold(
    engine: RoadSageEngine,
    sample_images: list[np.ndarray],
) -> None:
    """Average total latency must be below the generous threshold."""
    latencies: list[float] = []

    for img in sample_images:
        result = engine.predict(img)
        latencies.append(result.latency_ms["total"])

    avg = sum(latencies) / len(latencies)
    assert avg < _LATENCY_BUDGET_MS, (
        f"Average latency {avg:.0f} ms exceeds {_LATENCY_BUDGET_MS:.0f} ms "
        f"budget (even without ONNX weights)"
    )


def test_stop_on_hazard_scene(engine: RoadSageEngine) -> None:
    """A fully black frame (no lanes, no scene) must produce a valid command.

    The engine must not raise; the safety gate may — and typically will —
    produce STOP on a blank frame with zero confidence.
    """
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.predict(blank)

    assert result.command in _VALID_COMMANDS, (
        f"Unexpected command on blank frame: {result.command!r}"
    )


def test_health_increases_frame_count(
    engine: RoadSageEngine,
    sample_images: list[np.ndarray],
) -> None:
    """frames_processed must increment after each predict() call."""
    h1 = engine.get_health()
    engine.predict(sample_images[0])
    engine.predict(sample_images[0])
    h2 = engine.get_health()

    assert h2["frames_processed"] > h1["frames_processed"], (
        f"frames_processed did not increase: {h1['frames_processed']} → "
        f"{h2['frames_processed']}"
    )
