"""Shared pytest fixtures for the RoadSage test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from app.decision.safety_gate import SceneContext
from app.lane_detection.lane_geometry import LaneGeometry

_PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_config() -> dict:
    """Load and return the production YAML config as a plain dict.

    Uses an absolute path derived from this file's location so the fixture
    works regardless of the directory pytest is invoked from.
    """
    config_path = _PROJECT_ROOT / "configs" / "production.yaml"
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a blank 720 × 1280 × 3 BGR frame.

    Simulates a dark road image with no content — suitable as a safe default
    for any test that needs a valid image array without specific pixel values.
    """
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_lane_geometry() -> LaneGeometry:
    """Return a neutral :class:`LaneGeometry` instance.

    Represents a vehicle centred in its lane on a very gently curving road —
    below both the offset threshold (0.3) and curve threshold (0.005), so the
    expected geometric decision is ``FORWARD``.
    """
    return LaneGeometry(
        offset=0.0,
        curvature=0.001,
        left_lane_detected=True,
        right_lane_detected=True,
        left_x=400.0,
        right_x=880.0,
        confidence=0.95,
    )


@pytest.fixture
def sample_scene_context() -> SceneContext:
    """Return a clear-scene :class:`SceneContext` with no hazards or obstacles.

    Confidence is 1.0 — well above the min_confidence threshold (0.60) — so
    the safety gate should not trigger a STOP for this context.
    """
    return SceneContext(
        immediate_hazard=False,
        obstacle_detected=False,
        obstacle_distance=float("inf"),
        confidence=1.0,
    )
