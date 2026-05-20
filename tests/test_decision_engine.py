"""Phase 1 skeleton tests for the geometric driving decision engine."""

from __future__ import annotations

import pytest

from app.decision import confidence_fusion, geometric_logic, safety_gate
from app.decision.geometric_logic import FORWARD, LEFT, RIGHT, STOP, make_geometric_decision
from app.lane_detection.lane_geometry import LaneGeometry

# Config matching production.yaml decision_engine defaults
_CONFIG = {
    "offset_threshold": 0.3,
    "curve_threshold": 0.005,
    "obstacle_stop_distance": 2.0,
    "min_confidence": 0.60,
}

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_decision_engine_import():
    """All decision submodules must import cleanly."""
    assert geometric_logic is not None
    assert safety_gate is not None
    assert confidence_fusion is not None


# ---------------------------------------------------------------------------
# Straight-road decisions
# ---------------------------------------------------------------------------


def test_forward_decision():
    """Small offset + gentle curve (both below thresholds) → FORWARD."""
    geom = LaneGeometry(
        offset=0.05,
        curvature=0.001,
        left_lane_detected=True,
        right_lane_detected=True,
    )
    assert make_geometric_decision(geom, _CONFIG) == FORWARD


def test_forward_on_zero_offset_zero_curve():
    """Perfectly centred, straight road → FORWARD."""
    geom = LaneGeometry(offset=0.0, curvature=0.0)
    assert make_geometric_decision(geom, _CONFIG) == FORWARD


def test_forward_at_offset_boundary():
    """Offset exactly at threshold must NOT trigger a correction (exclusive)."""
    # offset == threshold is not strictly greater, so still FORWARD
    geom = LaneGeometry(offset=0.3, curvature=0.0)
    # offset > threshold required to correct; at exactly 0.3 → FORWARD
    result = make_geometric_decision(geom, _CONFIG)
    assert result == FORWARD


# ---------------------------------------------------------------------------
# Lateral drift correction
# ---------------------------------------------------------------------------


def test_left_correction():
    """Positive offset above threshold (drifted right) → LEFT correction."""
    geom = LaneGeometry(offset=+0.4, curvature=0.001)
    assert make_geometric_decision(geom, _CONFIG) == LEFT


def test_right_correction():
    """Negative offset below −threshold (drifted left) → RIGHT correction."""
    geom = LaneGeometry(offset=-0.4, curvature=0.001)
    assert make_geometric_decision(geom, _CONFIG) == RIGHT


# ---------------------------------------------------------------------------
# Curve-ahead decisions
# ---------------------------------------------------------------------------


def test_forward_on_left_curve():
    """Negative curvature exceeding threshold (left curve ahead) → LEFT."""
    geom = LaneGeometry(offset=0.0, curvature=-0.006)
    assert make_geometric_decision(geom, _CONFIG) == LEFT


def test_forward_on_right_curve():
    """Positive curvature exceeding threshold (right curve ahead) → RIGHT."""
    geom = LaneGeometry(offset=0.0, curvature=+0.006)
    assert make_geometric_decision(geom, _CONFIG) == RIGHT


def test_curvature_takes_priority_over_offset():
    """Curvature gate must fire before offset gate when both are triggered.

    Vehicle drifted right (offset=+0.4 → normally LEFT) but road also curves
    left (curvature=-0.006 → LEFT).  Both agree here, confirming priority
    order doesn't cause a contradiction in this case.
    """
    geom = LaneGeometry(offset=+0.4, curvature=-0.006)
    assert make_geometric_decision(geom, _CONFIG) == LEFT


def test_curvature_gate_overrides_offset_disagreement():
    """Curvature gate must fire first even when offset would suggest otherwise.

    Vehicle drifted left (offset=-0.4 → normally RIGHT) but sharp right curve
    (curvature=+0.008 → RIGHT via curvature).  Same result regardless of which
    gate fires, but curvature fires first per the documented priority.
    """
    geom = LaneGeometry(offset=-0.4, curvature=+0.008)
    assert make_geometric_decision(geom, _CONFIG) == RIGHT
