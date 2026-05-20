"""Phase 1 skeleton tests for the safety gate."""

from __future__ import annotations

import pytest

from app.decision import safety_gate as sg_module
from app.decision.safety_gate import SceneContext, should_stop

# Config matching production.yaml decision_engine defaults
_CONFIG = {
    "min_confidence": 0.60,
    "obstacle_stop_distance": 2.0,
}

# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_safety_gate_import():
    """Safety gate module must import cleanly."""
    assert sg_module is not None


# ---------------------------------------------------------------------------
# Immediate hazard trigger
# ---------------------------------------------------------------------------


def test_stop_on_immediate_hazard():
    """immediate_hazard=True must always trigger STOP, regardless of other fields."""
    scene = SceneContext(
        immediate_hazard=True,
        obstacle_detected=False,
        confidence=1.0,
    )
    assert should_stop(scene, _CONFIG) is True


def test_stop_on_immediate_hazard_overrides_high_confidence():
    """STOP must fire even when confidence is high, if hazard flag is set."""
    scene = SceneContext(immediate_hazard=True, confidence=0.99)
    assert should_stop(scene, _CONFIG) is True


# ---------------------------------------------------------------------------
# Clear scene — no stop
# ---------------------------------------------------------------------------


def test_no_stop_on_clear_scene():
    """No hazards, no obstacles, full confidence → safety gate must NOT trigger."""
    scene = SceneContext(
        immediate_hazard=False,
        obstacle_detected=False,
        obstacle_distance=float("inf"),
        confidence=1.0,
    )
    assert should_stop(scene, _CONFIG) is False


def test_no_stop_just_above_confidence_threshold():
    """Confidence at exactly min_confidence must NOT trigger (boundary is exclusive)."""
    scene = SceneContext(immediate_hazard=False, confidence=0.60)
    # confidence < 0.60 triggers; confidence == 0.60 does not
    assert should_stop(scene, _CONFIG) is False


# ---------------------------------------------------------------------------
# Low confidence trigger
# ---------------------------------------------------------------------------


def test_stop_on_low_confidence():
    """Confidence below min_confidence (0.60) → STOP."""
    scene = SceneContext(immediate_hazard=False, confidence=0.45)
    assert should_stop(scene, _CONFIG) is True


def test_stop_on_zero_confidence():
    """Zero confidence is the worst case — must always STOP."""
    scene = SceneContext(immediate_hazard=False, confidence=0.0)
    assert should_stop(scene, _CONFIG) is True


# ---------------------------------------------------------------------------
# Obstacle distance trigger
# ---------------------------------------------------------------------------


def test_stop_on_close_obstacle():
    """Obstacle within stop_distance (2.0 m) must trigger STOP."""
    scene = SceneContext(
        immediate_hazard=False,
        obstacle_detected=True,
        obstacle_distance=1.5,
        confidence=1.0,
    )
    assert should_stop(scene, _CONFIG) is True


def test_no_stop_on_distant_obstacle():
    """Obstacle beyond stop_distance must NOT trigger STOP."""
    scene = SceneContext(
        immediate_hazard=False,
        obstacle_detected=True,
        obstacle_distance=5.0,
        confidence=1.0,
    )
    assert should_stop(scene, _CONFIG) is False


def test_no_stop_when_obstacle_flag_false():
    """obstacle_distance below threshold alone is not enough — flag must also be True."""
    scene = SceneContext(
        immediate_hazard=False,
        obstacle_detected=False,   # flag off
        obstacle_distance=0.5,     # would trigger if flag were True
        confidence=1.0,
    )
    assert should_stop(scene, _CONFIG) is False


# ---------------------------------------------------------------------------
# Stateless / immediate response (no smoothing inside the gate)
# ---------------------------------------------------------------------------


def test_stop_not_blocked_by_smoothing():
    """STOP must be immediate — the gate must not require N frames of history.

    should_stop() is stateless: a brand-new SceneContext with an active hazard
    must trigger STOP on the very first call, with no warm-up or smoothing.
    Any temporal debouncing belongs in the caller, never inside this function.
    """
    scene = SceneContext(immediate_hazard=True, confidence=1.0)
    # First call, no prior state — must return True immediately
    assert should_stop(scene, _CONFIG) is True
    # Second call with identical args — same result (truly stateless)
    assert should_stop(scene, _CONFIG) is True
