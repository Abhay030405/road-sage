"""Phase 4 test suite — DriveCommand, DecisionResult, TemporalBuffer, geometric logic."""

from __future__ import annotations

from app.decision import DecisionPath, DecisionResult, DriveCommand, TemporalBuffer
from app.decision.geometric_logic import (
    GeometricConfig,
    compute_geometric_decision,
    compute_geometric_signal_strength,
)
from app.lane_detection.lane_geometry import LaneGeometry
from app.scene_understanding import SceneContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_geometry(
    offset: float,
    curvature: float,
    left: bool = True,
    right: bool = True,
    valid: bool = True,
) -> LaneGeometry:
    """Create a LaneGeometry with both extended and backward-compat fields set."""
    return LaneGeometry(
        offset=offset,
        curvature=curvature,
        offset_m=offset,
        curvature_inv_m=curvature,
        left_lane_detected=left,
        right_lane_detected=right,
        left_lane_confidence=0.90 if left else 0.0,
        right_lane_confidence=0.90 if right else 0.0,
        lane_geometry_valid=valid and (left or right),
    )


def make_scene(immediate_hazard: bool = False, hazard_reason: str | None = None) -> SceneContext:
    """Create a minimal SceneContext for testing."""
    return SceneContext(immediate_hazard=immediate_hazard, hazard_reason=hazard_reason)


def make_empty_scene() -> SceneContext:
    return make_scene()

# ---------------------------------------------------------------------------
# SECTION 1: DriveCommand enum
# ---------------------------------------------------------------------------


def test_drive_command_values():
    assert DriveCommand.FORWARD.value == "FORWARD"
    assert DriveCommand.LEFT.is_correction() is True
    assert DriveCommand.RIGHT.is_correction() is True
    assert DriveCommand.FORWARD.is_correction() is False
    assert DriveCommand.STOP.is_correction() is False


def test_drive_command_opposite():
    assert DriveCommand.LEFT.opposite() == DriveCommand.RIGHT
    assert DriveCommand.RIGHT.opposite() == DriveCommand.LEFT
    assert DriveCommand.FORWARD.opposite() == DriveCommand.FORWARD


def test_drive_command_to_int():
    assert DriveCommand.FORWARD.to_int() == 0
    assert DriveCommand.LEFT.to_int() == 1
    assert DriveCommand.RIGHT.to_int() == 2
    assert DriveCommand.STOP.to_int() == 3
    assert DriveCommand.from_int(0) == DriveCommand.FORWARD
    assert DriveCommand.from_int(3) == DriveCommand.STOP


# ---------------------------------------------------------------------------
# SECTION 2: DecisionResult
# ---------------------------------------------------------------------------


def test_decision_result_describe():
    result = DecisionResult(
        DriveCommand.FORWARD, 0.91, DecisionPath.GEOMETRIC,
        offset_m=0.12, curvature_inv_m=0.003,
    )
    desc = result.describe()
    assert "FORWARD" in desc
    assert any(s in desc for s in ("0.91", "91"))
    assert "geometric" in desc.lower()


def test_decision_result_to_dict():
    result = DecisionResult(DriveCommand.LEFT, 0.80, DecisionPath.SINGLE_LANE)
    d = result.to_dict()
    assert d["command"] == "LEFT"
    assert d["confidence"] == 0.80
    assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# SECTION 3: TemporalBuffer
# ---------------------------------------------------------------------------


def test_temporal_buffer_push_and_len():
    buf = TemporalBuffer(maxlen=3)
    for _ in range(5):
        buf.push(DecisionResult(DriveCommand.FORWARD, 0.9, DecisionPath.GEOMETRIC))
    assert len(buf._buffer) == 3


def test_temporal_buffer_dominant_command():
    buf = TemporalBuffer(maxlen=5)
    for _ in range(3):
        buf.push(DecisionResult(DriveCommand.LEFT, 0.85, DecisionPath.GEOMETRIC))
    buf.push(DecisionResult(DriveCommand.FORWARD, 0.80, DecisionPath.GEOMETRIC))
    buf.push(DecisionResult(DriveCommand.FORWARD, 0.80, DecisionPath.GEOMETRIC))
    assert buf.dominant_command(5) == DriveCommand.LEFT


def test_temporal_buffer_smoothed_confidence():
    buf = TemporalBuffer(maxlen=5)
    for c in [0.6, 0.7, 0.8, 0.85, 0.9]:
        buf.push(DecisionResult(DriveCommand.FORWARD, c, DecisionPath.GEOMETRIC))
    smoothed = buf.smoothed_confidence(5)
    assert smoothed > 0.75   # recent (high) values weighted more
    assert smoothed < 0.9


def test_temporal_buffer_empty():
    buf = TemporalBuffer(maxlen=5)
    assert buf.dominant_command(3) is None
    assert buf.smoothed_confidence(5) == 0.0
    assert buf.is_consistent(3) is False


# ---------------------------------------------------------------------------
# SECTION 4: GeometricConfig
# ---------------------------------------------------------------------------


def test_geometric_config_from_yaml():
    config = GeometricConfig.from_yaml("configs/decision_engine.yaml")
    assert config.offset_threshold == 0.3
    assert config.curve_threshold == 0.005
    assert isinstance(config.strong_offset_threshold, float)


# ---------------------------------------------------------------------------
# SECTION 5: compute_geometric_decision — 20 test cases
# ---------------------------------------------------------------------------


def test_forward_centered():
    geometry = make_geometry(0.05, 0.001)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.FORWARD


def test_forward_small_curve():
    geometry = make_geometry(0.1, 0.003)  # below curve_threshold=0.005
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.FORWARD


def test_left_positive_offset():
    geometry = make_geometry(+0.4, 0.001)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT


def test_right_negative_offset():
    geometry = make_geometry(-0.4, 0.001)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.RIGHT


def test_right_positive_curve():
    geometry = make_geometry(0.05, +0.008)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.RIGHT


def test_left_negative_curve():
    geometry = make_geometry(0.05, -0.008)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT


def test_left_strong_offset():
    geometry = make_geometry(+0.7, 0.001)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT
    assert result.confidence >= 0.90


def test_right_strong_offset():
    geometry = make_geometry(-0.7, 0.001)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.RIGHT


def test_right_strong_curve():
    geometry = make_geometry(0.05, +0.015)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.RIGHT
    assert result.confidence >= 0.85


def test_left_strong_curve():
    geometry = make_geometry(0.05, -0.015)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT


def test_single_lane_left_only():
    geometry = make_geometry(0.0, 0.0, left=True, right=False)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT
    assert result.decision_path == DecisionPath.SINGLE_LANE


def test_single_lane_right_only():
    geometry = make_geometry(0.0, 0.0, left=False, right=True)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.RIGHT
    assert result.decision_path == DecisionPath.SINGLE_LANE


def test_no_lanes_returns_none():
    geometry = make_geometry(0.0, 0.0, left=False, right=False, valid=False)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is None


def test_offset_just_above_threshold():
    geometry = make_geometry(+0.31, 0.001)  # just above 0.3
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT


def test_offset_just_below_threshold():
    geometry = make_geometry(+0.29, 0.001)  # just below 0.3
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.FORWARD


def test_offset_dominates_curve():
    geometry = make_geometry(+0.4, +0.003)  # strong offset, weak curve below threshold
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.command == DriveCommand.LEFT  # offset wins


def test_decision_path_geometric():
    geometry = make_geometry(0.0, 0.001)
    result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
    assert result is not None
    assert result.decision_path == DecisionPath.GEOMETRIC


def test_confidence_range():
    for offset in [0.0, 0.2, 0.4, 0.7]:
        geometry = make_geometry(offset, 0.001)
        result = compute_geometric_decision(geometry, make_empty_scene(), GeometricConfig())
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0


def test_geometric_signal_strength_valid():
    geometry = make_geometry(0.1, 0.002)
    signal = compute_geometric_signal_strength(geometry, GeometricConfig())
    assert 0.0 <= signal <= 1.0


def test_geometric_signal_zero_on_invalid():
    geometry = make_geometry(0.0, 0.0, valid=False)
    signal = compute_geometric_signal_strength(geometry, GeometricConfig())
    assert signal == 0.0
