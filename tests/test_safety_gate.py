"""Phase 4 test suite — SafetyGate, SafetyGateResult, is_stop_always_immediate."""

from __future__ import annotations

from app.decision import DecisionPath, DecisionResult, DriveCommand
from app.decision.safety_gate import (
    SafetyGate,
    SafetyGateResult,
    is_stop_always_immediate,
)
from app.scene_understanding import SceneContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scene(immediate_hazard: bool = False, hazard_reason: str | None = None) -> SceneContext:
    return SceneContext(immediate_hazard=immediate_hazard, hazard_reason=hazard_reason)


def _proposed(command: DriveCommand, confidence: float = 0.88) -> DecisionResult:
    return DecisionResult(command, confidence, DecisionPath.GEOMETRIC)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_safety_gate_triggers_on_hazard():
    scene = make_scene(immediate_hazard=True, hazard_reason="car at 1m")
    gate = SafetyGate()
    result = gate.evaluate(scene, _proposed(DriveCommand.FORWARD))
    assert result.command == DriveCommand.STOP
    assert result.decision_path == DecisionPath.SAFETY_GATE
    assert result.hazard_detected is True


def test_safety_gate_no_trigger_on_clear():
    scene = make_scene(immediate_hazard=False)
    gate = SafetyGate()
    result = gate.evaluate(scene, _proposed(DriveCommand.FORWARD))
    assert result.command == DriveCommand.FORWARD


def test_safety_gate_triggers_low_confidence():
    scene = make_scene(immediate_hazard=False)
    gate = SafetyGate()
    result = gate.evaluate(scene, _proposed(DriveCommand.LEFT, confidence=0.45))
    assert result.command == DriveCommand.STOP
    assert result.decision_path in (DecisionPath.SAFETY_GATE, DecisionPath.CONFIDENCE_GATE)


def test_stop_not_blocked():
    scene = make_scene(immediate_hazard=False)
    gate = SafetyGate()
    result = gate.evaluate(scene, _proposed(DriveCommand.STOP, confidence=0.99))
    assert result.command == DriveCommand.STOP  # STOP always passes through


def test_confidence_gate_just_below():
    scene = make_scene(immediate_hazard=False)
    gate = SafetyGate()
    result = gate.evaluate(scene, _proposed(DriveCommand.FORWARD, confidence=0.59))
    assert result.command == DriveCommand.STOP


def test_confidence_gate_just_above():
    scene = make_scene(immediate_hazard=False)
    gate = SafetyGate()
    result = gate.evaluate(scene, _proposed(DriveCommand.FORWARD, confidence=0.61))
    assert result.command == DriveCommand.FORWARD


def test_safety_gate_result_to_decision():
    gate_result = SafetyGateResult(
        triggered=True,
        reason="test hazard",
        override_command=DriveCommand.STOP,
        original_command=DriveCommand.FORWARD,
    )
    original = DecisionResult(DriveCommand.FORWARD, 0.8, DecisionPath.GEOMETRIC)
    final = gate_result.to_decision_result(original)
    assert final.command == DriveCommand.STOP
    assert final.confidence == 1.0


def test_is_stop_always_immediate():
    assert is_stop_always_immediate() is True
