"""
app.decision.safety_gate
=========================

Purely deterministic highest-priority safety layer.

A STOP issued by this module can never be overridden by anything downstream.
It is the first check performed by the main engine on every frame.

No ML, no randomness, no state beyond the loaded config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import yaml

from app.decision import DecisionPath, DecisionResult, DriveCommand

if TYPE_CHECKING:
    from app.scene_understanding import SceneContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    """Thresholds governing the safety gate.

    Attributes:
        obstacle_stop_depth_threshold: Normalised inverse depth value (in
            [0, 1]) above which an obstacle is considered immediately
            dangerous.  Populated from scene_understanding config; used by
            :class:`~app.scene_understanding.SceneAnalyzer` upstream.
        min_confidence: Decision confidence below which a STOP is forced.
            Covers cases where the full pipeline is too uncertain to act.
        temporal_consistency_frames: Window size passed to the temporal
            buffer helpers in :mod:`~app.decision.geometric_logic`.
        max_command_jump: Reserved for future smoothing — maximum allowed
            one-step change in command integer (0–3).
    """

    obstacle_stop_depth_threshold: float = 0.7
    min_confidence: float = 0.60
    temporal_consistency_frames: int = 3
    max_command_jump: int = 2

    @classmethod
    def from_yaml(cls, config_path: str) -> "SafetyConfig":
        """Load thresholds from the ``safety`` section of a YAML config file.

        Args:
            config_path: Path to ``configs/decision_engine.yaml``.

        Returns:
            A :class:`SafetyConfig` populated from the ``safety`` key.

        Raises:
            KeyError: If the YAML file does not contain a ``safety`` section.
            FileNotFoundError: If *config_path* does not exist.
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        sec = raw["safety"]
        return cls(
            obstacle_stop_depth_threshold=float(
                sec.get("obstacle_stop_depth_threshold", cls.obstacle_stop_depth_threshold)
            ),
            min_confidence=float(sec.get("min_confidence", cls.min_confidence)),
            temporal_consistency_frames=int(
                sec.get("temporal_consistency_frames", cls.temporal_consistency_frames)
            ),
            max_command_jump=int(sec.get("max_command_jump", cls.max_command_jump)),
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SafetyGateResult:
    """Output of a single safety-gate evaluation.

    Attributes:
        triggered: ``True`` when the gate forces a STOP.
        reason: Human-readable description of the trigger condition, or
            ``None`` when the gate did not trigger.
        override_command: Always :attr:`~app.decision.DriveCommand.STOP` when
            triggered, ``None`` otherwise.
        original_command: The command that was proposed before the gate ran.
    """

    triggered: bool
    reason: Optional[str]
    override_command: Optional[DriveCommand]
    original_command: Optional[DriveCommand]

    def to_decision_result(self, original: DecisionResult) -> DecisionResult:
        """Convert this gate result into a final :class:`~app.decision.DecisionResult`.

        If the gate was triggered, returns a STOP result with full hazard
        metadata while preserving the geometric fields (offset, curvature)
        from the original for logging purposes.

        If the gate was **not** triggered, returns *original* unchanged.

        Args:
            original: The proposed result from upstream modules.

        Returns:
            Either an overriding STOP :class:`~app.decision.DecisionResult`
            or *original* as-is.
        """
        if self.triggered:
            return DecisionResult(
                command=DriveCommand.STOP,
                confidence=1.0,
                decision_path=DecisionPath.SAFETY_GATE,
                hazard_detected=True,
                hazard_reason=self.reason,
                offset_m=original.offset_m,
                curvature_inv_m=original.curvature_inv_m,
            )
        return original


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_safety(
    scene: "SceneContext",
    proposed_result: DecisionResult,
    config: SafetyConfig,
) -> SafetyGateResult:
    """Evaluate whether the safety gate must override the proposed command.

    Conditions are checked **in priority order**; the first match returns
    immediately.

    1. **Immediate scene hazard** — ``scene.immediate_hazard`` is ``True``
       (set by :class:`~app.scene_understanding.SceneAnalyzer` when an
       obstacle is within the stop-depth threshold or a road hazard is
       detected with high confidence).
    2. **Low decision confidence** — ``proposed_result.confidence`` is below
       ``config.min_confidence``.  Covers ML-fallback uncertainty and frames
       where the geometric signal was too weak.
    3. **No trigger** — gate passes; returns a non-triggered result.

    Args:
        scene: Fused scene context for the current frame.
        proposed_result: The command produced by geometric or ML modules.
        config: Safety thresholds.

    Returns:
        A :class:`SafetyGateResult` indicating whether the gate fired and why.
    """
    # Condition 1 — immediate hazard flagged by scene understanding
    if scene.immediate_hazard:
        return SafetyGateResult(
            triggered=True,
            reason=scene.hazard_reason or "immediate obstacle detected",
            override_command=DriveCommand.STOP,
            original_command=proposed_result.command,
        )

    # Condition 2 — decision confidence too low
    if proposed_result.confidence < config.min_confidence:
        return SafetyGateResult(
            triggered=True,
            reason=(
                f"confidence {proposed_result.confidence:.2f} "
                f"below threshold {config.min_confidence:.2f}"
            ),
            override_command=DriveCommand.STOP,
            original_command=proposed_result.command,
        )

    # Condition 3 — no hazard detected
    return SafetyGateResult(
        triggered=False,
        reason=None,
        override_command=None,
        original_command=proposed_result.command,
    )


# ---------------------------------------------------------------------------
# Documentation helper
# ---------------------------------------------------------------------------

def is_stop_always_immediate() -> bool:
    """Return ``True``.

    This function exists purely as executable documentation: STOP commands
    issued by the safety gate are **never** smoothed, delayed, buffered, or
    blocked by temporal consistency logic downstream.
    """
    return True


# ---------------------------------------------------------------------------
# SafetyGate class
# ---------------------------------------------------------------------------

class SafetyGate:
    """Stateful wrapper around the safety gate evaluation functions.

    Holds a loaded :class:`SafetyConfig` and exposes two evaluation methods:

    * :meth:`evaluate` — full scene + proposed result check.
    * :meth:`evaluate_confidence_gate` — standalone confidence-only check,
      used when no scene context is available (e.g. during ML-only fallback).

    Args:
        config_path: Path to ``configs/decision_engine.yaml``.  Defaults to
            the project-relative path.
    """

    def __init__(self, config_path: str = "configs/decision_engine.yaml") -> None:
        self._config = SafetyConfig.from_yaml(config_path)
        logger.info(
            "SafetyGate loaded: min_confidence=%.2f", self._config.min_confidence
        )

    def evaluate(
        self,
        scene: "SceneContext",
        proposed: DecisionResult,
    ) -> DecisionResult:
        """Run the full safety check against the current scene.

        Logs a warning whenever the gate triggers.

        Args:
            scene: Fused scene context for the current frame.
            proposed: The command produced by geometric or ML modules.

        Returns:
            Either a forced-STOP :class:`~app.decision.DecisionResult` or
            *proposed* unchanged.
        """
        gate_result = evaluate_safety(scene, proposed, self._config)
        final = gate_result.to_decision_result(proposed)
        if gate_result.triggered:
            logger.warning("SAFETY GATE TRIGGERED: %s", gate_result.reason)
        return final

    def evaluate_confidence_gate(self, result: DecisionResult) -> DecisionResult:
        """Force STOP when decision confidence is below the configured threshold.

        This is a lightweight confidence-only check used when no full scene
        context is available.  It mirrors Condition 2 in :func:`evaluate_safety`
        but can be called independently.

        STOP commands pass through unchanged regardless of confidence.

        Args:
            result: A :class:`~app.decision.DecisionResult` to validate.

        Returns:
            Either *result* unchanged or a STOP result with
            :attr:`~app.decision.DecisionPath.CONFIDENCE_GATE` path.
        """
        if result.is_stop():
            return result
        if result.confidence < self._config.min_confidence:
            logger.warning(
                "Confidence gate triggered: conf=%.2f < %.2f",
                result.confidence,
                self._config.min_confidence,
            )
            return DecisionResult(
                command=DriveCommand.STOP,
                confidence=result.confidence,
                decision_path=DecisionPath.CONFIDENCE_GATE,
                hazard_detected=True,
                hazard_reason=(
                    f"uncertainty too high: conf={result.confidence:.2f}"
                ),
                offset_m=result.offset_m,
                curvature_inv_m=result.curvature_inv_m,
            )
        return result


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dataclasses import dataclass as _dc

    logging.basicConfig(level=logging.WARNING)

    # Minimal SceneContext stand-in for testing (real one lives in app.scene_understanding)
    @_dc
    class _FakeScene:
        immediate_hazard: bool = False
        hazard_reason: Optional[str] = None

    cfg = SafetyConfig()

    def _result(conf: float, cmd: DriveCommand = DriveCommand.FORWARD) -> DecisionResult:
        return DecisionResult(
            command=cmd,
            confidence=conf,
            decision_path=DecisionPath.GEOMETRIC,
        )

    test_cases = [
        # (description, scene, proposed_result, expect_stop)
        ("immediate_hazard=True",
         _FakeScene(immediate_hazard=True),
         _result(0.85),
         True),
        ("immediate_hazard=False, conf=0.85",
         _FakeScene(immediate_hazard=False),
         _result(0.85),
         False),
        ("immediate_hazard=False, conf=0.45",
         _FakeScene(immediate_hazard=False),
         _result(0.45),
         True),
        ("immediate_hazard=True, cmd was FORWARD → output STOP",
         _FakeScene(immediate_hazard=True, hazard_reason="pedestrian"),
         _result(0.90, DriveCommand.FORWARD),
         True),
        ("conf=0.61 just above threshold",
         _FakeScene(),
         _result(0.61),
         False),
        ("conf=0.59 just below threshold",
         _FakeScene(),
         _result(0.59),
         True),
    ]

    all_passed = True
    for desc, scene, proposed, expect_stop in test_cases:
        gate = evaluate_safety(scene, proposed, cfg)
        final = gate.to_decision_result(proposed)
        actual_stop = final.command == DriveCommand.STOP
        ok = actual_stop == expect_stop
        if not ok:
            all_passed = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        print(f"         triggered={gate.triggered}  final={final.command.value}"
              f"  reason={gate.reason!r}")

    print()
    if all_passed:
        print("All safety gate tests passed.")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
