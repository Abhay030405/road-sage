"""
app.decision.geometric_logic
=============================

Purely functional, stateless geometric decision module.

Given a :class:`~app.lane_detection.lane_geometry.LaneGeometry` and an
optional :class:`~app.scene_understanding.SceneContext`, this module maps
lane geometry measurements directly to a :class:`~app.decision.DecisionResult`
using a priority-ordered rule chain.

It returns ``None`` when geometric evidence is insufficient, signalling the
caller (the main engine) to invoke the ML fallback.

No I/O, no model loading, no side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import yaml

from app.decision import (
    DecisionPath,
    DecisionResult,
    DriveCommand,
    TemporalBuffer,
)
from app.lane_detection.lane_geometry import LaneGeometry

if TYPE_CHECKING:
    from app.scene_understanding import SceneContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeometricConfig:
    """Thresholds governing the geometric decision rules.

    Attributes:
        offset_threshold: Lateral offset (m) above which a correction command
            (LEFT / RIGHT) is issued.
        curve_threshold: Lane curvature (m⁻¹) above which a curve-following
            command is issued.
        strong_offset_threshold: Lateral offset (m) above which a
            high-confidence correction command is issued.
        strong_curve_threshold: Curvature (m⁻¹) above which a
            high-confidence curve-following command is issued.
        min_lane_confidence: Minimum per-lane confidence score required to
            treat a lane detection as reliable.
    """

    offset_threshold: float = 0.3
    curve_threshold: float = 0.005
    strong_offset_threshold: float = 0.6
    strong_curve_threshold: float = 0.012
    min_lane_confidence: float = 0.75

    @classmethod
    def from_yaml(cls, config_path: str) -> "GeometricConfig":
        """Load thresholds from the ``geometric`` section of a YAML config file.

        Args:
            config_path: Path to ``configs/decision_engine.yaml`` (or
                equivalent).

        Returns:
            A :class:`GeometricConfig` populated from the ``geometric`` key.

        Raises:
            KeyError: If the YAML file does not contain a ``geometric`` section.
            FileNotFoundError: If *config_path* does not exist.
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        geo = raw["geometric"]
        return cls(
            offset_threshold=float(geo.get("offset_threshold", cls.offset_threshold)),
            curve_threshold=float(geo.get("curve_threshold", cls.curve_threshold)),
            strong_offset_threshold=float(
                geo.get("strong_offset_threshold", cls.strong_offset_threshold)
            ),
            strong_curve_threshold=float(
                geo.get("strong_curve_threshold", cls.strong_curve_threshold)
            ),
            min_lane_confidence=float(
                geo.get("min_lane_confidence", cls.min_lane_confidence)
            ),
        )


# ---------------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------------

def compute_geometric_signal_strength(
    geometry: LaneGeometry,
    config: GeometricConfig,
) -> float:
    """Compute a scalar signal-strength score in [0, 1] for the lane geometry.

    A higher value means the geometric cue is clear and reliable.  A lower
    value means the geometry is marginal and the caller should weight it less
    or defer to ML.

    Penalties applied:

    - ``-0.20`` if only one lane is detected.
    - ``-0.15`` per lane whose confidence is below ``config.min_lane_confidence``.
    - ``-0.10`` if ``|offset_m|`` is within 10 % of ``offset_threshold``
      (i.e. near the decision boundary).

    Args:
        geometry: Lane geometry measurements for this frame.
        config: Thresholds used for boundary-proximity penalty.

    Returns:
        Signal strength in ``[0.0, 1.0]``.  Returns ``0.0`` if
        ``geometry.lane_geometry_valid`` is ``False``.
    """
    if not geometry.lane_geometry_valid:
        return 0.0

    strength = 1.0

    # Penalty: only one lane visible
    if not geometry.both_lanes_detected():
        strength -= 0.2

    # Penalty: low-confidence lane detections
    if geometry.left_lane_detected and geometry.left_lane_confidence < config.min_lane_confidence:
        strength -= 0.15
    if geometry.right_lane_detected and geometry.right_lane_confidence < config.min_lane_confidence:
        strength -= 0.15

    # Penalty: offset close to the decision boundary (±10 %)
    boundary_margin = config.offset_threshold * 0.10
    if abs(geometry.offset_m) > (config.offset_threshold - boundary_margin):
        if abs(geometry.offset_m) < (config.offset_threshold + boundary_margin):
            strength -= 0.10

    return float(max(0.0, min(1.0, strength)))


# ---------------------------------------------------------------------------
# Core decision function
# ---------------------------------------------------------------------------

def compute_geometric_decision(
    geometry: LaneGeometry,
    scene: Optional["SceneContext"],  # noqa: F821
    config: GeometricConfig,
) -> Optional[DecisionResult]:
    """Map lane geometry measurements to a :class:`DecisionResult`.

    Implements a priority chain:

    1. **Validity** — returns ``None`` when geometry is invalid.
    2. **Dual-lane** — uses offset and curvature to issue
       FORWARD / LEFT / RIGHT with calibrated confidence.
    3. **Single-lane fallback** — uses whichever lane is visible to issue
       a low-confidence directional correction.
    4. **Insufficient evidence** — returns ``None``.

    ``None`` propagates up to the engine, which will then invoke the ML
    fallback model.

    Args:
        geometry: Per-frame lane geometry.
        scene: Optional scene context (reserved for future integration;
            not used in the current rule chain).
        config: Geometric thresholds.

    Returns:
        A :class:`DecisionResult` or ``None`` when evidence is too weak.
    """
    # ------------------------------------------------------------------
    # STEP 1 — Validity gate
    # ------------------------------------------------------------------
    if not geometry.lane_geometry_valid:
        logger.debug("Geometric decision skipped: lane_geometry_valid=False")
        return None

    signal = compute_geometric_signal_strength(geometry, config)
    offset = geometry.offset_m
    curve = geometry.curvature_inv_m

    # ------------------------------------------------------------------
    # STEP 2 — Both lanes detected
    # ------------------------------------------------------------------
    if geometry.both_lanes_detected():
        # --- Strong lateral offset (highest priority) ---
        if abs(offset) > config.strong_offset_threshold:
            command = DriveCommand.LEFT if offset > 0 else DriveCommand.RIGHT
            confidence = 0.95
            logger.debug(
                "Strong offset %+.3f m → %s (conf=%.2f)", offset, command.value, confidence
            )

        # --- Moderate lateral offset ---
        elif abs(offset) > config.offset_threshold:
            command = DriveCommand.LEFT if offset > 0 else DriveCommand.RIGHT
            confidence = 0.80 + 0.10 * signal
            logger.debug(
                "Moderate offset %+.3f m → %s (conf=%.2f)", offset, command.value, confidence
            )

        # --- Strong curvature ---
        elif abs(curve) > config.strong_curve_threshold:
            # Positive curvature = lane curves to the left → steer RIGHT
            command = DriveCommand.RIGHT if curve > 0 else DriveCommand.LEFT
            confidence = 0.90
            logger.debug(
                "Strong curve %.4f m⁻¹ → %s (conf=%.2f)", curve, command.value, confidence
            )

        # --- Moderate curvature ---
        elif abs(curve) > config.curve_threshold:
            command = DriveCommand.RIGHT if curve > 0 else DriveCommand.LEFT
            confidence = 0.75 + 0.10 * signal
            logger.debug(
                "Moderate curve %.4f m⁻¹ → %s (conf=%.2f)", curve, command.value, confidence
            )

        # --- Centred, low curvature → go forward ---
        else:
            command = DriveCommand.FORWARD
            confidence = 0.70 + 0.25 * signal
            logger.debug("Centred → FORWARD (conf=%.2f)", confidence)

        return DecisionResult(
            command=command,
            confidence=float(confidence),
            decision_path=DecisionPath.GEOMETRIC,
            offset_m=offset,
            curvature_inv_m=curve,
            geometric_signal_strength=signal,
        )

    # ------------------------------------------------------------------
    # STEP 3 — Single-lane fallback
    # ------------------------------------------------------------------
    if geometry.left_lane_detected and not geometry.right_lane_detected:
        # Only the left lane is visible → vehicle is drifting right → steer LEFT
        logger.debug("Single-lane fallback (left only) → LEFT (conf=0.60)")
        return DecisionResult(
            command=DriveCommand.LEFT,
            confidence=0.60,
            decision_path=DecisionPath.SINGLE_LANE,
            offset_m=offset,
            curvature_inv_m=curve,
            geometric_signal_strength=signal,
        )

    if geometry.right_lane_detected and not geometry.left_lane_detected:
        # Only the right lane is visible → vehicle is drifting left → steer RIGHT
        logger.debug("Single-lane fallback (right only) → RIGHT (conf=0.60)")
        return DecisionResult(
            command=DriveCommand.RIGHT,
            confidence=0.60,
            decision_path=DecisionPath.SINGLE_LANE,
            offset_m=offset,
            curvature_inv_m=curve,
            geometric_signal_strength=signal,
        )

    # ------------------------------------------------------------------
    # STEP 4 — Insufficient evidence
    # ------------------------------------------------------------------
    logger.debug("Geometric decision: insufficient lane evidence → None")
    return None


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def apply_temporal_consistency(
    result: DecisionResult,
    buffer: TemporalBuffer,
    config: dict,
) -> DecisionResult:
    """Override a noisy single-frame result with the buffer's dominant command.

    STOP commands are **never** delayed — they are returned immediately
    regardless of buffer state.

    If the last ``n`` buffered commands are all the same *and* they disagree
    with the current frame's command, the buffer's dominant command is
    returned instead (with the buffer's smoothed confidence).  In all other
    cases the frame result is returned unchanged.

    Args:
        result: The :class:`DecisionResult` produced for the current frame.
        buffer: Recent history of :class:`DecisionResult` objects.
        config: Dict containing the key ``temporal_consistency_frames``
            (int, default 3).

    Returns:
        Potentially overridden :class:`DecisionResult`.
    """
    # STOP is safety-critical — never delayed
    if result.command == DriveCommand.STOP:
        return result

    n = int(config.get("temporal_consistency_frames", 3))
    dominant = buffer.dominant_command(n)

    if dominant is not None and dominant != result.command:
        if buffer.is_consistent(n):
            logger.debug(
                "Temporal override: frame=%s → buffer=%s (conf=%.2f)",
                result.command.value,
                dominant.value,
                buffer.smoothed_confidence(n),
            )
            return DecisionResult(
                command=dominant,
                confidence=buffer.smoothed_confidence(n),
                decision_path=result.decision_path,
                offset_m=result.offset_m,
                curvature_inv_m=result.curvature_inv_m,
                geometric_signal_strength=result.geometric_signal_strength,
            )

    return result


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

def _make_geometry(
    *,
    left: bool = True,
    right: bool = True,
    offset: float = 0.0,
    curve: float = 0.0,
    valid: bool = True,
    left_conf: float = 0.9,
    right_conf: float = 0.9,
) -> LaneGeometry:
    """Helper: build a minimal LaneGeometry for testing."""
    return LaneGeometry(
        left_lane_detected=left,
        right_lane_detected=right,
        left_lane_confidence=left_conf,
        right_lane_confidence=right_conf,
        offset_m=offset,
        curvature_inv_m=curve,
        lane_geometry_valid=valid,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)

    cfg = GeometricConfig()

    test_cases = [
        # (description, geometry, expected_command)
        ("Both lanes, centred",
         _make_geometry(offset=0.05, curve=0.001),
         DriveCommand.FORWARD),
        ("Both lanes, moderate offset right (+0.4)",
         _make_geometry(offset=+0.4, curve=0.001),
         DriveCommand.LEFT),
        ("Both lanes, moderate offset left (-0.4)",
         _make_geometry(offset=-0.4, curve=0.001),
         DriveCommand.RIGHT),
        ("Both lanes, moderate curve left (+0.008)",
         _make_geometry(offset=0.1, curve=+0.008),
         DriveCommand.RIGHT),
        ("Both lanes, moderate curve right (-0.008)",
         _make_geometry(offset=0.1, curve=-0.008),
         DriveCommand.LEFT),
        ("Both lanes, strong offset right (+0.7)",
         _make_geometry(offset=+0.7, curve=0.001),
         DriveCommand.LEFT),
        ("Left lane only",
         _make_geometry(left=True, right=False, offset=0.1, curve=0.001),
         DriveCommand.LEFT),
        ("No lanes detected",
         _make_geometry(left=False, right=False, valid=False),
         None),
    ]

    all_passed = True
    for desc, geom, expected in test_cases:
        result = compute_geometric_decision(geom, scene=None, config=cfg)
        actual = result.command if result is not None else None
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {desc}")
        print(f"         expected={expected}  got={actual}")
        if result:
            print(f"         {result.describe()}")

    print()
    if all_passed:
        print("All geometric logic tests passed.")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
