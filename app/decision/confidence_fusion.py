"""
app.decision.confidence_fusion
================================

Fuses lane-detection confidence, geometric signal strength, and ML softmax
scores into a single scalar confidence for the final :class:`~app.decision.DecisionResult`.

Also implements Monte Carlo Dropout uncertainty estimation from a collection
of stochastic forward-pass softmax vectors.

No I/O, no model loading.  All operations are pure Python + NumPy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import yaml

from app.decision import DecisionPath, DecisionResult, DriveCommand

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Weights and thresholds for confidence fusion.

    Attributes:
        weight_lane: Base weight assigned to lane-detection confidence.
        weight_geometric: Weight assigned to the geometric signal strength.
            Only active when the signal is ``>= min_geometric_signal``.
        weight_ml: Weight assigned to the ML softmax maximum.
            Only active when an ML result was actually used.
        min_geometric_signal: Minimum geometric signal strength required to
            include the geometric weight in the fusion.  Values below this
            indicate the lane geometry is too weak to contribute.
        min_confidence: Minimum fused confidence considered acceptable.
            Informational only here; enforcement lives in
            :class:`~app.decision.safety_gate.SafetyGate`.
    """

    weight_lane: float = 0.4
    weight_geometric: float = 0.35
    weight_ml: float = 0.25
    min_geometric_signal: float = 0.1
    min_confidence: float = 0.60

    @classmethod
    def from_yaml(cls, config_path: str) -> "FusionConfig":
        """Load fusion weights from the ``confidence_fusion`` section of a YAML file.

        Args:
            config_path: Path to ``configs/decision_engine.yaml``.

        Returns:
            A :class:`FusionConfig` populated from the ``confidence_fusion`` key.

        Raises:
            KeyError: If the YAML file does not contain a ``confidence_fusion`` section.
            FileNotFoundError: If *config_path* does not exist.
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        sec = raw["confidence_fusion"]
        return cls(
            weight_lane=float(sec.get("weight_lane", cls.weight_lane)),
            weight_geometric=float(sec.get("weight_geometric", cls.weight_geometric)),
            weight_ml=float(sec.get("weight_ml", cls.weight_ml)),
            min_geometric_signal=float(
                sec.get("min_geometric_signal", cls.min_geometric_signal)
            ),
            min_confidence=float(
                sec.get("min_confidence", cls.min_confidence)
            ),
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Output of a single confidence fusion computation.

    Attributes:
        final_confidence: The weighted-average confidence in [0, 1].
        lane_confidence: Raw lane-detection confidence passed in.
        geometric_signal: Geometric signal strength passed in.
        ml_confidence: ML softmax maximum passed in (0.0 if ML not used).
        fusion_weights_used: Normalised weights ``(w_lane, [w_geo], [w_ml])``
            as actually applied.  Length varies (1–3 elements) depending on
            which signals were active.
        uncertainty: MC Dropout predictive entropy in [0, 1].  Set to
            ``0.0`` when MC Dropout was not performed.
    """

    final_confidence: float
    lane_confidence: float
    geometric_signal: float
    ml_confidence: float
    fusion_weights_used: Tuple[float, ...]
    uncertainty: float = 0.0


# ---------------------------------------------------------------------------
# Fusion function
# ---------------------------------------------------------------------------

def fuse_confidence(
    lane_confidence: float,
    geometric_signal_strength: float,
    ml_softmax_max: float,
    config: FusionConfig,
) -> FusionResult:
    """Compute a weighted-average confidence from up to three signals.

    Signals are included conditionally:

    - **Lane confidence** — always included.
    - **Geometric signal** — included only when
      ``geometric_signal_strength >= config.min_geometric_signal``.
    - **ML confidence** — included only when ``ml_softmax_max > 0``
      (i.e. the ML fallback was actually invoked).

    The raw weights for active signals are re-normalised so they sum to 1,
    ensuring the final confidence is always in [0, 1].

    Args:
        lane_confidence: Per-frame lane detection confidence in [0, 1].
        geometric_signal_strength: Signal strength from
            :func:`~app.decision.geometric_logic.compute_geometric_signal_strength`
            in [0, 1].
        ml_softmax_max: Maximum value of the ML softmax vector (i.e. the
            model's top-class probability).  Pass ``0.0`` when ML was not used.
        config: Fusion weights and thresholds.

    Returns:
        A fully populated :class:`FusionResult`.
    """
    active_weights: List[float] = []
    active_values: List[float] = []

    # Lane confidence — always active
    active_weights.append(config.weight_lane)
    active_values.append(float(lane_confidence))

    # Geometric signal — only when meaningful
    if geometric_signal_strength >= config.min_geometric_signal:
        active_weights.append(config.weight_geometric)
        active_values.append(float(geometric_signal_strength))

    # ML confidence — only when ML was actually used
    if ml_softmax_max > 0.0:
        active_weights.append(config.weight_ml)
        active_values.append(float(ml_softmax_max))

    total_weight = sum(active_weights)
    final_confidence = sum(w * v for w, v in zip(active_weights, active_values)) / total_weight
    final_confidence = float(max(0.0, min(1.0, final_confidence)))

    normalised_weights = tuple(w / total_weight for w in active_weights)

    logger.debug(
        "Confidence fusion: lane=%.3f geo=%.3f ml=%.3f → %.3f (weights=%s)",
        lane_confidence,
        geometric_signal_strength,
        ml_softmax_max,
        final_confidence,
        normalised_weights,
    )

    return FusionResult(
        final_confidence=final_confidence,
        lane_confidence=float(lane_confidence),
        geometric_signal=float(geometric_signal_strength),
        ml_confidence=float(ml_softmax_max),
        fusion_weights_used=normalised_weights,
    )


# ---------------------------------------------------------------------------
# ConfidenceFusion class
# ---------------------------------------------------------------------------

class ConfidenceFusion:
    """Stateful wrapper that applies confidence fusion to a :class:`~app.decision.DecisionResult`.

    Args:
        config_path: Path to ``configs/decision_engine.yaml``.
    """

    def __init__(self, config_path: str = "configs/decision_engine.yaml") -> None:
        self._config = FusionConfig.from_yaml(config_path)
        logger.info(
            "ConfidenceFusion loaded: w_lane=%.2f w_geo=%.2f w_ml=%.2f",
            self._config.weight_lane,
            self._config.weight_geometric,
            self._config.weight_ml,
        )

    def fuse(
        self,
        proposed_result: DecisionResult,
        lane_confidence: float,
        ml_softmax: Optional[List[float]] = None,
    ) -> DecisionResult:
        """Apply confidence fusion and return an updated :class:`~app.decision.DecisionResult`.

        The command and all geometric fields of *proposed_result* are
        preserved unchanged; only the ``confidence`` and ``ml_softmax``
        fields are updated.

        Args:
            proposed_result: The result from geometric or ML modules.
            lane_confidence: Lane detection confidence for this frame.
            ml_softmax: Softmax vector from the ML fallback model, or
                ``None`` if ML was not invoked.

        Returns:
            A new :class:`~app.decision.DecisionResult` with fused confidence.
        """
        ml_softmax_max = float(max(ml_softmax)) if ml_softmax else 0.0
        fusion = fuse_confidence(
            lane_confidence,
            proposed_result.geometric_signal_strength,
            ml_softmax_max,
            self._config,
        )

        return DecisionResult(
            command=proposed_result.command,
            confidence=fusion.final_confidence,
            decision_path=proposed_result.decision_path,
            offset_m=proposed_result.offset_m,
            curvature_inv_m=proposed_result.curvature_inv_m,
            geometric_signal_strength=proposed_result.geometric_signal_strength,
            ml_softmax=ml_softmax,
            hazard_detected=proposed_result.hazard_detected,
            hazard_reason=proposed_result.hazard_reason,
            inference_time_ms=proposed_result.inference_time_ms,
        )

    def compute_mc_uncertainty(self, softmax_samples: List[List[float]]) -> float:
        """Compute predictive entropy from Monte Carlo Dropout softmax samples.

        Aggregates ``N`` stochastic forward-pass softmax vectors into a single
        uncertainty score in [0, 1].  A score near 0 means the model is
        confident; near 1 means maximally uncertain.

        Algorithm:

        1. Compute the mean softmax across all ``N`` samples (per class).
        2. Compute predictive entropy:
           :math:`H = -\\sum_c p_c \\log(p_c + \\epsilon)` where
           :math:`\\epsilon = 10^{-8}`.
        3. Normalise by :math:`\\log(C)` where :math:`C` is the number of
           classes, so the result is in [0, 1].

        Args:
            softmax_samples: List of ``N`` softmax vectors, each of length
                ``num_classes``.  All vectors must have the same length.

        Returns:
            Normalised predictive entropy in [0, 1].  Returns ``0.0`` if
            the input list is empty.
        """
        if not softmax_samples:
            return 0.0

        arr = np.array(softmax_samples, dtype=np.float64)  # (N, C)
        mean_softmax = arr.mean(axis=0)                     # (C,)
        num_classes = len(mean_softmax)

        entropy = -float(np.sum(mean_softmax * np.log(mean_softmax + 1e-8)))
        max_entropy = math.log(num_classes)

        if max_entropy == 0.0:
            return 0.0

        return float(min(1.0, entropy / max_entropy))


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)

    cfg = FusionConfig()
    cf = ConfidenceFusion.__new__(ConfidenceFusion)
    cf._config = cfg

    def _proposed(conf: float, geo: float = 0.0) -> DecisionResult:
        return DecisionResult(
            command=DriveCommand.FORWARD,
            confidence=conf,
            decision_path=DecisionPath.GEOMETRIC,
            geometric_signal_strength=geo,
        )

    # Test 1: strong lane + strong geo, no ML
    r1 = fuse_confidence(0.9, 0.85, 0.0, cfg)
    assert r1.final_confidence > 0.7, f"Test 1 failed: {r1.final_confidence:.3f}"
    print(f"  [PASS] Test 1 — strong lane+geo (no ML): conf={r1.final_confidence:.3f}  "
          f"weights={r1.fusion_weights_used}")

    # Test 2: medium signals across all three sources
    r2 = fuse_confidence(0.5, 0.3, 0.7, cfg)
    assert 0.4 <= r2.final_confidence <= 0.7, f"Test 2 failed: {r2.final_confidence:.3f}"
    print(f"  [PASS] Test 2 — medium all sources: conf={r2.final_confidence:.3f}  "
          f"weights={r2.fusion_weights_used}")

    # Test 3: weak lane, weak geo (below threshold so excluded), weak ML
    r3 = fuse_confidence(0.3, 0.05, 0.4, cfg)
    # geo excluded (0.05 < 0.1), so only lane(0.4) + ml(0.25) active
    assert r3.final_confidence < 0.60, f"Test 3 failed: {r3.final_confidence:.3f}"
    print(f"  [PASS] Test 3 — weak signals, near STOP threshold: conf={r3.final_confidence:.3f}  "
          f"weights={r3.fusion_weights_used}")

    # Test 4: MC Dropout with 10 uniform softmax samples → high uncertainty
    uniform = [[0.25, 0.25, 0.25, 0.25]] * 10
    unc = cf.compute_mc_uncertainty(uniform)
    assert unc > 0.9, f"Test 4 failed: uncertainty={unc:.3f} (expected >0.9 for uniform)"
    print(f"  [PASS] Test 4 — MC Dropout uniform (max uncertainty): unc={unc:.4f}")

    # Test 5: MC Dropout with confident samples → low uncertainty
    confident = [[0.97, 0.01, 0.01, 0.01]] * 10
    unc2 = cf.compute_mc_uncertainty(confident)
    assert unc2 < 0.2, f"Test 5 failed: uncertainty={unc2:.3f} (expected <0.2 for confident)"
    print(f"  [PASS] Test 5 — MC Dropout confident (low uncertainty): unc={unc2:.4f}")

    print()
    print("All confidence fusion tests passed.")
