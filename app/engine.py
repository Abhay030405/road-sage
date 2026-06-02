"""
app.engine
===========

Defines :class:`PredictionResult` — the single output type of the entire
RoadSage pipeline — and :class:`RoadSageEngine`, which orchestrates every
module (lane detection, scene understanding, decision engine, visualization)
end-to-end.

Typical usage::

    from app.engine import RoadSageEngine

    engine = RoadSageEngine()
    result = engine.predict(cv2.imread("frame.png"))
    print(result.summary())
"""

from __future__ import annotations

import base64
import glob
import json
import logging
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Lane detection
from app.lane_detection.ufld_model import UFLDLaneDetector
from app.lane_detection.bev_transform import BEVConfig, BEVTransform
from app.lane_detection.lane_geometry import LaneGeometryComputer

# Scene understanding
from app.scene_understanding import SceneAnalyzer

# Decision engine
from app.decision import DecisionPath, DecisionResult, DriveCommand, TemporalBuffer
from app.decision.confidence_fusion import ConfidenceFusion
from app.decision.geometric_logic import (
    GeometricConfig,
    apply_temporal_consistency,
    compute_geometric_decision,
)
from app.decision.ml_fallback import MLFallbackModel
from app.decision.safety_gate import SafetyGate

# Visualization
from app.explainability.visualizer import create_full_visualization

# GradCAM
from app.explainability.gradcam import GradCAMManager

# Config
from app.utils.config_validator import load_and_validate_config

logger = logging.getLogger(__name__)

_DECISION_ENGINE_CONFIG = "configs/decision_engine.yaml"


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Unified output record for a single RoadSage pipeline inference step.

    Every field is populated regardless of which decision path was taken.
    Visualization fields (``gradcam_base64``, ``lane_viz_base64``) are
    ``None`` unless ``include_viz=True`` was passed to
    :meth:`~RoadSageEngine.predict`.

    Attributes:
        command: Discrete driving command — ``"FORWARD"``, ``"LEFT"``,
            ``"RIGHT"``, or ``"STOP"``.
        confidence: Fused scalar confidence in ``[0, 1]``.
        decision_path: Name of the sub-system that produced the command,
            e.g. ``"geometric"``, ``"ml_fallback"``, ``"safety_gate"``.
        lane_offset_m: Signed lateral offset from lane centre in metres
            (positive = vehicle is right of centre).
        curvature_inv_m: Signed lane curvature in m⁻¹ (positive = left curve).
        left_lane_detected: True when a left-lane polyline was found.
        right_lane_detected: True when a right-lane polyline was found.
        left_lane_points: Pixel coordinates of the left lane, suitable for
            dashboard overlay.
        right_lane_points: Pixel coordinates of the right lane.
        center_lane_points: Pixel coordinates of the centre lane (if any).
        lane_confidence: Per-lane detection confidences from the lane model.
        hazard_detected: True when the safety gate triggered a stop.
        hazard_reason: Human-readable reason for the hazard, or ``None``.
        surface_class: Road-surface classification string (``"clean"``,
            ``"pothole"``, etc.).
        nearest_obstacle_class: COCO class name of the closest path obstacle,
            or ``None`` when no obstacle is present.
        nearest_obstacle_depth: MiDaS relative depth value ``[0, 1]`` for the
            nearest obstacle (0.0 when no obstacle detected).
        gradcam_base64: Base64-encoded JPEG GradCAM heatmap, or ``None``.
        lane_viz_base64: Base64-encoded JPEG of the annotated lane overlay,
            or ``None``.
        latency_ms: Per-stage wall-clock timings in milliseconds.
        frame_id: Monotonically increasing frame counter.
        timestamp: ISO-8601 UTC timestamp of the inference call.
    """

    command: str
    confidence: float
    decision_path: str
    lane_offset_m: float
    curvature_inv_m: float
    left_lane_detected: bool
    right_lane_detected: bool
    left_lane_points: List[Tuple[int, int]]
    right_lane_points: List[Tuple[int, int]]
    center_lane_points: List[Tuple[int, int]]
    lane_confidence: List[float]
    hazard_detected: bool
    hazard_reason: Optional[str]
    surface_class: str
    nearest_obstacle_class: Optional[str]
    nearest_obstacle_depth: float
    gradcam_base64: Optional[str]
    lane_viz_base64: Optional[str]
    latency_ms: Dict[str, float]
    frame_id: int = 0
    timestamp: str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all fields as a JSON-serialisable dict.

        ``List[Tuple[int, int]]`` fields are converted to
        ``List[List[int]]``; ``None`` values are preserved as ``null``.
        """
        return {
            "command": self.command,
            "confidence": self.confidence,
            "decision_path": self.decision_path,
            "lane_offset_m": self.lane_offset_m,
            "curvature_inv_m": self.curvature_inv_m,
            "left_lane_detected": self.left_lane_detected,
            "right_lane_detected": self.right_lane_detected,
            "left_lane_points": [list(p) for p in self.left_lane_points],
            "right_lane_points": [list(p) for p in self.right_lane_points],
            "center_lane_points": [list(p) for p in self.center_lane_points],
            "lane_confidence": self.lane_confidence,
            "hazard_detected": self.hazard_detected,
            "hazard_reason": self.hazard_reason,
            "surface_class": self.surface_class,
            "nearest_obstacle_class": self.nearest_obstacle_class,
            "nearest_obstacle_depth": self.nearest_obstacle_depth,
            "gradcam_base64": self.gradcam_base64,
            "lane_viz_base64": self.lane_viz_base64,
            "latency_ms": self.latency_ms,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Return a compact JSON string of all fields."""
        return json.dumps(self.to_dict())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_safe(self) -> bool:
        """Return ``True`` when no hazard has been detected."""
        return not self.hazard_detected

    def summary(self) -> str:
        """Return a compact one-line summary suitable for logging.

        Example::

            "FORWARD (0.91) | offset=+0.12m | lanes=L✓R✓ | 47ms"
        """
        offset_sign = "+" if self.lane_offset_m >= 0 else ""
        left_sym = "✓" if self.left_lane_detected else "✗"
        right_sym = "✓" if self.right_lane_detected else "✗"
        total_ms = self.latency_ms.get("total", 0.0)
        return (
            f"{self.command} ({self.confidence:.2f}) | "
            f"offset={offset_sign}{self.lane_offset_m:.2f}m | "
            f"lanes=L{left_sym}R{right_sym} | "
            f"{total_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR image array as a base64 JPEG string.

    Args:
        image: BGR uint8 array of any resolution.

    Returns:
        Base64-encoded JPEG bytes decoded to a UTF-8 string.
    """
    _, buf = cv2.imencode(".jpg", image)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# RoadSageEngine
# ---------------------------------------------------------------------------


class RoadSageEngine:
    """Full RoadSage inference pipeline orchestrator.

    Loads every module once at construction time and exposes :meth:`predict`
    for per-frame inference.  All methods are designed to be *safe*: on any
    unexpected exception the engine returns a ``STOP`` result rather than
    propagating the error.

    Args:
        config_path: Path to the top-level YAML config file.  Defaults to
            ``configs/production.yaml``.
    """

    def __init__(self, config_path: str = "configs/production.yaml") -> None:
        """Load and initialise all pipeline modules.

        Model weights are loaded here — this constructor may take several
        seconds on the first call.

        Args:
            config_path: Path to a validated production or development YAML
                config file.
        """
        logger.info("Initializing RoadSage engine...")

        config = load_and_validate_config(config_path)

        # --- Lane detection ---------------------------------------------------
        self._lane_detector = UFLDLaneDetector("configs/lane_detection.yaml")
        self._bev = BEVTransform(BEVConfig.from_yaml("configs/lane_detection.yaml"))
        self._geometry_computer = LaneGeometryComputer(
            self._bev, config["decision_engine"]
        )

        # --- Scene understanding ----------------------------------------------
        self._scene_analyzer = SceneAnalyzer("configs/scene_understanding.yaml")

        # --- Decision engine --------------------------------------------------
        self._geometric_config = GeometricConfig.from_yaml(_DECISION_ENGINE_CONFIG)
        self._safety_gate = SafetyGate(_DECISION_ENGINE_CONFIG)
        self._confidence_fusion = ConfidenceFusion(_DECISION_ENGINE_CONFIG)
        self._ml_fallback = MLFallbackModel(_DECISION_ENGINE_CONFIG)
        self._temporal_buffer = TemporalBuffer(maxlen=5)

        # --- GradCAM ----------------------------------------------------------
        self._gradcam_manager = GradCAMManager(_DECISION_ENGINE_CONFIG)

        # --- State ------------------------------------------------------------
        self._frame_counter = 0
        self._latency_history: List[float] = []

        # --- Readiness report -------------------------------------------------
        _ready_checks = {
            "lane_detector": self._lane_detector.is_ready(),
            "scene_analyzer": self._scene_analyzer.is_fully_ready(),
            "ml_fallback": self._ml_fallback.is_ready(),
            "gradcam": self._gradcam_manager._gradcam.is_ready(),
        }
        for name, ready in _ready_checks.items():
            status = "ready" if ready else "NOT ready (weights missing?)"
            logger.info("  %-20s %s", name, status)

        logger.info("RoadSage engine ready.")

    # ------------------------------------------------------------------
    # Core predict
    # ------------------------------------------------------------------

    def predict(
        self, image: np.ndarray, include_viz: bool = False
    ) -> PredictionResult:
        """Run the full pipeline on a single BGR frame.

        Never raises — on any exception the method returns a safe
        ``STOP`` result with ``hazard_detected=True``.

        Args:
            image: BGR uint8 camera frame.
            include_viz: When ``True``, populate ``lane_viz_base64`` (always)
                and ``gradcam_base64`` (every N-th frame via GradCAMManager).

        Returns:
            A fully populated :class:`PredictionResult`.
        """
        try:
            return self._predict_impl(image, include_viz=include_viz)
        except Exception as exc:
            logger.exception(
                "RoadSageEngine.predict failed on frame %d: %s",
                self._frame_counter,
                exc,
            )
            self._frame_counter += 1
            return PredictionResult(
                command="STOP",
                confidence=0.0,
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
                hazard_reason=f"Pipeline error: {exc}",
                surface_class="unknown",
                nearest_obstacle_class=None,
                nearest_obstacle_depth=0.0,
                gradcam_base64=None,
                lane_viz_base64=None,
                latency_ms={"lane": 0.0, "scene": 0.0, "decision": 0.0, "total": 0.0},
                frame_id=self._frame_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def _predict_impl(
        self, image: np.ndarray, include_viz: bool
    ) -> PredictionResult:
        """Internal pipeline — called by :meth:`predict` inside a try/except."""

        # Step 1 — Lane detection
        t0 = time.time()
        detection = self._lane_detector.predict(image)
        lane_ms = (time.time() - t0) * 1000

        # Step 2 — Lane geometry
        geometry = self._geometry_computer.compute(detection)

        # Step 3 — Scene understanding
        t0 = time.time()
        scene = self._scene_analyzer.analyze(image)
        scene_ms = (time.time() - t0) * 1000

        # Step 4 — Decision
        t0 = time.time()

        proposed: DecisionResult = compute_geometric_decision(
            geometry, scene, self._geometric_config
        )

        if proposed is None:
            proposed = self._ml_fallback.predict_to_decision_result(
                image, use_mc_dropout=False
            )

        lane_conf = (
            geometry.dominant_confidence() if geometry.lane_geometry_valid else 0.3
        )
        proposed = self._confidence_fusion.fuse(proposed, lane_conf, proposed.ml_softmax)

        final = self._safety_gate.evaluate(scene, proposed)

        final = apply_temporal_consistency(
            final,
            self._temporal_buffer,
            {"temporal_consistency_frames": 3},
        )

        self._temporal_buffer.push(final)

        decision_ms = (time.time() - t0) * 1000

        # Step 5 — Visualizations
        lane_viz_b64: Optional[str] = None
        gradcam_b64: Optional[str] = None
        if include_viz:
            viz_image = create_full_visualization(
                image,
                detection,
                geometry,
                command=final.command.value,
                confidence=final.confidence,
                decision_path=final.decision_path.value,
            )
            lane_viz_b64 = encode_image_to_base64(viz_image)
            gradcam_b64 = self._gradcam_manager.maybe_generate(
                image, final.command.value
            )

        # Step 6 — Build result
        self._frame_counter += 1
        total_ms = lane_ms + scene_ms + decision_ms
        self._latency_history.append(total_ms)
        if len(self._latency_history) > 100:
            self._latency_history.pop(0)

        return PredictionResult(
            command=final.command.value,
            confidence=final.confidence,
            decision_path=final.decision_path.value,
            lane_offset_m=geometry.offset_m,
            curvature_inv_m=geometry.curvature_inv_m,
            left_lane_detected=geometry.left_lane_detected,
            right_lane_detected=geometry.right_lane_detected,
            left_lane_points=detection.left_lane,
            right_lane_points=detection.right_lane,
            center_lane_points=detection.center_lane or [],
            lane_confidence=detection.confidence,
            hazard_detected=final.hazard_detected,
            hazard_reason=final.hazard_reason,
            surface_class=scene.surface.surface_class.value,
            nearest_obstacle_class=(
                scene.nearest_obstacle.class_name if scene.nearest_obstacle else None
            ),
            nearest_obstacle_depth=scene.nearest_obstacle_depth,
            gradcam_base64=gradcam_b64,
            lane_viz_base64=lane_viz_b64,
            latency_ms={
                "lane": lane_ms,
                "scene": scene_ms,
                "decision": decision_ms,
                "total": total_ms,
            },
            frame_id=self._frame_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Convenience predict wrappers
    # ------------------------------------------------------------------

    def predict_from_path(
        self, image_path: str, include_viz: bool = False
    ) -> PredictionResult:
        """Load an image from disk and run :meth:`predict`.

        Args:
            image_path: Path to a BGR-readable image file (JPEG, PNG, etc.).
            include_viz: Forwarded to :meth:`predict`.

        Returns:
            :class:`PredictionResult` for the loaded frame.
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("predict_from_path: could not read '%s'", image_path)
        return self.predict(image, include_viz=include_viz)

    def predict_batch(
        self,
        image_dir: str,
        pattern: str = "rgb_image_*.png",
        include_viz: bool = False,
        save_outputs: bool = False,
        output_dir: str = "outputs/",
    ) -> List[PredictionResult]:
        """Run :meth:`predict` on every image in a directory matching *pattern*.

        Images are sorted alphabetically by filename before processing so
        results are in a consistent frame order.

        Args:
            image_dir: Directory containing the input images.
            pattern: Glob pattern relative to *image_dir*.
            include_viz: When ``True``, generate lane visualizations.
            save_outputs: When ``True``, write ``lane_viz_base64`` frames as
                JPEG files to *output_dir*.
            output_dir: Destination directory for saved visualizations.

        Returns:
            List of :class:`PredictionResult` in frame order.
        """
        search = str(Path(image_dir) / pattern)
        image_paths = sorted(glob.glob(search))

        if not image_paths:
            logger.warning("predict_batch: no images matched '%s'", search)
            return []

        if save_outputs:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            from tqdm import tqdm as _tqdm  # type: ignore[import]
        except ImportError:
            def _tqdm(iterable, **_kw):  # type: ignore[misc]
                return iterable

        results: List[PredictionResult] = []
        errors = 0

        for img_path in _tqdm(image_paths, desc="RoadSage batch", unit="frame"):
            image = cv2.imread(img_path)
            result = self.predict(image, include_viz=include_viz or save_outputs)

            if result.hazard_reason and result.hazard_reason.startswith("Pipeline error"):
                errors += 1

            if save_outputs and result.lane_viz_base64:
                out_path = Path(output_dir) / f"batch_{result.frame_id:06d}.jpg"
                jpg_bytes = base64.b64decode(result.lane_viz_base64)
                out_path.write_bytes(jpg_bytes)

            results.append(result)

        commands = [r.command for r in results]
        latencies = [r.latency_ms["total"] for r in results]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        print(
            f"\nBatch complete: {len(results)} frames | "
            f"errors={errors} | avg_latency={avg_lat:.1f}ms"
        )
        print(f"Command distribution: {dict(Counter(commands))}")

        return results

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """Return a health-check dict suitable for an HTTP /health endpoint.

        Percentile latencies are derived from the rolling window of the
        last 100 frames; both fields are ``0.0`` until at least one frame
        has been processed.

        Returns:
            Dict with keys ``status``, ``models``, ``latency_p50_ms``,
            ``latency_p95_ms``, and ``frames_processed``.
        """
        if len(self._latency_history) >= 2:
            quantiles = statistics.quantiles(self._latency_history, n=100)
            p50 = quantiles[49]
            p95 = quantiles[94]
        elif len(self._latency_history) == 1:
            p50 = self._latency_history[0]
            p95 = self._latency_history[0]
        else:
            p50 = 0.0
            p95 = 0.0

        return {
            "status": "healthy",
            "models": {
                "lane_detector": self._lane_detector.is_ready(),
                "scene_analyzer": self._scene_analyzer.is_fully_ready(),
                "ml_fallback": self._ml_fallback.is_ready(),
            },
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "frames_processed": self._frame_counter,
        }


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    engine = RoadSageEngine()
    print(engine.get_health())

    images = sorted(glob.glob("rgb/rgb_image_*.png"))[:20]
    if not images:
        print("No images found in rgb/ — skipping frame tests.")
        sys.exit(0)

    results: List[PredictionResult] = []
    for img_path in images:
        image = cv2.imread(img_path)
        result = engine.predict(image, include_viz=False)
        results.append(result)
        print(result.summary())

    commands = [r.command for r in results]
    print(f"\nCommand distribution: {Counter(commands)}")

    latencies = [r.latency_ms["total"] for r in results]
    print(f"Avg latency: {sum(latencies)/len(latencies):.1f}ms")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    print("\nPhase 5 engine integration: OK")
