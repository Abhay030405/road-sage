"""RoadSage scene understanding package.

Fuses three sub-models into a single :class:`SceneContext`:

* :class:`~app.scene_understanding.object_detector.ObjectDetector`
  — NanoDet-Plus-m object detection (person, bicycle, car, motorcycle, bus, truck)
* :class:`~app.scene_understanding.depth_estimator.DepthEstimator`
  — MiDaS v2.1 Small relative inverse depth estimation
* :class:`~app.scene_understanding.surface_classifier.SurfaceClassifier`
  — MobileNetV2 road surface classification (clean / pothole / speed_breaker / waterlogged)

Usage::

    analyzer = SceneAnalyzer("configs/scene_understanding.yaml")
    context  = analyzer.analyze(bgr_frame)

    if not context.is_safe():
        print(context.describe())        # "HAZARD: person detected in path"
    else:
        print(context.describe())        # "Clear — 0 path obstacles, surface: clean"
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import yaml

from app.scene_understanding.depth_estimator import DepthEstimationResult, DepthEstimator
from app.scene_understanding.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
)
from app.scene_understanding.surface_classifier import (
    SurfaceClassificationResult,
    SurfaceClass,
    SurfaceClassifier,
)

log = logging.getLogger(__name__)

__all__ = [
    "SceneContext",
    "SceneAnalyzer",
    # Re-export sub-module public types for convenience
    "ObjectDetector",
    "ObjectDetectionResult",
    "DetectedObject",
    "DepthEstimator",
    "DepthEstimationResult",
    "SurfaceClassifier",
    "SurfaceClassificationResult",
    "SurfaceClass",
]


# ---------------------------------------------------------------------------
# SceneContext
# ---------------------------------------------------------------------------


@dataclass
class SceneContext:
    """Fused scene understanding result for a single camera frame.

    Aggregates outputs from the object detector, depth estimator, and surface
    classifier into a single object that the decision engine can query.

    Attributes:
        detections:              Full object detection result for the frame.
        depth:                   Full depth estimation result for the frame.
        surface:                 Road surface classification result.
        nearest_obstacle:        The path obstacle with the largest bounding
            box area (proxy for closest to camera), or ``None``.
        nearest_obstacle_depth:  Normalised inverse-depth value at the nearest
            obstacle's bounding box.  ``0.0`` when there is no obstacle.
        immediate_hazard:        ``True`` when an obstacle is within the danger
            zone or a road hazard (pothole / waterlogged) is detected with
            high confidence.
        hazard_reason:           Human-readable explanation of the hazard, or
            ``None`` when the scene is clear.
        total_inference_time_ms: Sum of all three sub-model inference times.
    """

    detections: ObjectDetectionResult = field(
        default_factory=ObjectDetectionResult
    )
    depth: DepthEstimationResult = field(
        default_factory=DepthEstimationResult
    )
    surface: SurfaceClassificationResult = field(
        default_factory=SurfaceClassificationResult
    )
    nearest_obstacle: Optional[DetectedObject] = None
    nearest_obstacle_depth: float = 0.0
    immediate_hazard: bool = False
    hazard_reason: Optional[str] = None
    total_inference_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def is_safe(self) -> bool:
        """Return ``True`` when no immediate hazard has been detected.

        Returns:
            ``not self.immediate_hazard``
        """
        return not self.immediate_hazard

    def describe(self) -> str:
        """Return a compact human-readable summary of the scene.

        Returns:
            A short string suitable for logging or HUD display, e.g.:
            ``"HAZARD: person detected in path"`` or
            ``"Clear — 2 path obstacles, surface: clean"``.
        """
        if self.immediate_hazard:
            return f"HAZARD: {self.hazard_reason}"
        n_obstacles = len(self.detections.path_obstacles)
        surface_label = self.surface.surface_class.value
        return f"Clear — {n_obstacles} path obstacle(s), surface: {surface_label}"


# ---------------------------------------------------------------------------
# SceneAnalyzer
# ---------------------------------------------------------------------------


class SceneAnalyzer:
    """Orchestrates the three scene understanding sub-models.

    Runs object detection, depth estimation, and road surface classification
    on each frame, then fuses the outputs into a :class:`SceneContext` using
    configurable depth and hazard thresholds.

    Args:
        config_path: Path to ``configs/scene_understanding.yaml``.

    Example::

        analyzer = SceneAnalyzer("configs/scene_understanding.yaml")
        context  = analyzer.analyze(bgr_frame)

        if not context.is_safe():
            trigger_braking()
    """

    def __init__(self, config_path: str = "configs/scene_understanding.yaml") -> None:
        try:
            with open(config_path, encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh)
        except FileNotFoundError:
            log.error("Config not found: %s — using defaults", config_path)
            full_cfg = {}

        depth_cfg   = full_cfg.get("depth_estimator", {})
        fusion_cfg  = full_cfg.get("fusion", {})

        self._stop_threshold: float = float(
            depth_cfg.get("obstacle_stop_depth_threshold", 0.7)
        )
        self._corridor_fraction: float = float(
            fusion_cfg.get("path_corridor_fraction", 0.4)
        )
        self._surface_hazard_conf_threshold: float = 0.8  # high bar for surface hazards

        self._detector   = ObjectDetector(config_path)
        self._depth      = DepthEstimator(config_path)
        self._classifier = SurfaceClassifier(config_path)

        log.info(
            "SceneAnalyzer ready — detector=%s, depth=%s, surface=%s",
            self._detector.is_ready(),
            self._depth.is_ready(),
            self._classifier.is_ready(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image: "cv2.Mat") -> SceneContext:
        """Run all three sub-models and fuse their outputs.

        The three models are executed sequentially.  Future work may
        parallelise them using ``concurrent.futures.ThreadPoolExecutor``.

        Args:
            image: BGR uint8 camera frame.

        Returns:
            A fully populated :class:`SceneContext`.
        """
        t_total = time.perf_counter()

        # ── Sub-model inference ──────────────────────────────────────────
        detections = self._detector.detect(image)
        depth      = self._depth.estimate(image)
        surface    = self._classifier.classify_center_patch(image)

        # ── Fusion ───────────────────────────────────────────────────────
        nearest               = detections.nearest_obstacle()
        nearest_depth: float  = 0.0
        immediate_hazard      = False
        hazard_reason: Optional[str] = None

        if nearest is not None:
            nearest_depth = depth.get_depth_at_bbox(nearest.bbox, padding=5)
            if depth.is_obstacle_close(nearest.bbox, self._stop_threshold):
                immediate_hazard = True
                hazard_reason    = f"{nearest.class_name} detected in path"

        # Surface hazard overrides (high-confidence only)
        if (
            not immediate_hazard
            and surface.is_hazard()
            and surface.confidence >= self._surface_hazard_conf_threshold
        ):
            immediate_hazard = True
            hazard_reason    = (
                f"road surface hazard: {surface.surface_class.value} "
                f"(conf={surface.confidence:.2f})"
            )

        total_ms = (time.perf_counter() - t_total) * 1000.0

        return SceneContext(
            detections=detections,
            depth=depth,
            surface=surface,
            nearest_obstacle=nearest,
            nearest_obstacle_depth=nearest_depth,
            immediate_hazard=immediate_hazard,
            hazard_reason=hazard_reason,
            total_inference_time_ms=total_ms,
        )

    def analyze_from_path(self, image_path: str) -> SceneContext:
        """Load an image from disk and run full scene analysis.

        Args:
            image_path: Filesystem path to a BGR image file.

        Returns:
            A :class:`SceneContext`.  Returns an empty context if the image
            cannot be read.
        """
        import numpy as np

        image = cv2.imread(image_path)
        if image is None:
            log.warning("Could not read image: %s", image_path)
            import numpy as np
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return self.analyze(blank)
        return self.analyze(image)

    def is_fully_ready(self) -> bool:
        """Return ``True`` when both the detector and depth estimator are loaded.

        The surface classifier is optional; its absence does not cause this
        method to return ``False``.

        Returns:
            ``True`` if :meth:`ObjectDetector.is_ready` and
            :meth:`DepthEstimator.is_ready` both return ``True``.
        """
        return self._detector.is_ready() and self._depth.is_ready()


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    os.makedirs("outputs", exist_ok=True)

    analyzer = SceneAnalyzer("configs/scene_understanding.yaml")
    print(
        f"\nReadiness — "
        f"detector: {analyzer._detector.is_ready()}, "
        f"depth: {analyzer._depth.is_ready()}, "
        f"surface: {analyzer._classifier.is_ready()}"
    )

    images = sorted(glob.glob("rgb/rgb_image_*.png"))[:5]
    if not images:
        log.warning("No images found in rgb/. Run from the project root.")
        sys.exit(0)

    for n, img_path in enumerate(images):
        context = analyzer.analyze_from_path(img_path)

        # Save depth visualisation
        depth_vis = analyzer._depth.visualize_depth(context.depth)
        out_path = f"outputs/scene_{n + 1}_depth.png"
        cv2.imwrite(out_path, depth_vis)

        print(
            f"{Path(img_path).name:25s}  "
            f"{context.describe():55s}  "
            f"total={context.total_inference_time_ms:6.1f}ms  "
            f"→ {out_path}"
        )

    print("\nScene analysis pipeline ready.")
