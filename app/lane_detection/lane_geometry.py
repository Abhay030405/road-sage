"""Lane geometry data structures and computation.

Bridges raw lane pixel coordinates (from UFLDv2) with metric quantities
(lateral offset, road curvature, lane width) needed by the decision engine.

Coordinate conventions
----------------------
* All pixel coordinates are in the BEV (bird's-eye-view) frame after
  perspective warping unless stated otherwise.
* Polynomial form: ``x = A·y² + B·y + C`` (y increases downward).
* Offset sign: positive → vehicle is right of lane centre (steer left).
* Curvature sign: positive → right curve, negative → left curve.

Usage::

    bev = BEVTransform(BEVConfig.from_yaml("configs/lane_detection.yaml"))
    detector = UFLDLaneDetector()
    computer = LaneGeometryComputer(bev, config["confidence"])

    result  = detector.predict(frame)
    geom    = computer.compute(result)

    print(geom.describe())           # "lanes=L+R offset=0.03m curvature=0.001m⁻¹ …"
    print(geom.both_lanes_detected()) # True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from app.lane_detection.bev_transform import BEVTransform
    from app.lane_detection.ufld_model import LaneDetectionResult, UFLDLaneDetector

log = logging.getLogger(__name__)

# Standard assumed lane width used when only one marking is visible.
_DEFAULT_LANE_WIDTH_M = 3.5


# ---------------------------------------------------------------------------
# LaneGeometry dataclass
# ---------------------------------------------------------------------------


@dataclass
class LaneGeometry:
    """Computed geometric properties of the detected lane ahead.

    Backward-compatible fields (used by existing tests and decision engine):
        offset, curvature, left_lane_detected, right_lane_detected,
        left_x, right_x, confidence.

    Extended spec fields:
        offset_m, curvature_inv_m, vanishing_point_x, road_width_m,
        center_lane_detected, left_lane_confidence, right_lane_confidence,
        lane_geometry_valid, left_coeffs, right_coeffs.

    Note: ``offset`` ≡ ``offset_m`` and ``curvature`` ≡ ``curvature_inv_m``.
    Both pairs are always set to the same value so callers can use either name.
    """

    # ------------------------------------------------------------------ #
    # Backward-compatible fields                                           #
    # ------------------------------------------------------------------ #

    offset: float = 0.0
    """Signed lateral offset in metres (positive = drifted right)."""

    curvature: float = 0.0
    """Signed road curvature in m⁻¹ (positive = right curve)."""

    left_lane_detected: bool = True
    right_lane_detected: bool = True

    left_x: float = 0.0
    """BEV x-pixel of the left lane marking at the bottom image row."""

    right_x: float = 0.0
    """BEV x-pixel of the right lane marking at the bottom image row."""

    confidence: float = 1.0
    """Overall detection confidence in [0, 1]."""

    # ------------------------------------------------------------------ #
    # Extended spec fields                                                 #
    # ------------------------------------------------------------------ #

    offset_m: float = 0.0
    """Same as ``offset`` — lateral offset in metres."""

    curvature_inv_m: float = 0.0
    """Same as ``curvature`` — road curvature in m⁻¹."""

    vanishing_point_x: Optional[int] = None
    """BEV x-pixel where left and right polynomials intersect (horizon proxy)."""

    road_width_m: float = 0.0
    """Lane width in metres at the bottom image row."""

    center_lane_detected: bool = False

    left_lane_confidence: float = 0.0
    right_lane_confidence: float = 0.0

    lane_geometry_valid: bool = True
    """False when no usable lane data was extracted from the detection result."""

    left_coeffs: Optional[List[float]] = None
    """Polynomial coefficients [A, B, C] for the left lane, or None."""

    right_coeffs: Optional[List[float]] = None
    """Polynomial coefficients [A, B, C] for the right lane, or None."""

    # ------------------------------------------------------------------ #
    # Convenience methods                                                  #
    # ------------------------------------------------------------------ #

    def both_lanes_detected(self) -> bool:
        """Return True when both left and right markings were found."""
        return self.left_lane_detected and self.right_lane_detected

    def dominant_confidence(self) -> float:
        """Return the highest per-lane confidence available for this frame.

        Both lanes detected → max of left and right confidences.
        One lane detected   → confidence of that lane.
        No lanes detected   → 0.0.
        """
        if self.left_lane_detected and self.right_lane_detected:
            return max(self.left_lane_confidence, self.right_lane_confidence)
        if self.left_lane_detected:
            return self.left_lane_confidence
        if self.right_lane_detected:
            return self.right_lane_confidence
        return 0.0

    def describe(self) -> str:
        """Return a compact human-readable summary of the geometry."""
        parts = []
        if self.left_lane_detected:
            parts.append("L")
        if self.center_lane_detected:
            parts.append("C")
        if self.right_lane_detected:
            parts.append("R")
        lane_str = "+".join(parts) if parts else "none"

        if abs(self.curvature_inv_m) < 0.005:
            direction = "straight"
        elif self.curvature_inv_m > 0:
            direction = "right curve"
        else:
            direction = "left curve"

        return (
            f"lanes={lane_str} "
            f"offset={self.offset_m:+.3f}m "
            f"curvature={self.curvature_inv_m:+.4f}m⁻¹ ({direction}) "
            f"road_width={self.road_width_m:.2f}m "
            f"conf={self.dominant_confidence():.2f}"
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def make_empty_geometry() -> LaneGeometry:
    """Return an invalid, zeroed geometry used when detection completely fails.

    Sets ``lane_geometry_valid=False`` and all ``*_detected`` flags to False
    so downstream consumers can reject the frame safely.
    """
    return LaneGeometry(
        offset=0.0,
        curvature=0.0,
        left_lane_detected=False,
        right_lane_detected=False,
        left_x=0.0,
        right_x=0.0,
        confidence=0.0,
        offset_m=0.0,
        curvature_inv_m=0.0,
        vanishing_point_x=None,
        road_width_m=0.0,
        center_lane_detected=False,
        left_lane_confidence=0.0,
        right_lane_confidence=0.0,
        lane_geometry_valid=False,
        left_coeffs=None,
        right_coeffs=None,
    )


def compute_lateral_offset(
    left_x: float,
    right_x: float,
    image_center_x: float,
) -> float:
    """Compute the normalised lateral offset of the vehicle from lane centre.

    The result is normalised by the lane half-width so that ±1.0 means the
    vehicle centre is exactly over a lane boundary.

    Formula::

        lane_center     = (left_x + right_x) / 2
        lane_half_width = (right_x - left_x) / 2
        offset          = (image_center_x - lane_center) / lane_half_width

    Sign convention:
        * Positive → vehicle is **right** of lane centre (drifted right).
        * Negative → vehicle is **left** of lane centre (drifted left).

    Args:
        left_x: x-pixel coordinate of the left lane marking.
        right_x: x-pixel coordinate of the right lane marking (must be
            strictly greater than *left_x*).
        image_center_x: x-pixel coordinate representing the vehicle position
            (typically half the image width, or an ego-motion-corrected value).

    Returns:
        Signed normalised lateral offset.  Values outside ``[-1, 1]`` indicate
        the vehicle has crossed a lane boundary.

    Raises:
        ValueError: When *left_x* >= *right_x* (degenerate or inverted lane).
    """
    if left_x >= right_x:
        raise ValueError(
            f"left_x ({left_x}) must be strictly less than right_x ({right_x})."
        )
    lane_center = (left_x + right_x) / 2.0
    lane_half_width = (right_x - left_x) / 2.0
    return (image_center_x - lane_center) / lane_half_width


def geometry_from_single_lane(
    lane_x_points: List[Tuple[int, int]],
    image_width: int,
    pixels_per_meter: float,
    is_left_lane: bool,
) -> LaneGeometry:
    """Build a best-effort LaneGeometry from a single detected lane marking.

    When only one boundary is visible, the missing boundary is inferred by
    assuming a standard lane width of 3.5 m.

    Args:
        lane_x_points: List of ``(x, y)`` pixel coordinates of the detected
            lane marking (already in BEV space).
        image_width: Width of the BEV image in pixels; used to derive the
            assumed vehicle centre.
        pixels_per_meter: BEV calibration constant (px per real-world metre).
        is_left_lane: ``True`` if the provided points are the left boundary;
            ``False`` if they are the right boundary.

    Returns:
        A :class:`LaneGeometry` instance with one lane marked as detected.
        ``lane_geometry_valid`` is set to ``True`` only when ≥ 5 points are
        provided (enough for a reliable polynomial fit).
    """
    if len(lane_x_points) < 2:
        return make_empty_geometry()

    xs = np.array([p[0] for p in lane_x_points], dtype=np.float64)
    ys = np.array([p[1] for p in lane_x_points], dtype=np.float64)
    y_bottom = float(ys.max())

    # Fit polynomial if enough points
    coeffs: Optional[np.ndarray] = None
    if len(lane_x_points) >= 5:
        try:
            coeffs = np.polyfit(ys, xs, deg=2)
        except np.linalg.LinAlgError:
            coeffs = None

    lane_x_at_bottom = (
        float(np.polyval(coeffs, y_bottom)) if coeffs is not None
        else float(xs[np.argmax(ys)])
    )

    assumed_width_px = _DEFAULT_LANE_WIDTH_M * pixels_per_meter
    image_center_x = image_width / 2.0

    if is_left_lane:
        left_x = lane_x_at_bottom
        right_x = left_x + assumed_width_px
        left_detected, right_detected = True, False
        left_conf = float(len(lane_x_points)) / max(len(lane_x_points), 1)
        right_conf = 0.0
        left_coeffs_list = coeffs.tolist() if coeffs is not None else None
        right_coeffs_list = None
    else:
        right_x = lane_x_at_bottom
        left_x = right_x - assumed_width_px
        left_detected, right_detected = False, True
        right_conf = float(len(lane_x_points)) / max(len(lane_x_points), 1)
        left_conf = 0.0
        left_coeffs_list = None
        right_coeffs_list = coeffs.tolist() if coeffs is not None else None

    lane_center_x = (left_x + right_x) / 2.0
    offset_m = (lane_center_x - image_center_x) / pixels_per_meter
    overall_conf = left_conf if is_left_lane else right_conf

    return LaneGeometry(
        offset=offset_m,
        curvature=0.0,
        left_lane_detected=left_detected,
        right_lane_detected=right_detected,
        left_x=left_x,
        right_x=right_x,
        confidence=overall_conf,
        offset_m=offset_m,
        curvature_inv_m=0.0,
        vanishing_point_x=None,
        road_width_m=_DEFAULT_LANE_WIDTH_M,
        center_lane_detected=False,
        left_lane_confidence=left_conf,
        right_lane_confidence=right_conf,
        lane_geometry_valid=coeffs is not None,
        left_coeffs=left_coeffs_list,
        right_coeffs=right_coeffs_list,
    )


# ---------------------------------------------------------------------------
# LaneGeometryComputer
# ---------------------------------------------------------------------------


class LaneGeometryComputer:
    """Converts a :class:`LaneDetectionResult` into a :class:`LaneGeometry`.

    Applies the BEV perspective transform to detected lane pixels, fits
    2nd-degree polynomials, and computes metric quantities (offset, curvature,
    road width, vanishing point) using the calibrated ``pixels_per_meter``.

    Args:
        bev_transform: A configured :class:`BEVTransform` instance.
        config: Dict matching the ``confidence`` section of
            ``configs/lane_detection.yaml``.  Reads:
            * ``lane_conf_threshold`` (default 0.75)
            * ``min_points_per_lane`` (default 5)
    """

    def __init__(self, bev_transform: "BEVTransform", config: dict) -> None:
        self._bev = bev_transform
        self._conf_threshold = float(config.get("lane_conf_threshold", 0.75))
        self._min_points = int(config.get("min_points_per_lane", 5))

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compute(self, detection_result: "LaneDetectionResult") -> LaneGeometry:
        """Compute full lane geometry from a single detection result.

        Args:
            detection_result: Raw output from :class:`UFLDLaneDetector`.

        Returns:
            A populated :class:`LaneGeometry`.  Returns
            :func:`make_empty_geometry` when no lanes meet the minimum-point
            threshold.
        """
        left_pts  = detection_result.left_lane
        right_pts = detection_result.right_lane
        center_pts = detection_result.center_lane or []

        left_ok   = len(left_pts)   >= self._min_points
        right_ok  = len(right_pts)  >= self._min_points
        center_ok = len(center_pts) >= self._min_points

        if not left_ok and not right_ok:
            log.debug("No lanes meet min_points threshold — returning empty geometry")
            return make_empty_geometry()

        # ---- Transform detected pixels into BEV space ----------------
        left_bev   = self._bev.transform_points(left_pts)  if left_ok  else []
        right_bev  = self._bev.transform_points(right_pts) if right_ok else []

        # ---- Fit polynomials -----------------------------------------
        left_coeffs_arr  = self._bev.fit_polynomial(left_bev)  if left_ok  else None
        right_coeffs_arr = self._bev.fit_polynomial(right_bev) if right_ok else None

        left_coeffs_list  = left_coeffs_arr.tolist()  if left_coeffs_arr  is not None else None
        right_coeffs_list = right_coeffs_arr.tolist() if right_coeffs_arr is not None else None

        # ---- Derive metric quantities at the bottom BEV row ----------
        y_eval        = float(self._bev.config.output_height - 1)
        image_center  = float(self._bev.config.output_width) / 2.0
        ppm           = self._bev.config.pixels_per_meter

        left_x  = 0.0
        right_x = 0.0
        offset_m        = 0.0
        curvature_inv_m = 0.0
        road_width_m    = 0.0
        vanishing_point_x: Optional[int] = None

        if left_coeffs_arr is not None and right_coeffs_arr is not None:
            left_x  = float(np.polyval(left_coeffs_arr,  y_eval))
            right_x = float(np.polyval(right_coeffs_arr, y_eval))
            offset_m = self._bev.compute_offset(
                left_coeffs_arr, right_coeffs_arr, y_eval, image_center
            )
            curv_left  = self._bev.compute_curvature(left_coeffs_arr,  y_eval)
            curv_right = self._bev.compute_curvature(right_coeffs_arr, y_eval)
            curvature_inv_m = (curv_left + curv_right) / 2.0
            road_width_m = self._bev.compute_road_width(
                left_coeffs_arr, right_coeffs_arr, y_eval
            )
            vp = self._bev.compute_vanishing_point(left_coeffs_arr, right_coeffs_arr)
            if vp is not None:
                vanishing_point_x = int(vp[0])

        elif left_coeffs_arr is not None:
            left_x  = float(np.polyval(left_coeffs_arr, y_eval))
            right_x = left_x + _DEFAULT_LANE_WIDTH_M * ppm
            curvature_inv_m = self._bev.compute_curvature(left_coeffs_arr, y_eval)
            lane_center = (left_x + right_x) / 2.0
            offset_m = (lane_center - image_center) / ppm

        elif right_coeffs_arr is not None:
            right_x = float(np.polyval(right_coeffs_arr, y_eval))
            left_x  = right_x - _DEFAULT_LANE_WIDTH_M * ppm
            curvature_inv_m = self._bev.compute_curvature(right_coeffs_arr, y_eval)
            lane_center = (left_x + right_x) / 2.0
            offset_m = (lane_center - image_center) / ppm

        # ---- Extract per-lane confidences from detection result ------
        confs = detection_result.confidence
        left_conf  = float(confs[0]) if len(confs) > 0 else 0.0
        right_conf = float(confs[1]) if len(confs) > 1 else 0.0

        if not left_ok:
            left_conf = 0.0
        if not right_ok:
            right_conf = 0.0

        if left_ok and right_ok:
            overall_conf = (left_conf + right_conf) / 2.0
        else:
            overall_conf = left_conf if left_ok else right_conf

        return LaneGeometry(
            # Backward-compatible fields
            offset=offset_m,
            curvature=curvature_inv_m,
            left_lane_detected=left_ok,
            right_lane_detected=right_ok,
            left_x=left_x,
            right_x=right_x,
            confidence=overall_conf,
            # Extended spec fields
            offset_m=offset_m,
            curvature_inv_m=curvature_inv_m,
            vanishing_point_x=vanishing_point_x,
            road_width_m=road_width_m,
            center_lane_detected=center_ok,
            left_lane_confidence=left_conf,
            right_lane_confidence=right_conf,
            lane_geometry_valid=True,
            left_coeffs=left_coeffs_list,
            right_coeffs=right_coeffs_list,
        )

    def compute_from_image(
        self,
        image: np.ndarray,
        detector: "UFLDLaneDetector",
    ) -> LaneGeometry:
        """Detect lanes in *image* then compute geometry.

        Convenience wrapper that chains :meth:`UFLDLaneDetector.predict` with
        :meth:`compute`.

        Args:
            image: BGR camera frame (any resolution — the detector handles
                internal resizing).
            detector: A ready :class:`UFLDLaneDetector` instance.

        Returns:
            Computed :class:`LaneGeometry` for the frame.
        """
        result = detector.predict(image)
        return self.compute(result)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import yaml
    from pathlib import Path

    from app.lane_detection.bev_transform import BEVConfig, BEVTransform

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    CONFIG_PATH = "configs/lane_detection.yaml"

    try:
        cfg = BEVConfig.from_yaml(CONFIG_PATH)
        bev = BEVTransform(cfg)
    except (FileNotFoundError, KeyError) as exc:
        log.error("Config error: %s", exc)
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        full_cfg = yaml.safe_load(fh)

    computer = LaneGeometryComputer(bev, full_cfg.get("confidence", {}))

    # Smoke-test: synthesise a detection result with plausible lane points
    # and verify the geometry output.
    try:
        from app.lane_detection.ufld_model import LaneDetectionResult
    except ImportError:
        log.error("Could not import LaneDetectionResult — ensure the package is installed.")
        sys.exit(1)

    H = cfg.output_height
    W = cfg.output_width

    # 20 points along left and right lane boundaries in camera space
    y_vals = list(range(H // 2, H, H // 20 + 1))
    left_cam  = [(W // 4,     y) for y in y_vals]
    right_cam = [(3 * W // 4, y) for y in y_vals]

    fake_result = LaneDetectionResult(
        left_lane=left_cam,
        right_lane=right_cam,
        confidence=[0.90, 0.88],
        no_lanes_detected=False,
    )

    geom = computer.compute(fake_result)
    print("\n=== LaneGeometry smoke test ===")
    print(geom.describe())
    print(f"  both_lanes_detected : {geom.both_lanes_detected()}")
    print(f"  dominant_confidence : {geom.dominant_confidence():.3f}")
    print(f"  lane_geometry_valid : {geom.lane_geometry_valid}")
    print(f"  vanishing_point_x   : {geom.vanishing_point_x}")
    print(f"  left_coeffs         : {geom.left_coeffs}")
    print(f"  right_coeffs        : {geom.right_coeffs}")

    # Also exercise geometry_from_single_lane
    single = geometry_from_single_lane(
        lane_x_points=left_cam,
        image_width=W,
        pixels_per_meter=cfg.pixels_per_meter,
        is_left_lane=True,
    )
    print("\n=== geometry_from_single_lane (left only) ===")
    print(single.describe())

    # Exercise make_empty_geometry
    empty = make_empty_geometry()
    assert not empty.lane_geometry_valid
    assert not empty.both_lanes_detected()
    print("\n=== make_empty_geometry ===")
    print(f"  lane_geometry_valid : {empty.lane_geometry_valid}")
    print(f"  both_lanes_detected : {empty.both_lanes_detected()}")

    print("\nAll smoke tests passed.")
