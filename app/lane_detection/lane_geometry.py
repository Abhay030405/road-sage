"""Lane geometry data structures and lateral offset computation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LaneGeometry:
    """Computed geometric properties of the detected lane ahead.

    All pixel coordinates are in the BEV (bird's-eye-view) frame after
    perspective warping.

    Attributes:
        offset: Signed lateral offset normalised by lane half-width.
            Positive → vehicle drifted right of centre.
            Negative → vehicle drifted left of centre.
            Range is roughly [-1, 1] when the vehicle is within the lane.
        curvature: Signed road curvature in m⁻¹.
            Positive → right curve ahead.
            Negative → left curve ahead.
        left_lane_detected: Whether the left lane marking was found.
        right_lane_detected: Whether the right lane marking was found.
        left_x: x-pixel of the left lane marking at the image bottom row.
        right_x: x-pixel of the right lane marking at the image bottom row.
        confidence: Detection confidence in [0, 1].
    """

    offset: float = 0.0
    curvature: float = 0.0
    left_lane_detected: bool = True
    right_lane_detected: bool = True
    left_x: float = 0.0
    right_x: float = 0.0
    confidence: float = 1.0


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
        * Negative → vehicle is **left** of lane centre  (drifted left).

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
