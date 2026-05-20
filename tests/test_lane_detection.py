"""Phase 1 skeleton tests for lane detection modules.

Tests that require model weights (UFLDv2 inference) are marked skip and will
be activated in Phase 5 once weights are downloaded.
"""

from __future__ import annotations

import pytest
import numpy as np

from app.lane_detection import bev_transform as bev_module
from app.lane_detection import lane_geometry as lane_module
from app.lane_detection import ufld_model
from app.lane_detection.bev_transform import BEVTransform
from app.lane_detection.lane_geometry import LaneGeometry, compute_lateral_offset

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_lane_detection_import():
    """All lane detection submodules must import without error."""
    assert ufld_model is not None
    assert bev_module is not None
    assert lane_module is not None


# ---------------------------------------------------------------------------
# LaneGeometry dataclass
# ---------------------------------------------------------------------------


def test_lane_geometry_dataclass():
    """LaneGeometry must be constructable and expose all expected fields."""
    geom = LaneGeometry(
        offset=0.1,
        curvature=0.003,
        left_lane_detected=True,
        right_lane_detected=False,
        left_x=420.0,
        right_x=860.0,
        confidence=0.9,
    )
    assert hasattr(geom, "offset")
    assert hasattr(geom, "curvature")
    assert hasattr(geom, "left_lane_detected")
    assert hasattr(geom, "right_lane_detected")
    assert hasattr(geom, "left_x")
    assert hasattr(geom, "right_x")
    assert hasattr(geom, "confidence")

    assert geom.offset == pytest.approx(0.1)
    assert geom.curvature == pytest.approx(0.003)
    assert geom.left_lane_detected is True
    assert geom.right_lane_detected is False
    assert geom.confidence == pytest.approx(0.9)


def test_lane_geometry_defaults():
    """LaneGeometry default values must represent a safe, centred state."""
    geom = LaneGeometry()
    assert geom.offset == 0.0
    assert geom.left_lane_detected is True
    assert geom.right_lane_detected is True
    assert geom.confidence == 1.0


# ---------------------------------------------------------------------------
# BEVTransform
# ---------------------------------------------------------------------------


def test_bev_transform_matrix_shape():
    """BEVTransform must produce a 3×3 perspective transform matrix."""
    src = [[200, 720], [1100, 720], [685, 450], [595, 450]]
    dst = [[300, 720], [980, 720], [980, 0], [300, 0]]
    bev = BEVTransform(src_points=src, dst_points=dst)
    M = bev.transform_matrix
    assert M.shape == (3, 3), f"Expected matrix shape (3, 3), got {M.shape}"
    assert M.dtype in (np.float32, np.float64)


def test_bev_transform_matrix_is_invertible():
    """A valid perspective transform matrix must be invertible (det ≠ 0)."""
    src = [[200, 720], [1100, 720], [685, 450], [595, 450]]
    dst = [[300, 720], [980, 720], [980, 0], [300, 0]]
    bev = BEVTransform(src_points=src, dst_points=dst)
    det = np.linalg.det(bev.transform_matrix)
    assert abs(det) > 1e-6, "Transform matrix must be invertible"


# ---------------------------------------------------------------------------
# Lateral offset calculation
# ---------------------------------------------------------------------------


def test_offset_calculation():
    """compute_lateral_offset must return the correct signed normalised value.

    Setup:
        lane spans x = [400, 880]  →  lane_center = 640,  half_width = 240
        vehicle at image_center_x = 448  →  192 px left of lane centre

    Expected:
        offset = (448 - 640) / 240 = -192 / 240 = -0.8  (drifted left)
    """
    offset = compute_lateral_offset(left_x=400, right_x=880, image_center_x=448)
    assert offset == pytest.approx(-0.8, abs=1e-6)


def test_offset_zero_when_centred():
    """Offset must be exactly 0.0 when the vehicle is at lane centre."""
    offset = compute_lateral_offset(left_x=400, right_x=880, image_center_x=640)
    assert offset == pytest.approx(0.0, abs=1e-9)


def test_offset_positive_when_drifted_right():
    """Offset must be positive when the vehicle is right of lane centre."""
    offset = compute_lateral_offset(left_x=400, right_x=880, image_center_x=800)
    assert offset > 0


def test_offset_invalid_raises():
    """compute_lateral_offset must raise ValueError for degenerate lane width."""
    with pytest.raises(ValueError, match="left_x"):
        compute_lateral_offset(left_x=880, right_x=400, image_center_x=640)


# ---------------------------------------------------------------------------
# Model inference tests — skipped until Phase 5
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Model weights not yet downloaded — Phase 5")
def test_ufld_inference_on_blank_frame():
    """UFLDv2 must return lane coordinates for a valid input frame."""
    pass


@pytest.mark.skip(reason="Model weights not yet downloaded — Phase 5")
def test_ufld_confidence_threshold():
    """Low-confidence predictions must be filtered below the threshold."""
    pass


@pytest.mark.skip(reason="Model weights not yet downloaded — Phase 5")
def test_ufld_num_output_lanes():
    """Model must return exactly num_lanes lane predictions."""
    pass
