"""Bird's-eye-view (BEV) perspective transform for road images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BEVTransform:
    """Perspective warp that maps a front-facing camera view to bird's-eye view.

    Constructed from four corresponding point pairs: the trapezoidal road
    region in the original image and its rectangular destination in BEV space.

    Args:
        src_points: Four ``[x, y]`` pixel coordinates in the camera image.
        dst_points: Corresponding four ``[x, y]`` coordinates in BEV space.

    Example::

        src = [[200, 720], [1100, 720], [685, 450], [595, 450]]
        dst = [[300, 720], [980, 720], [980, 0],   [300, 0]]
        bev = BEVTransform(src_points=src, dst_points=dst)
        M   = bev.transform_matrix          # (3, 3) float64
        top_view = bev.warp(frame, (1280, 720))
    """

    src_points: List[List[float]]
    dst_points: List[List[float]]

    @property
    def transform_matrix(self) -> np.ndarray:
        """Compute the 3×3 homogeneous perspective transform matrix.

        Uses ``cv2.getPerspectiveTransform`` which requires exactly four
        non-collinear point correspondences in *src_points* and *dst_points*.

        Returns:
            A ``(3, 3)`` ``float64`` array *M* such that a homogeneous source
            point **p** maps to *M* @ **p** in BEV space.
        """
        src = np.float32(self.src_points)
        dst = np.float32(self.dst_points)
        return cv2.getPerspectiveTransform(src, dst)

    def warp(self, image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Warp *image* into bird's-eye view.

        Args:
            image: BGR image array (H × W × 3).
            output_size: ``(width, height)`` of the output BEV image.

        Returns:
            Warped BGR image array of shape ``(height, width, 3)``.
        """
        return cv2.warpPerspective(image, self.transform_matrix, output_size)
