"""Bird's-Eye-View (BEV) / Inverse-Perspective-Mapping (IPM) transform.

Warps a front-facing camera frame into a metrically calibrated top-down
view so that lane curvature, lateral offset, and road width can be computed
in real-world units (metres).

Coordinate convention
---------------------
* Camera image  — origin top-left, x right, y down.
* BEV image     — same orientation; bottom of image = closest to vehicle.
* Polynomial    — ``x = A·y² + B·y + C`` where y increases downward.
* Curvature sign — positive = right curve, negative = left curve.

Usage::

    cfg = BEVConfig.from_yaml("configs/lane_detection.yaml")
    bev = BEVTransform(cfg)
    top_down  = bev.transform_image(frame)
    bev_pts   = bev.transform_points(lane_pixel_coords)
    coeffs    = bev.fit_polynomial(bev_pts)
    curvature = bev.compute_curvature(coeffs, y_eval=bev.config.output_height - 1)
    offset    = bev.compute_offset(left_c, right_c, y_eval, image_center_x=640)
"""

from __future__ import annotations

import glob
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class BEVConfig:
    """All parameters required to build and calibrate a BEV transform.

    Attributes:
        src_points: Four ``[x, y]`` pixel coordinates forming a trapezoid that
            spans the road region in the **original** camera image.
        dst_points: Corresponding four ``[x, y]`` coordinates that define the
            rectangle those points should map to in the BEV output image.
        output_width: Width of the BEV output image in pixels.
        output_height: Height of the BEV output image in pixels.
        pixels_per_meter: Calibration constant — how many BEV pixels equal one
            real-world metre.  Used to convert pixel measurements to metres.
    """

    src_points: List[List[int]]
    dst_points: List[List[int]]
    output_width: int = 1280
    output_height: int = 720
    pixels_per_meter: float = 30.0

    @classmethod
    def from_yaml(cls, config_path: str = "configs/lane_detection.yaml") -> "BEVConfig":
        """Load BEV configuration from the ``bev_transform`` section of a YAML file.

        Args:
            config_path: Path to ``configs/lane_detection.yaml`` (or any YAML
                file that contains a ``bev_transform`` top-level key).

        Returns:
            A populated :class:`BEVConfig` instance.

        Raises:
            FileNotFoundError: When *config_path* does not exist.
            KeyError: When the ``bev_transform`` key is absent.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: '{config_path}'")
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        bev = raw.get("bev_transform")
        if bev is None:
            raise KeyError(f"'bev_transform' section missing in '{config_path}'")
        return cls(
            src_points=bev["src_points"],
            dst_points=bev["dst_points"],
            output_width=bev.get("output_width", 1280),
            output_height=bev.get("output_height", 720),
            pixels_per_meter=float(bev.get("pixels_per_meter", 30.0)),
        )


# ---------------------------------------------------------------------------
# BEV transform class
# ---------------------------------------------------------------------------


class BEVTransform:
    """Perspective warp between camera view and bird's-eye view.

    Pre-computes the forward and inverse 3×3 homography matrices from the
    four point correspondences in *config* (or from raw ``src_points`` /
    ``dst_points`` keyword arguments for backward compatibility).

    Args:
        config: A :class:`BEVConfig` instance.  If ``None``, the class can
            also be constructed directly with keyword arguments
            ``src_points``, ``dst_points``, ``output_width``,
            ``output_height``, and ``pixels_per_meter``.

    Example::

        cfg = BEVConfig.from_yaml("configs/lane_detection.yaml")
        bev = BEVTransform(cfg)
        top_down = bev.transform_image(frame)

    Backward-compatible usage (tests)::

        bev = BEVTransform(src_points=[[200,720],...], dst_points=[[300,720],...])
        M   = bev.transform_matrix    # (3,3) float64
    """

    def __init__(
        self,
        config: Optional[BEVConfig] = None,
        *,
        src_points: Optional[List[List[int]]] = None,
        dst_points: Optional[List[List[int]]] = None,
        output_width: int = 1280,
        output_height: int = 720,
        pixels_per_meter: float = 30.0,
    ) -> None:
        if config is None:
            if src_points is None or dst_points is None:
                raise ValueError(
                    "Provide either a BEVConfig object or both src_points and dst_points."
                )
            config = BEVConfig(
                src_points=src_points,
                dst_points=dst_points,
                output_width=output_width,
                output_height=output_height,
                pixels_per_meter=pixels_per_meter,
            )

        self.config = config

        src = np.float32(config.src_points)
        dst = np.float32(config.dst_points)
        self._M     = cv2.getPerspectiveTransform(src, dst)   # camera → BEV
        self._M_inv = cv2.getPerspectiveTransform(dst, src)   # BEV → camera

    # ------------------------------------------------------------------
    # Matrix access
    # ------------------------------------------------------------------

    @property
    def transform_matrix(self) -> np.ndarray:
        """3×3 forward perspective transform matrix (camera → BEV).

        Returns:
            ``float64`` array of shape ``(3, 3)``.
        """
        return self._M

    def get_transform_matrix(self) -> np.ndarray:
        """Return the 3×3 forward perspective transform matrix.

        Equivalent to the :attr:`transform_matrix` property; provided as a
        method for callers that prefer explicit method calls.

        Returns:
            ``float64`` array of shape ``(3, 3)``.
        """
        return self._M

    # ------------------------------------------------------------------
    # Image warping
    # ------------------------------------------------------------------

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """Warp *image* from camera perspective into bird's-eye view.

        Args:
            image: BGR (or grayscale) image array of any resolution.

        Returns:
            Warped image of shape
            ``(output_height, output_width[, channels])``.
        """
        return cv2.warpPerspective(
            image,
            self._M,
            (self.config.output_width, self.config.output_height),
        )

    def inverse_transform_image(
        self,
        bev_image: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Warp a BEV image back into the original camera perspective.

        Args:
            bev_image: BGR (or grayscale) BEV image array.
            output_size: ``(width, height)`` of the target camera-view image.

        Returns:
            Image warped back to camera perspective, shape
            ``(height, width[, channels])``.
        """
        return cv2.warpPerspective(bev_image, self._M_inv, output_size)

    # ------------------------------------------------------------------
    # Point transforms
    # ------------------------------------------------------------------

    def transform_points(
        self,
        points: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Map ``(x, y)`` pixel coordinates from camera view to BEV space.

        Args:
            points: List of ``(x, y)`` integer tuples in camera-view pixel
                coordinates.

        Returns:
            List of ``(x, y)`` integer tuples in BEV pixel coordinates.
            Empty list when *points* is empty.
        """
        if not points:
            return []
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._M)
        return [(int(p[0][0]), int(p[0][1])) for p in transformed]

    def inverse_transform_points(
        self,
        points: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Map ``(x, y)`` BEV pixel coordinates back to camera-view space.

        Args:
            points: List of ``(x, y)`` integer tuples in BEV pixel coordinates.

        Returns:
            List of ``(x, y)`` integer tuples in camera-view pixel coordinates.
        """
        if not points:
            return []
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._M_inv)
        return [(int(p[0][0]), int(p[0][1])) for p in transformed]

    # ------------------------------------------------------------------
    # Polynomial geometry
    # ------------------------------------------------------------------

    def fit_polynomial(
        self,
        lane_points: List[Tuple[int, int]],
    ) -> Optional[np.ndarray]:
        """Fit a 2nd-degree polynomial ``x = A·y² + B·y + C`` to BEV lane points.

        Args:
            lane_points: BEV pixel coordinates ``(x, y)`` of the lane
                boundary.

        Returns:
            Coefficient array ``[A, B, C]`` (highest power first) as returned
            by :func:`numpy.polyfit`, or ``None`` when fewer than five points
            are provided.
        """
        if len(lane_points) < 5:
            return None
        xs = np.array([p[0] for p in lane_points], dtype=np.float64)
        ys = np.array([p[1] for p in lane_points], dtype=np.float64)
        # polyfit(y, x, 2) → coefficients for x = A·y² + B·y + C
        return np.polyfit(ys, xs, deg=2)

    def evaluate_polynomial(
        self,
        coeffs: np.ndarray,
        y_values: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the polynomial ``x = A·y² + B·y + C`` at each *y*.

        Args:
            coeffs: Coefficient array ``[A, B, C]`` from :meth:`fit_polynomial`.
            y_values: 1-D array of y coordinates at which to evaluate.

        Returns:
            1-D ``float64`` array of x coordinates.
        """
        A, B, C = coeffs
        return A * y_values ** 2 + B * y_values + C

    def compute_curvature(
        self,
        coeffs: np.ndarray,
        y_eval: float,
    ) -> float:
        """Compute signed road curvature in m⁻¹ at a given image row.

        Uses the standard radius-of-curvature formula for a parametric curve:

        .. math::

            R = \\frac{(1 + (2Ay + B)^2)^{3/2}}{|2A|}

        The pixel-space radius is then converted to metres via
        ``pixels_per_meter``.

        Args:
            coeffs: Polynomial coefficients ``[A, B, C]``.
            y_eval: The y pixel position at which to evaluate curvature
                (typically ``output_height - 1``, i.e. the bottom row).

        Returns:
            Signed curvature ``κ = 1 / R_metres`` in m⁻¹.
            Positive values indicate a right curve; negative a left curve.
            Returns ``0.0`` when the lane is effectively straight (``|A|``
            below a numerical threshold).
        """
        A, B, _ = coeffs
        if abs(A) < 1e-10:
            return 0.0

        dydx   = 2.0 * A * y_eval + B
        R_px   = ((1.0 + dydx ** 2) ** 1.5) / abs(2.0 * A)
        R_m    = R_px / self.config.pixels_per_meter
        sign   = 1.0 if A > 0.0 else -1.0
        return sign / R_m

    def compute_offset(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
        y_eval: float,
        image_center_x: int,
    ) -> float:
        """Compute the lateral offset of the vehicle from lane centre in metres.

        Evaluates both polynomials at *y_eval* (usually the bottom image row,
        closest to the vehicle), finds the lane midpoint, and compares it to
        *image_center_x*.

        Sign convention:
            * Positive → vehicle is **right** of lane centre (steer left).
            * Negative → vehicle is **left** of lane centre (steer right).

        Args:
            left_coeffs: Left-lane polynomial coefficients, or ``None``.
            right_coeffs: Right-lane polynomial coefficients, or ``None``.
            y_eval: Y position in BEV pixels at which to measure offset.
            image_center_x: X pixel coordinate of the vehicle's assumed
                centre (typically ``output_width // 2``).

        Returns:
            Lateral offset in metres.  Returns ``0.0`` when no coefficients
            are available.
        """
        ppm = self.config.pixels_per_meter

        if left_coeffs is not None and right_coeffs is not None:
            left_x  = float(np.polyval(left_coeffs,  y_eval))
            right_x = float(np.polyval(right_coeffs, y_eval))
            lane_center_x = (left_x + right_x) / 2.0
        elif left_coeffs is not None:
            lane_center_x = float(np.polyval(left_coeffs, y_eval))
        elif right_coeffs is not None:
            lane_center_x = float(np.polyval(right_coeffs, y_eval))
        else:
            return 0.0

        offset_px = lane_center_x - image_center_x
        return offset_px / ppm

    def compute_vanishing_point(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
    ) -> Optional[Tuple[int, int]]:
        """Find where the left and right lane polynomials intersect in BEV space.

        Solves ``left_A·y² + left_B·y + left_C = right_A·y² + right_B·y + right_C``,
        which reduces to the quadratic
        ``(dA)·y² + (dB)·y + (dC) = 0``.

        Args:
            left_coeffs: Left-lane polynomial coefficients ``[A, B, C]``.
            right_coeffs: Right-lane polynomial coefficients ``[A, B, C]``.

        Returns:
            ``(x, y)`` integer BEV pixel coordinates of the intersection,
            or ``None`` when one or both polynomials are absent, the lines
            are parallel, or the intersection lies outside the image bounds.
        """
        if left_coeffs is None or right_coeffs is None:
            return None

        dA = left_coeffs[0] - right_coeffs[0]
        dB = left_coeffs[1] - right_coeffs[1]
        dC = left_coeffs[2] - right_coeffs[2]

        if abs(dA) < 1e-10:
            # Effectively linear: dB·y + dC = 0
            if abs(dB) < 1e-10:
                return None   # parallel — no intersection
            y_sol = -dC / dB
        else:
            discriminant = dB ** 2 - 4 * dA * dC
            if discriminant < 0:
                return None
            sqrt_d = math.sqrt(discriminant)
            y1 = (-dB + sqrt_d) / (2 * dA)
            y2 = (-dB - sqrt_d) / (2 * dA)
            # Pick the solution inside the image (smallest positive y)
            candidates = [
                y for y in (y1, y2)
                if 0 <= y <= self.config.output_height
            ]
            if not candidates:
                return None
            y_sol = min(candidates)

        x_sol = float(np.polyval(left_coeffs, y_sol))

        if not (0 <= x_sol <= self.config.output_width):
            return None

        return (int(x_sol), int(y_sol))

    def compute_road_width(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
        y_eval: float,
    ) -> float:
        """Return the lane width in metres at a given BEV row.

        Args:
            left_coeffs: Left-lane polynomial coefficients, or ``None``.
            right_coeffs: Right-lane polynomial coefficients, or ``None``.
            y_eval: Y position in BEV pixels at which to measure width.

        Returns:
            Lane width in metres, or ``0.0`` when either polynomial is absent.
        """
        if left_coeffs is None or right_coeffs is None:
            return 0.0
        left_x  = float(np.polyval(left_coeffs,  y_eval))
        right_x = float(np.polyval(right_coeffs, y_eval))
        width_px = max(0.0, right_x - left_x)
        return width_px / self.config.pixels_per_meter

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_transform(self, original_image: np.ndarray) -> np.ndarray:
        """Draw the four source calibration points on *original_image*.

        Useful for verifying that the BEV trapezoid aligns with the road
        markings before running the pipeline.

        Args:
            original_image: BGR camera frame to annotate.

        Returns:
            A new BGR array with red circles at each ``src_point`` and red
            lines connecting them to form the trapezoid outline.
        """
        vis = original_image.copy()
        pts = [(int(p[0]), int(p[1])) for p in self.config.src_points]
        RED = (0, 0, 255)

        for pt in pts:
            cv2.circle(vis, pt, radius=8, color=RED, thickness=-1)
            cv2.circle(vis, pt, radius=10, color=RED, thickness=2)

        # Draw trapezoid outline: connect points in order bl, br, tr, tl
        for i in range(len(pts)):
            cv2.line(vis, pts[i], pts[(i + 1) % len(pts)], RED, thickness=2)

        return vis


# ---------------------------------------------------------------------------
# Interactive calibration helper
# ---------------------------------------------------------------------------


def calibrate_bev_interactively(image_path: str) -> BEVConfig:
    """Collect four source points via mouse clicks and produce a :class:`BEVConfig`.

    Opens the image in an OpenCV window.  Click four points in this order:
    bottom-left, bottom-right, top-right, top-left (the road trapezoid).
    After the fourth click the window closes and the config is printed.

    Args:
        image_path: Path to a representative road image (ideally one with
            clear, straight lane markings).

    Returns:
        A :class:`BEVConfig` with the clicked source points and an
        automatically generated rectangular destination region.

    Raises:
        FileNotFoundError: When *image_path* cannot be read by OpenCV.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: '{image_path}'")

    h, w = image.shape[:2]
    clicked: List[List[int]] = []
    display = image.copy()

    def _on_mouse(event, x, y, flags, param):  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([x, y])
            cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow("BEV Calibration — click 4 road points", display)
            print(f"  Point {len(clicked)}: ({x}, {y})")
            if len(clicked) == 4:
                print("  4 points collected — press any key to continue.")

    cv2.namedWindow("BEV Calibration — click 4 road points")
    cv2.setMouseCallback("BEV Calibration — click 4 road points", _on_mouse)
    cv2.imshow("BEV Calibration — click 4 road points", display)

    print("\n=== BEV Calibration ===")
    print("Click the 4 road corners: bottom-left → bottom-right → top-right → top-left")

    while len(clicked) < 4:
        if cv2.waitKey(50) != -1:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Generate a sensible rectangular destination
    margin = int(w * 0.2)
    dst_points = [
        [margin,     h],
        [w - margin, h],
        [w - margin, 0],
        [margin,     0],
    ]

    cfg = BEVConfig(
        src_points=clicked,
        dst_points=dst_points,
        output_width=w,
        output_height=h,
        pixels_per_meter=30.0,
    )

    print("\n=== Paste into configs/lane_detection.yaml ===")
    print("bev_transform:")
    print("  src_points:")
    for p in clicked:
        print(f"    - [{p[0]}, {p[1]}]")
    print("  dst_points:")
    for p in dst_points:
        print(f"    - [{p[0]}, {p[1]}]")
    print(f"  output_width: {w}")
    print(f"  output_height: {h}")
    print("  pixels_per_meter: 30.0  # calibrate empirically")

    return cfg


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    SOURCE_DIR  = "rgb"
    OUTPUT_DIR  = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find first available image
    matches = sorted(glob.glob(str(Path(SOURCE_DIR) / "rgb_image_*.png")))
    if not matches:
        log.error("No images found in '%s/'. Nothing to do.", SOURCE_DIR)
        sys.exit(1)

    image_path = matches[0]
    image      = cv2.imread(image_path)
    if image is None:
        log.error("Could not read '%s'.", image_path)
        sys.exit(1)

    log.info("Using image: %s  (%dx%d)", image_path, image.shape[1], image.shape[0])

    # Load config and build transform
    try:
        cfg = BEVConfig.from_yaml("configs/lane_detection.yaml")
    except (FileNotFoundError, KeyError) as exc:
        log.error("Config error: %s", exc)
        sys.exit(1)

    bev = BEVTransform(cfg)

    # Save calibration overlay (src points drawn on original image)
    calib_img  = bev.visualize_transform(image)
    calib_path = OUTPUT_DIR / "bev_calibration_check.png"
    cv2.imwrite(str(calib_path), calib_img)
    log.info("Saved calibration overlay → %s", calib_path)

    # Save BEV-warped output
    bev_img    = bev.transform_image(image)
    bev_path   = OUTPUT_DIR / "bev_output_sample.png"
    cv2.imwrite(str(bev_path), bev_img)
    log.info("Saved BEV output        → %s", bev_path)

    # Print transform matrix
    M = bev.get_transform_matrix()
    print("\nTransform matrix M (camera → BEV):")
    for row in M:
        print("  " + "  ".join(f"{v:+.6f}" for v in row))
    print(f"\nConfig: {cfg.output_width}×{cfg.output_height} px, "
          f"{cfg.pixels_per_meter} px/m")
