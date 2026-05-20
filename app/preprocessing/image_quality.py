"""Lightweight image quality checks for inline use during data collection.

Designed to run before saving a frame from the camera stream — no heavy
dependencies, no I/O beyond a single ``cv2.imread`` in the path-based helpers.

Typical usage (inline)::

    checker = QualityChecker(config["quality_filters"])
    result = checker.check(bgr_frame)
    if not result.passed:
        log.debug("Frame dropped: %s", result.rejection_reason)

Typical usage (batch audit)::

    results = checker.batch_check("data/mnnit/raw")
    print(checker.summary(results))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)

except ImportError:  # tqdm is optional; degrade gracefully for inline use
    def _progress(iterable, **kwargs):
        return iterable


_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QualityCheckResult:
    """Outcome of a single image quality check pass.

    All numeric scores are always populated, even when the image fails early,
    so callers can log or plot the distribution of rejected images without
    re-running checks.

    Attributes:
        passed: ``True`` when the image clears all enabled gates.
        blur_score: Laplacian variance of the grayscale image.  Higher values
            mean sharper edges.  Typical threshold: 50.
        brightness_score: Mean grayscale pixel value in ``[0, 255]``.
        road_coverage_ratio: Fraction of pixels that match the road/asphalt
            HSV mask (low-saturation, mid-value gray tones).
        rejection_reason: Name of the first failing gate (e.g.
            ``"blur"``, ``"brightness"``, ``"road_coverage"``), or ``None``
            when the image passes all gates.
    """

    passed: bool
    blur_score: float
    brightness_score: float
    road_coverage_ratio: float
    rejection_reason: Optional[str]


# ---------------------------------------------------------------------------
# Internal score helpers (pure computation, no early-exit logic)
# ---------------------------------------------------------------------------


def _score_blur(gray: np.ndarray) -> float:
    """Return Laplacian variance of *gray*."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _score_brightness(gray: np.ndarray) -> float:
    """Return mean pixel value of *gray*."""
    return float(gray.mean())


def _score_road_coverage(image: np.ndarray) -> float:
    """Return fraction of pixels matching the road/asphalt HSV mask.

    The mask targets low-saturation (S: 0–50), mid-value (V: 30–200) pixels
    across all hues — characteristic of gray asphalt under varying lighting.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 30], dtype=np.uint8)
    upper = np.array([180, 50, 200], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    total_pixels = image.shape[0] * image.shape[1]
    road_pixels = int(mask.sum()) // 255
    return road_pixels / total_pixels


# ---------------------------------------------------------------------------
# Public gate function
# ---------------------------------------------------------------------------


def fast_quality_check(
    image: np.ndarray,
    blur_threshold: float = 50.0,
    brightness_min: float = 30.0,
    brightness_max: float = 220.0,
    road_min: float = 0.20,
) -> QualityCheckResult:
    """Run blur → brightness → road_coverage gates with early-exit on failure.

    All three scores are always computed and stored in the result so that the
    caller can record them for distribution analysis regardless of outcome.
    The gate that first fails sets ``rejection_reason``; subsequent gates are
    still scored but do not short-circuit scoring.

    Args:
        image: BGR image array as returned by ``cv2.imread``.
        blur_threshold: Minimum Laplacian variance.  Images with lower variance
            are too blurry.
        brightness_min: Minimum acceptable mean grayscale pixel value (0–255).
        brightness_max: Maximum acceptable mean grayscale pixel value (0–255).
        road_min: Minimum fraction of pixels that must match the road mask.

    Returns:
        A :class:`QualityCheckResult` with all scores populated and
        ``passed`` / ``rejection_reason`` set accordingly.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = _score_blur(gray)
    brightness_score = _score_brightness(gray)
    road_ratio = _score_road_coverage(image)

    rejection_reason: Optional[str] = None

    if blur_score < blur_threshold:
        rejection_reason = "blur"
    elif not (brightness_min <= brightness_score <= brightness_max):
        rejection_reason = "brightness"
    elif road_ratio < road_min:
        rejection_reason = "road_coverage"

    return QualityCheckResult(
        passed=rejection_reason is None,
        blur_score=blur_score,
        brightness_score=brightness_score,
        road_coverage_ratio=road_ratio,
        rejection_reason=rejection_reason,
    )


# ---------------------------------------------------------------------------
# Dataset stats helper
# ---------------------------------------------------------------------------


def compute_image_stats(image: np.ndarray) -> dict:
    """Compute descriptive statistics for a single image.

    Intended for dataset exploration and per-image reporting rather than
    real-time use.

    Args:
        image: BGR image array as returned by ``cv2.imread``.

    Returns:
        A dictionary with the following keys:

        * ``width`` (int) – image width in pixels
        * ``height`` (int) – image height in pixels
        * ``channels`` (int) – number of colour channels (1 or 3)
        * ``mean_brightness`` (float) – mean grayscale pixel value
        * ``std_brightness`` (float) – standard deviation of grayscale values
        * ``blur_score`` (float) – Laplacian variance (sharpness)
        * ``is_color`` (bool) – ``True`` when the image has 3 channels
        * ``dominant_color_hsv`` (tuple[float, float, float]) – HSV values of
          the most common colour, found via k-means (k=1) on a down-sampled
          version of the image to keep cost low
    """
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    is_color = channels == 3

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image
    mean_brightness = float(gray.mean())
    std_brightness = float(gray.std())
    blur_score = _score_blur(gray)

    # Dominant colour via k-means on small sample to keep it cheap
    small = cv2.resize(image, (64, 64)) if is_color else image
    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV) if is_color else small
    pixels = hsv_small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    dominant_hsv = tuple(float(v) for v in centers[0])

    return {
        "width": w,
        "height": h,
        "channels": channels,
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "blur_score": blur_score,
        "is_color": is_color,
        "dominant_color_hsv": dominant_hsv,
    }


# ---------------------------------------------------------------------------
# QualityChecker class
# ---------------------------------------------------------------------------


class QualityChecker:
    """Stateful quality checker that reads thresholds once from config.

    Accepts the ``quality_filters`` section of ``configs/production.yaml``
    so callers do not have to thread threshold values through every call site.

    Args:
        config: Dictionary matching the ``quality_filters`` YAML section::

            quality_filters:
              blur_laplacian_threshold: 50
              brightness_min: 30
              brightness_max: 220
              road_coverage_min: 0.20

        Missing keys fall back to the same defaults as
        :func:`fast_quality_check`.

    Example::

        with open("configs/production.yaml") as fh:
            cfg = yaml.safe_load(fh)
        checker = QualityChecker(cfg["quality_filters"])
        result = checker.check(frame)
    """

    def __init__(self, config: dict) -> None:
        self._blur_threshold = float(config.get("blur_laplacian_threshold", 50.0))
        self._brightness_min = float(config.get("brightness_min", 30.0))
        self._brightness_max = float(config.get("brightness_max", 220.0))
        self._road_min = float(config.get("road_coverage_min", 0.20))

    def check(self, image: np.ndarray) -> QualityCheckResult:
        """Run all quality gates on *image* and return the result.

        Args:
            image: BGR image array as returned by ``cv2.imread``.

        Returns:
            A :class:`QualityCheckResult` with all scores populated.
        """
        return fast_quality_check(
            image,
            blur_threshold=self._blur_threshold,
            brightness_min=self._brightness_min,
            brightness_max=self._brightness_max,
            road_min=self._road_min,
        )

    def check_from_path(self, image_path: str) -> QualityCheckResult:
        """Load *image_path* from disk and run quality gates.

        Args:
            image_path: Filesystem path to a JPEG or PNG image.

        Returns:
            A :class:`QualityCheckResult`.

        Raises:
            FileNotFoundError: When *image_path* does not exist.
            ValueError: When ``cv2.imread`` cannot decode the file.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: '{image_path}'")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cv2.imread could not decode '{image_path}'.")
        return self.check(image)

    def batch_check(self, image_dir: str) -> list[QualityCheckResult]:
        """Run quality gates on every image in *image_dir*.

        Searches non-recursively for files with extensions in
        ``{.png, .jpg, .jpeg}``.  Unreadable files are skipped.

        Args:
            image_dir: Directory to scan.

        Returns:
            A list of :class:`QualityCheckResult` objects, one per readable
            image found, in filesystem iteration order.
        """
        dir_path = Path(image_dir)
        candidates = [
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
        results: list[QualityCheckResult] = []
        for img_path in _progress(candidates, desc="Quality check", unit="img"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            results.append(self.check(image))
        return results

    def summary(self, results: list[QualityCheckResult]) -> dict:
        """Aggregate pass/fail statistics over a list of results.

        Args:
            results: Output of :meth:`batch_check` or a manually assembled
                list of :class:`QualityCheckResult` instances.

        Returns:
            A dictionary with the following keys:

            * ``total`` (int) – number of results
            * ``passed`` (int) – images that cleared all gates
            * ``failed`` (int) – images that failed at least one gate
            * ``acceptance_rate`` (float) – ``passed / total`` (0.0 if empty)
            * ``rejected_blur`` (int) – images rejected by the blur gate
            * ``rejected_brightness`` (int) – images rejected by brightness
            * ``rejected_road_coverage`` (int) – images rejected by road gate
            * ``mean_blur_score`` (float) – average blur score across all images
            * ``mean_brightness`` (float) – average brightness across all images
            * ``mean_road_coverage`` (float) – average road coverage ratio
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "acceptance_rate": 0.0,
                "rejected_blur": 0,
                "rejected_brightness": 0,
                "rejected_road_coverage": 0,
                "mean_blur_score": 0.0,
                "mean_brightness": 0.0,
                "mean_road_coverage": 0.0,
            }

        passed = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "acceptance_rate": passed / total,
            "rejected_blur": sum(1 for r in results if r.rejection_reason == "blur"),
            "rejected_brightness": sum(1 for r in results if r.rejection_reason == "brightness"),
            "rejected_road_coverage": sum(1 for r in results if r.rejection_reason == "road_coverage"),
            "mean_blur_score": float(np.mean([r.blur_score for r in results])),
            "mean_brightness": float(np.mean([r.brightness_score for r in results])),
            "mean_road_coverage": float(np.mean([r.road_coverage_ratio for r in results])),
        }


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------


def visualize_quality_check(
    image: np.ndarray,
    result: QualityCheckResult,
) -> np.ndarray:
    """Overlay quality check results on *image* and draw a coloured border.

    The returned image is a new array; *image* is not modified.

    Overlay contents:
        * Top-left text block: blur score, brightness, road coverage ratio
        * Centred label: ``PASSED`` (green) or ``FAILED: <reason>`` (red)
        * Border: 6-pixel green (passed) or red (failed)

    Args:
        image: BGR source image array.
        result: The :class:`QualityCheckResult` for this image.

    Returns:
        A new BGR ``uint8`` array the same size as *image* with overlays
        applied.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    border_color = (0, 200, 0) if result.passed else (0, 0, 220)
    border_thickness = 6
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border_color, border_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, w / 1280)
    line_h = int(22 * font_scale * (w / 640))
    shadow = (0, 0, 0)
    text_color = (255, 255, 255)

    stats_lines = [
        f"Blur:       {result.blur_score:7.1f}",
        f"Brightness: {result.brightness_score:7.1f}",
        f"Road cov:   {result.road_coverage_ratio:.3f}",
    ]
    margin = border_thickness + 4
    for i, line in enumerate(stats_lines):
        y = margin + line_h + i * line_h
        cv2.putText(vis, line, (margin + 1, y + 1), font, font_scale, shadow, 1, cv2.LINE_AA)
        cv2.putText(vis, line, (margin, y), font, font_scale, text_color, 1, cv2.LINE_AA)

    if result.passed:
        label = "PASSED"
        label_color = (0, 230, 0)
    else:
        label = f"FAILED: {result.rejection_reason}"
        label_color = (0, 0, 230)

    label_scale = font_scale * 1.3
    (lw, lh), _ = cv2.getTextSize(label, font, label_scale, 2)
    lx = max(margin, (w - lw) // 2)
    ly = h - margin - lh
    cv2.putText(vis, label, (lx + 1, ly + 1), font, label_scale, shadow, 2, cv2.LINE_AA)
    cv2.putText(vis, label, (lx, ly), font, label_scale, label_color, 2, cv2.LINE_AA)

    return vis
