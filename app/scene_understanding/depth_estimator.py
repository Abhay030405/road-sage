"""MiDaS v2.1 Small depth estimator wrapper for RoadSage scene understanding.

Runs in CPU-only mode via ONNX Runtime.  MiDaS produces a **relative inverse
depth map** — higher pixel values indicate objects that are *closer* to the
camera.  The map is normalised to ``[0, 1]`` after inference and scaled back to
the original image resolution.

Coordinate convention
---------------------
All depth-map coordinates use the same ``(x, y)`` / ``(col, row)`` convention
as OpenCV.  ``depth_map[y, x]`` is the normalised inverse depth at pixel
``(x, y)``.

Usage::

    estimator = DepthEstimator("configs/scene_understanding.yaml")
    result    = estimator.estimate(bgr_frame)

    # Visualise — warm colours = close, dark colours = far
    coloured = estimator.visualize_depth(result)
    cv2.imwrite("outputs/depth.png", coloured)

    # Check whether a detected object is close to the vehicle
    if result.is_obstacle_close(detected_obj.bbox, threshold=0.7):
        print("Obstacle too close — consider stopping")
"""

from __future__ import annotations

import glob
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np
import yaml

if TYPE_CHECKING:
    from app.scene_understanding.object_detector import BoundingBox

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

log = logging.getLogger(__name__)

# ImageNet normalisation constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# DepthEstimationResult
# ---------------------------------------------------------------------------


@dataclass
class DepthEstimationResult:
    """Structured output of a single MiDaS inference pass.

    Attributes:
        depth_map:         ``(H, W)`` float32 array of relative inverse depth
            values in ``[0, 1]``.  Higher = closer to camera.
        normalized_map:    ``(H, W)`` uint8 array (``depth_map * 255``), ready
            for colourmap visualisation.
        inference_time_ms: Wall-clock duration of the full ``estimate()`` call
            (pre-processing + ONNX inference + post-processing) in ms.
        image_shape:       ``(H, W, C)`` of the image passed to ``estimate()``.
    """

    depth_map: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.float32)
    )
    normalized_map: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    inference_time_ms: float = 0.0
    image_shape: Tuple[int, int, int] = (0, 0, 0)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_depth_at_bbox(
        self,
        bbox: "BoundingBox",
        padding: int = 5,
    ) -> float:
        """Return the maximum inverse-depth value inside a bounding box ROI.

        A higher return value indicates the object is closer to the camera.
        The ROI is expanded by *padding* pixels on each side and clamped to
        the image bounds.

        Args:
            bbox:    Bounding box in original image coordinates.
            padding: Number of extra pixels added around the box.

        Returns:
            Maximum float32 depth value in the ROI, or ``0.0`` when the
            depth map is empty.
        """
        if self.depth_map.size == 0:
            return 0.0
        h, w = self.depth_map.shape[:2]
        x1 = max(0, bbox.x1 - padding)
        y1 = max(0, bbox.y1 - padding)
        x2 = min(w, bbox.x2 + padding)
        y2 = min(h, bbox.y2 + padding)
        roi = self.depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(roi.max())

    def get_depth_at_point(self, x: int, y: int, radius: int = 5) -> float:
        """Return the mean inverse-depth in a square window around a pixel.

        Args:
            x:      Column coordinate in the original image.
            y:      Row coordinate in the original image.
            radius: Half-size of the sampling window.  The window spans
                ``[x-radius, x+radius] × [y-radius, y+radius]``.

        Returns:
            Mean float32 depth value in the window, or ``0.0`` when the
            depth map is empty or the point is out of bounds.
        """
        if self.depth_map.size == 0:
            return 0.0
        h, w = self.depth_map.shape[:2]
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius + 1)
        y2 = min(h, y + radius + 1)
        window = self.depth_map[y1:y2, x1:x2]
        if window.size == 0:
            return 0.0
        return float(window.mean())

    def is_obstacle_close(
        self,
        bbox: "BoundingBox",
        threshold: float = 0.7,
    ) -> bool:
        """Return ``True`` when the depth at a bounding box exceeds *threshold*.

        A value of 0.7 (70 % of maximum relative depth) is a reasonable
        default for the MNNIT campus dataset; use
        :meth:`DepthEstimator.calibrate_threshold` to tune it for your
        camera and environment.

        Args:
            bbox:      Bounding box of the detected obstacle.
            threshold: Normalised inverse-depth threshold in ``[0, 1]``.

        Returns:
            ``True`` when the obstacle is within the danger zone.
        """
        return self.get_depth_at_bbox(bbox) > threshold


# ---------------------------------------------------------------------------
# MiDaSPreprocessor
# ---------------------------------------------------------------------------


class MiDaSPreprocessor:
    """Prepare a BGR camera frame for MiDaS Small inference.

    The preprocessor converts to RGB, resizes to the model's fixed input,
    applies ImageNet normalisation, converts to float32, and transposes to
    the NCHW layout expected by ONNX Runtime.

    Args:
        input_size: ``(width, height)`` of the MiDaS model input tensor.
    """

    def __init__(self, input_size: Tuple[int, int] = (384, 384)) -> None:
        self._input_size = input_size  # (W, H)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalise, and format an image for ONNX inference.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            Float32 NCHW array of shape ``(1, 3, input_H, input_W)``.
        """
        target_w, target_h = self._input_size

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Normalise with ImageNet stats
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC → NCHW
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob


# ---------------------------------------------------------------------------
# MiDaSPostprocessor
# ---------------------------------------------------------------------------


class MiDaSPostprocessor:
    """Decode and resize the raw MiDaS output back to the original image size.

    MiDaS produces a single-channel output whose spatial dimensions match the
    model's input size.  The postprocessor resizes it to the original frame
    dimensions and normalises to ``[0, 1]``.
    """

    def postprocess(
        self,
        raw_output: np.ndarray,
        original_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Resize and normalise the raw depth output.

        Args:
            raw_output:      Model output of shape ``(1, H_out, W_out)`` or
                ``(H_out, W_out)``.
            original_shape:  ``(H, W, C)`` of the original input image.

        Returns:
            Float32 depth map of shape ``(original_H, original_W)`` with
            values in ``[0, 1]``.  Higher values = closer to camera.
        """
        # Squeeze batch / channel dims
        depth = raw_output.squeeze()

        orig_h, orig_w = original_shape[:2]
        resized = cv2.resize(
            depth.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )

        # Normalise to [0, 1]
        d_min = resized.min()
        d_max = resized.max()
        normalised = (resized - d_min) / (d_max - d_min + 1e-8)
        return normalised.astype(np.float32)


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------


class DepthEstimator:
    """High-level wrapper around the MiDaS v2.1 Small ONNX model.

    Handles config loading, model initialisation, preprocessing, ONNX
    inference, and postprocessing in a single ``estimate()`` call.

    Args:
        config_path: Path to the scene understanding YAML config.
            Reads the ``depth_estimator`` section.

    Example::

        estimator = DepthEstimator("configs/scene_understanding.yaml")
        result    = estimator.estimate(bgr_frame)
        coloured  = estimator.visualize_depth(result)
        cv2.imwrite("outputs/depth.png", coloured)
    """

    def __init__(self, config_path: str = "configs/scene_understanding.yaml") -> None:
        self._model_loaded = False
        self._session: Optional[ort.InferenceSession] = None  # type: ignore[type-arg]
        self._input_name: str = ""

        try:
            with open(config_path, encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh)
            cfg = full_cfg.get("depth_estimator", full_cfg)
        except FileNotFoundError:
            log.error("Config not found: %s", config_path)
            cfg = {}

        input_size: Tuple[int, int] = tuple(cfg.get("input_size", [384, 384]))  # type: ignore[assignment]

        self._preprocessor  = MiDaSPreprocessor(input_size=input_size)
        self._postprocessor = MiDaSPostprocessor()

        onnx_path = cfg.get("onnx_path", "models/depth_estimator.onnx")
        self._load_model(onnx_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image: np.ndarray) -> DepthEstimationResult:
        """Run depth estimation on a single BGR camera frame.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            A :class:`DepthEstimationResult`.  When the model is not loaded
            or an error occurs, returns a zero-filled result.
        """
        h, w = image.shape[:2]
        zero_map = np.zeros((h, w), dtype=np.float32)

        if not self._model_loaded:
            return DepthEstimationResult(
                depth_map=zero_map,
                normalized_map=zero_map.astype(np.uint8),
                inference_time_ms=0.0,
                image_shape=image.shape,  # type: ignore[arg-type]
            )

        t0 = time.perf_counter()
        try:
            blob = self._preprocessor.preprocess(image)
            raw = self._session.run(None, {self._input_name: blob})[0]
            depth_map = self._postprocessor.postprocess(raw, image.shape)  # type: ignore[arg-type]
        except Exception as exc:
            log.error("Depth inference error: %s", exc, exc_info=True)
            return DepthEstimationResult(
                depth_map=zero_map,
                normalized_map=zero_map.astype(np.uint8),
                inference_time_ms=0.0,
                image_shape=image.shape,  # type: ignore[arg-type]
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        normalized_map = (depth_map * 255.0).astype(np.uint8)

        return DepthEstimationResult(
            depth_map=depth_map,
            normalized_map=normalized_map,
            inference_time_ms=elapsed_ms,
            image_shape=image.shape,  # type: ignore[arg-type]
        )

    def estimate_from_path(self, image_path: str) -> DepthEstimationResult:
        """Load an image from disk and run depth estimation.

        Args:
            image_path: Filesystem path to a BGR image file.

        Returns:
            A :class:`DepthEstimationResult`.  Returns a zero-filled result
            if the image cannot be read.
        """
        image = cv2.imread(image_path)
        if image is None:
            log.warning("Could not read image: %s", image_path)
            return DepthEstimationResult()
        return self.estimate(image)

    def visualize_depth(self, result: DepthEstimationResult) -> np.ndarray:
        """Render the depth map as a false-colour BGR image.

        Uses the MAGMA colourmap — warm colours (yellow/white) indicate
        close objects; dark colours (purple/black) indicate distant objects.

        Args:
            result: A :class:`DepthEstimationResult` from :meth:`estimate`.

        Returns:
            BGR uint8 array of the same spatial dimensions as the depth map.
        """
        return cv2.applyColorMap(result.normalized_map, cv2.COLORMAP_MAGMA)

    def is_ready(self) -> bool:
        """Return ``True`` when the ONNX model has been loaded successfully."""
        return self._model_loaded

    def warmup(self) -> float:
        """Run one inference pass on a blank image to warm up the ONNX session.

        Returns:
            Inference time in milliseconds, or ``0.0`` if the model is not
            loaded.
        """
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return self.estimate(blank).inference_time_ms

    def calibrate_threshold(
        self,
        image: np.ndarray,
        known_obstacle_bbox: "BoundingBox",
        known_distance_m: float,
    ) -> float:
        """Estimate the inverse-depth threshold that corresponds to a known distance.

        Runs one inference pass on *image*, samples the depth at the provided
        bounding box, and prints calibration guidance.

        Args:
            image:                BGR frame containing the known obstacle.
            known_obstacle_bbox:  Bounding box of the obstacle in the image.
            known_distance_m:     Real-world distance to the obstacle in metres.

        Returns:
            The raw inverse-depth value at the obstacle ROI.  Multiply by
            ``1.1`` for a 10 % safety margin.
        """
        result = self.estimate(image)
        depth_val = result.get_depth_at_bbox(known_obstacle_bbox)
        threshold_suggested = depth_val * 1.1
        print(
            f"Depth value {depth_val:.4f} corresponds to ~{known_distance_m:.1f} m"
        )
        print(
            f"Suggested stop threshold: {threshold_suggested:.4f} (10% safety margin)"
        )
        return depth_val

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, onnx_path: str) -> None:
        """Load the MiDaS ONNX model from disk.

        Sets ``_model_loaded = True`` on success.  Logs a warning and leaves
        the estimator in degraded (zero-depth) mode if the file is missing or
        ONNX Runtime is unavailable.

        Args:
            onnx_path: Filesystem path to the ``.onnx`` weight file.
        """
        if not _ORT_AVAILABLE:
            log.warning("onnxruntime not installed — depth estimator disabled.")
            return

        if not Path(onnx_path).exists():
            log.warning(
                "ONNX model not found at '%s'. Download with: bash models/download_models.sh",
                onnx_path,
            )
            return

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 1
            sess_opts.intra_op_num_threads = 2
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._model_loaded = True
            log.info("MiDaS model loaded: %s", onnx_path)
        except Exception as exc:
            log.error("Failed to load ONNX model '%s': %s", onnx_path, exc)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import os
    os.makedirs("outputs", exist_ok=True)

    estimator = DepthEstimator("configs/scene_understanding.yaml")
    log.info("Model loaded: %s", estimator.is_ready())
    if not estimator.is_ready():
        log.info("Expected — ONNX weights not yet downloaded")

    images = sorted(glob.glob("rgb/rgb_image_*.png"))[:3]
    if not images:
        log.warning("No images found in rgb/. Run from the project root.")
        sys.exit(0)

    for n, img_path in enumerate(images):
        result = estimator.estimate_from_path(img_path)
        coloured = estimator.visualize_depth(result)
        out_path = f"outputs/depth_{n + 1}.png"
        cv2.imwrite(out_path, coloured)
        print(
            f"{Path(img_path).name:25s}  "
            f"inference={result.inference_time_ms:6.1f}ms  "
            f"depth min={result.depth_map.min():.3f}  "
            f"max={result.depth_map.max():.3f}  "
            f"mean={result.depth_map.mean():.3f}  "
            f"→ {out_path}"
        )
