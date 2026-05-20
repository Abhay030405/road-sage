"""UFLDv2 (Ultra-Fast Lane Detection v2) model wrapper for RoadSage.

Runs in CPU-only mode via ONNX Runtime with a ResNet-18 backbone.
Source images: ``rgb/rgb_image_*.png``  (MNNIT campus road dataset).

Architecture note
-----------------
UFLDv2 treats lane detection as a row-wise classification problem.  For each
of the ``num_row_anchors`` horizontal slices and each of the ``num_lanes``
lane channels the model predicts a softmax distribution over ``num_grid_cells
+ 1`` classes.  The last class means "no lane present at this row".

Usage::

    detector = UFLDLaneDetector("configs/lane_detection.yaml")
    result   = detector.predict(bgr_frame)
    batch    = detector.predict_batch("rgb/", pattern="rgb_image_*.png")
"""

from __future__ import annotations

import glob
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LaneDetectionResult:
    """Structured output of a single UFLDv2 inference pass.

    All coordinate lists are in the pixel space of the *original* input image
    (before any resizing done by the preprocessor).

    Attributes:
        left_lane: Ordered ``(x, y)`` pixel coordinates for the left lane
            boundary, sorted from bottom to top of the image.
        right_lane: Same for the right lane boundary.
        center_lane: Middle lane coordinates when three or more lanes are
            detected; ``None`` otherwise.
        confidence: Per-lane confidence scores in ``[0, 1]``.  Ordering
            matches ``[left, right]`` for two lanes, ``[left, centre, right]``
            for three or more, and ``[0.0, conf]`` / ``[conf, 0.0]`` when only
            one lane passes the threshold.
        no_lanes_detected: ``True`` when zero lanes survive both the
            confidence threshold and the minimum-points filter.
        inference_time_ms: Wall-clock duration of the full ``predict()`` call
            (pre-processing + ONNX inference + post-processing) in
            milliseconds.
        original_image_shape: ``(H, W, C)`` of the image passed to
            ``predict()``.
    """

    left_lane: List[Tuple[int, int]] = field(default_factory=list)
    right_lane: List[Tuple[int, int]] = field(default_factory=list)
    center_lane: Optional[List[Tuple[int, int]]] = None
    confidence: List[float] = field(default_factory=list)
    no_lanes_detected: bool = True
    inference_time_ms: float = 0.0
    original_image_shape: Tuple[int, int, int] = (0, 0, 0)

    def detected_lane_count(self, min_points: int = 1) -> int:
        """Return the number of lanes that contain at least *min_points* points.

        Args:
            min_points: Minimum point count for a lane to be considered
                present.  Defaults to 1 (any non-empty lane counts).

        Returns:
            Count of lanes whose point list has length ``>= min_points``.
        """
        count = sum(
            1 for lane in (self.left_lane, self.right_lane)
            if len(lane) >= min_points
        )
        if self.center_lane is not None and len(self.center_lane) >= min_points:
            count += 1
        return count


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class UFLDPreprocessor:
    """Converts a raw BGR camera frame into a normalised ONNX-ready tensor.

    Pipeline:

    1. BGR → RGB colour conversion.
    2. Bilinear resize to ``(input_width, input_height)``.
    3. Per-channel normalisation with ImageNet mean and std.
    4. Cast to ``float32``, transpose to ``C×H×W``, add batch dimension
       → final shape ``(1, 3, H, W)``.

    Args:
        input_width: Model input width in pixels (default 800).
        input_height: Model input height in pixels (default 288).
        mean: Per-channel RGB mean for normalisation.
        std: Per-channel RGB standard deviation for normalisation.
    """

    def __init__(
        self,
        input_width: int = 800,
        input_height: int = 288,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self._w = input_width
        self._h = input_height
        self._mean = np.array(mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array(std  or [0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert *image* to a normalised ``(1, 3, H, W)`` float32 tensor.

        Args:
            image: BGR ``uint8`` array of any resolution, as returned by
                ``cv2.imread``.

        Returns:
            ``float32`` ndarray of shape ``(1, 3, H, W)`` ready to pass
            directly to ``onnxruntime.InferenceSession.run()``.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        normalised = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        chw = normalised.transpose(2, 0, 1)          # HWC → CHW
        return np.expand_dims(chw, axis=0)            # → (1, 3, H, W)


# ---------------------------------------------------------------------------
# Postprocessor
# ---------------------------------------------------------------------------


class UFLDPostprocessor:
    """Decodes raw UFLDv2 ONNX logits into structured lane coordinates.

    The ONNX output tensor has shape
    ``(1, num_lanes, num_row_anchors, num_grid_cells + 1)``.
    The last grid-cell index (``num_grid_cells``) is the "no lane at this row"
    class; indices ``0 … num_grid_cells - 1`` encode the horizontal position.

    Args:
        num_lanes: Lane channels in the model output (default 4).
        num_row_anchors: Vertical sample positions per lane (default 72).
        num_grid_cells: Horizontal quantisation bins (default 200).
        row_anchor_start: Fractional image height of the first row anchor,
            roughly the horizon line (default 0.42).
        row_anchor_end: Fractional image height of the last row anchor
            (default 1.0 — bottom edge).
        conf_threshold: Minimum per-lane confidence (fraction of anchors
            with a valid prediction) to retain a lane (default 0.75).
        min_points: Minimum number of valid anchor points for a lane to
            survive filtering (default 5).
    """

    def __init__(
        self,
        num_lanes: int = 4,
        num_row_anchors: int = 72,
        num_grid_cells: int = 200,
        row_anchor_start: float = 0.42,
        row_anchor_end: float = 1.0,
        conf_threshold: float = 0.75,
        min_points: int = 5,
    ) -> None:
        self._L  = num_lanes
        self._R  = num_row_anchors
        self._G  = num_grid_cells
        self._y0 = row_anchor_start
        self._y1 = row_anchor_end
        self._conf_thr   = conf_threshold
        self._min_points = min_points

    def postprocess(
        self,
        onnx_output: np.ndarray,
        original_shape: Tuple[int, int, int],
    ) -> LaneDetectionResult:
        """Decode *onnx_output* into a :class:`LaneDetectionResult`.

        For each lane and each row anchor the argmax over the grid-cell
        dimension gives the predicted horizontal position.  When argmax equals
        ``num_grid_cells`` the model predicts no lane at that row.

        Args:
            onnx_output: Raw model output of shape
                ``(1, num_lanes, num_row_anchors, num_grid_cells + 1)``.
            original_shape: ``(H, W, C)`` of the source image, used to map
                fractional coordinates back to pixel space.

        Returns:
            A fully populated :class:`LaneDetectionResult`.  Lanes that fall
            below ``conf_threshold`` or ``min_points`` are silently discarded.
        """
        orig_h, orig_w = original_shape[:2]
        output = onnx_output[0]  # strip batch dim → (L, R, G+1)

        # --- decode each lane channel ------------------------------------
        valid: list[tuple[list[tuple[int, int]], float]] = []

        for lane_idx in range(self._L):
            lane_logits = output[lane_idx]   # (R, G+1)
            points: list[tuple[int, int]] = []

            for row_idx in range(self._R):
                argmax = int(np.argmax(lane_logits[row_idx]))

                if argmax == self._G:
                    continue   # "no lane" class — skip this anchor

                x = int(argmax * (orig_w / self._G))
                frac_y = self._y0 + (row_idx / self._R) * (self._y1 - self._y0)
                y = int(frac_y * orig_h)
                points.append((x, y))

            # per-lane confidence: fraction of anchors with a valid hit
            confidence = len(points) / self._R

            if confidence < self._conf_thr:
                continue
            if len(points) < self._min_points:
                continue

            valid.append((points, confidence))

        if not valid:
            return LaneDetectionResult(
                no_lanes_detected=True,
                original_image_shape=original_shape,
            )

        # sort left-to-right by mean x-position
        valid.sort(key=lambda t: float(np.mean([p[0] for p in t[0]])))

        image_center_x = orig_w / 2.0

        if len(valid) == 1:
            pts, conf = valid[0]
            mean_x = float(np.mean([p[0] for p in pts]))
            # Single lane: left half of image → right_lane; right half → left_lane
            if mean_x < image_center_x:
                left_lane, right_lane = [], pts
                confidences = [0.0, conf]
            else:
                left_lane, right_lane = pts, []
                confidences = [conf, 0.0]
            center_lane: Optional[List[Tuple[int, int]]] = None

        elif len(valid) == 2:
            left_lane   = valid[0][0]
            right_lane  = valid[1][0]
            center_lane = None
            confidences = [valid[0][1], valid[1][1]]

        else:
            # 3+ lanes: leftmost = left, rightmost = right, middles = center
            left_lane  = valid[0][0]
            right_lane = valid[-1][0]
            mid_pts: list[tuple[int, int]] = []
            mid_confs: list[float] = []
            for pts, conf in valid[1:-1]:
                mid_pts.extend(pts)
                mid_confs.append(conf)
            mid_pts.sort(key=lambda p: p[1])          # order top→bottom
            center_lane = mid_pts or None
            confidences = [valid[0][1]] + mid_confs + [valid[-1][1]]

        return LaneDetectionResult(
            left_lane=left_lane,
            right_lane=right_lane,
            center_lane=center_lane,
            confidence=confidences,
            no_lanes_detected=False,
            original_image_shape=original_shape,
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class UFLDLaneDetector:
    """End-to-end UFLDv2 lane detector backed by ONNX Runtime.

    Reads ``configs/lane_detection.yaml`` to configure the preprocessor,
    postprocessor, and ONNX session.  If the model file is absent the
    detector degrades gracefully: :meth:`predict` always returns an empty
    :class:`LaneDetectionResult` with ``no_lanes_detected=True``.

    Args:
        config_path: Path to the lane-detection YAML config.

    Example::

        detector = UFLDLaneDetector()
        if detector.is_ready():
            result = detector.predict(frame)
            print(result.detected_lane_count())
        else:
            print("Model not loaded — run models/download_models.sh")
    """

    def __init__(self, config_path: str = "configs/lane_detection.yaml") -> None:
        self._model_loaded = False
        self._session: Optional[object] = None
        self._input_name: str = ""

        cfg = self._load_config(config_path)

        inp = cfg.get("input", {})
        self._preprocessor = UFLDPreprocessor(
            input_width=inp.get("width", 800),
            input_height=inp.get("height", 288),
            mean=inp.get("mean", [0.485, 0.456, 0.406]),
            std=inp.get("std",  [0.229, 0.224, 0.225]),
        )

        ra   = cfg.get("row_anchors", {})
        conf = cfg.get("confidence",  {})
        self._postprocessor = UFLDPostprocessor(
            num_lanes=ra.get("num_lanes", 4),
            num_row_anchors=ra.get("num_row_anchors", 72),
            num_grid_cells=ra.get("num_grid_cells", 200),
            row_anchor_start=ra.get("row_anchor_start", 0.42),
            row_anchor_end=ra.get("row_anchor_end", 1.0),
            conf_threshold=conf.get("lane_conf_threshold", 0.75),
            min_points=conf.get("min_points_per_lane", 5),
        )

        model_path = cfg.get("model", {}).get("onnx_path", "models/lane_detector.onnx")
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Return parsed YAML config, or ``{}`` if the file is absent."""
        path = Path(config_path)
        if not path.exists():
            log.warning("Config not found at '%s' — using defaults.", config_path)
            return {}
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _load_model(self, model_path: str) -> None:
        """Load the ONNX model from *model_path*.

        Sets ``self._model_loaded = True`` on success.  Logs a warning
        without raising if ORT is unavailable or the file is missing.
        """
        if not _ORT_AVAILABLE:
            log.warning(
                "onnxruntime is not installed — lane detection disabled. "
                "Install with: pip install onnxruntime"
            )
            return

        path = Path(model_path)
        if not path.exists():
            log.warning(
                "ONNX model not found at '%s'. "
                "Download with: bash models/download_models.sh",
                model_path,
            )
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._model_loaded = True
            log.info("UFLDv2 ONNX model loaded from '%s'.", model_path)
        except Exception as exc:
            log.warning("Failed to load ONNX model from '%s': %s", model_path, exc)

    def _empty_result(
        self,
        original_shape: Tuple[int, int, int] = (0, 0, 0),
        inference_time_ms: float = 0.0,
    ) -> LaneDetectionResult:
        """Return a safe all-empty :class:`LaneDetectionResult`."""
        return LaneDetectionResult(
            left_lane=[],
            right_lane=[],
            center_lane=None,
            confidence=[],
            no_lanes_detected=True,
            inference_time_ms=inference_time_ms,
            original_image_shape=original_shape,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` when the ONNX model is loaded and ready."""
        return self._model_loaded

    def predict(self, image: np.ndarray) -> LaneDetectionResult:
        """Run lane detection on a single BGR image.

        Never raises: any inference error is caught and an empty result is
        returned so the calling pipeline is never interrupted.

        Args:
            image: BGR ``uint8`` array as returned by ``cv2.imread``.

        Returns:
            :class:`LaneDetectionResult` with coordinates in the original
            image pixel space.
        """
        original_shape: Tuple[int, int, int] = image.shape  # type: ignore[assignment]
        t0 = time.perf_counter()

        if not self._model_loaded:
            return self._empty_result(original_shape)

        try:
            tensor  = self._preprocessor.preprocess(image)
            outputs = self._session.run(None, {self._input_name: tensor})
            result  = self._postprocessor.postprocess(outputs[0], original_shape)
            result.inference_time_ms = (time.perf_counter() - t0) * 1_000.0
            return result
        except Exception as exc:
            log.warning("Inference error on image shape %s: %s", original_shape, exc)
            return self._empty_result(
                original_shape,
                inference_time_ms=(time.perf_counter() - t0) * 1_000.0,
            )

    def predict_from_path(self, image_path: str) -> LaneDetectionResult:
        """Load *image_path* from disk and run lane detection.

        Args:
            image_path: Filesystem path to a JPEG or PNG image.

        Returns:
            :class:`LaneDetectionResult`, or an empty result if the file
            cannot be decoded.
        """
        image = cv2.imread(image_path)
        if image is None:
            log.warning("Could not read image at '%s'.", image_path)
            return self._empty_result()
        return self.predict(image)

    def predict_batch(
        self,
        image_dir: str,
        pattern: str = "rgb_image_*.png",
    ) -> List[Tuple[str, LaneDetectionResult]]:
        """Run lane detection on all images matching *pattern* in *image_dir*.

        Images are processed in sorted filename order for reproducibility.

        Args:
            image_dir: Directory that contains the images.
            pattern: Glob pattern relative to *image_dir*.  Default matches
                the MNNIT naming convention ``rgb_image_*.png``.

        Returns:
            List of ``(filename, LaneDetectionResult)`` tuples in sorted
            filename order.
        """
        matches = sorted(glob.glob(str(Path(image_dir) / pattern)))
        if not matches:
            log.warning("No images found for pattern '%s' in '%s'.", pattern, image_dir)
            return []

        results: List[Tuple[str, LaneDetectionResult]] = []
        for img_path in tqdm(matches, desc="Lane detection", unit="img"):
            results.append((Path(img_path).name, self.predict_from_path(img_path)))
        return results

    def warmup(self) -> float:
        """Infer on a blank frame and return latency in ms.

        Primes the ONNX Runtime thread pool so the first real inference call
        is not penalised by JIT / thread-startup overhead.  Also useful as a
        health-check signal on startup.

        Returns:
            Inference latency in milliseconds, or ``0.0`` if not loaded.
        """
        if not self._model_loaded:
            return 0.0
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return self.predict(blank).inference_time_ms


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

    SOURCE_DIR = "rgb"
    PATTERN    = "rgb_image_*.png"

    detector = UFLDLaneDetector("configs/lane_detection.yaml")

    if not detector.is_ready():
        print(
            "\nModel weights not found — results will be empty.\n"
            "Download with:  bash models/download_models.sh\n"
        )

    batch = detector.predict_batch(SOURCE_DIR, pattern=PATTERN)

    if not batch:
        print(f"No images matched '{PATTERN}' in '{SOURCE_DIR}/'.")
        sys.exit(0)

    total_ms    = 0.0
    two_lane_ct = 0

    print(f"\n{'Filename':<32} {'Lanes':>5}  {'Confidence':<28} {'ms':>7}")
    print("─" * 76)

    for fname, result in batch:
        n        = result.detected_lane_count()
        conf_str = ", ".join(f"{c:.2f}" for c in result.confidence) or "—"
        ms       = result.inference_time_ms
        total_ms += ms
        if n >= 2:
            two_lane_ct += 1
        print(f"{fname:<32} {n:>5}  {conf_str:<28} {ms:>6.1f}")

    avg_ms = total_ms / len(batch)
    print("─" * 76)
    print(f"\n  Total images          : {len(batch)}")
    print(f"  Images with 2+ lanes  : {two_lane_ct}  ({two_lane_ct / len(batch):.1%})")
    print(f"  Avg inference time    : {avg_ms:.1f} ms")
