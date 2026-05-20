"""NanoDet-Plus-m object detector wrapper for RoadSage scene understanding.

Runs in CPU-only mode via ONNX Runtime.  Detects relevant road-scene classes
(person, bicycle, car, motorcycle, bus, truck) and marks boxes that fall inside
the vehicle's path corridor as *path obstacles*.

Coordinate convention
---------------------
All bounding-box coordinates are in the pixel space of the **original** input
image (before any resize done by the preprocessor).

Usage::

    detector = ObjectDetector("configs/scene_understanding.yaml")
    result   = detector.detect(bgr_frame)

    for obj in result.path_obstacles:
        print(f"{obj.class_name} @ conf={obj.confidence:.2f}  {obj.bbox}")

    nearest = result.nearest_obstacle()
    if nearest:
        print(f"Closest obstacle: {nearest.class_name}")
"""

from __future__ import annotations

import glob
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in image pixel coordinates.

    Attributes:
        x1: Left edge (inclusive).
        y1: Top edge (inclusive).
        x2: Right edge (exclusive).
        y2: Bottom edge (exclusive).
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        """Return the width of the box in pixels."""
        return self.x2 - self.x1

    def height(self) -> int:
        """Return the height of the box in pixels."""
        return self.y2 - self.y1

    def area(self) -> int:
        """Return the area of the box in square pixels."""
        return self.width() * self.height()

    def center(self) -> Tuple[int, int]:
        """Return the ``(cx, cy)`` centre pixel of the box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def is_in_corridor(
        self,
        image_width: int,
        corridor_fraction: float = 0.4,
    ) -> bool:
        """Return True when the box centre lies inside the path corridor.

        The path corridor is a vertical strip centred on the image spanning
        ``corridor_fraction`` of the total image width.

        Args:
            image_width:        Pixel width of the full camera frame.
            corridor_fraction:  Fraction of image width covered by the
                corridor (default 0.4 = central 40 %).

        Returns:
            ``True`` if the bounding-box centre x-coordinate falls inside
            ``[corridor_left, corridor_right]``.
        """
        corridor_left = image_width * (0.5 - corridor_fraction / 2.0)
        corridor_right = image_width * (0.5 + corridor_fraction / 2.0)
        box_center_x = (self.x1 + self.x2) / 2.0
        return corridor_left <= box_center_x <= corridor_right

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x1={self.x1}, y1={self.y1}, "
            f"x2={self.x2}, y2={self.y2}, "
            f"w={self.width()}, h={self.height()})"
        )


# ---------------------------------------------------------------------------
# DetectedObject
# ---------------------------------------------------------------------------


@dataclass
class DetectedObject:
    """A single detection output from the NanoDet model.

    Attributes:
        bbox:             Bounding box in original image coordinates.
        class_id:         COCO class index.
        class_name:       Human-readable class label (e.g. ``"car"``).
        confidence:       Detection confidence in ``[0, 1]``.
        is_path_obstacle: ``True`` when the box centre lies inside the
            vehicle's path corridor.
    """

    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    is_path_obstacle: bool

    def __repr__(self) -> str:
        obs = " [OBSTACLE]" if self.is_path_obstacle else ""
        return (
            f"DetectedObject({self.class_name}, conf={self.confidence:.2f}, "
            f"{self.bbox}{obs})"
        )


# ---------------------------------------------------------------------------
# ObjectDetectionResult
# ---------------------------------------------------------------------------


@dataclass
class ObjectDetectionResult:
    """Structured output of a single NanoDet inference pass.

    Attributes:
        detections:        All detections that passed the confidence and
            class filters after NMS.
        path_obstacles:    Subset of *detections* whose box centre falls
            inside the path corridor.
        inference_time_ms: Wall-clock duration of the full ``detect()`` call
            (pre-processing + ONNX inference + post-processing) in ms.
        image_shape:       ``(H, W, C)`` of the image passed to ``detect()``.
    """

    detections: List[DetectedObject] = field(default_factory=list)
    path_obstacles: List[DetectedObject] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: Tuple[int, int, int] = (0, 0, 0)

    def nearest_obstacle(self) -> Optional[DetectedObject]:
        """Return the path obstacle with the largest bounding-box area.

        A larger bounding box generally indicates an object that is closer to
        the camera.  Returns ``None`` when there are no path obstacles.
        """
        if not self.path_obstacles:
            return None
        return max(self.path_obstacles, key=lambda o: o.bbox.area())


# ---------------------------------------------------------------------------
# NanoDetPreprocessor
# ---------------------------------------------------------------------------


class NanoDetPreprocessor:
    """Prepare a BGR camera frame for NanoDet-Plus-m inference.

    The preprocessor resizes the image to the model's fixed input size,
    normalises pixel values to ``[0, 1]``, converts to float32, and transposes
    to the NCHW layout expected by ONNX Runtime.

    Args:
        input_size: ``(width, height)`` of the model input tensor.
    """

    def __init__(self, input_size: Tuple[int, int] = (416, 416)) -> None:
        self._input_size = input_size  # (W, H)

    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Resize, normalise, and transpose an image for ONNX inference.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            A tuple ``(blob, scale_factor)`` where *blob* has shape
            ``(1, 3, input_H, input_W)`` with dtype float32 and
            *scale_factor* converts model-space coordinates back to the
            original image pixel space.
        """
        target_w, target_h = self._input_size
        orig_h, orig_w = image.shape[:2]

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))       # HWC → CHW
        blob = np.expand_dims(blob, axis=0)        # CHW → 1CHW

        # Use the width ratio as the single scale factor (assumes square input)
        scale_factor = orig_w / target_w
        return blob, scale_factor


# ---------------------------------------------------------------------------
# NanoDetPostprocessor
# ---------------------------------------------------------------------------


class NanoDetPostprocessor:
    """Decode NanoDet-Plus-m raw output into structured detections.

    The model produces a tensor of shape ``(1, num_boxes, 4 + num_classes)``
    where the first four values per row are ``[x1, y1, x2, y2]`` in model
    input coordinates, and the remaining values are class scores.

    Args:
        conf_threshold:      Minimum class confidence to keep a detection.
        nms_iou_threshold:   IoU threshold for greedy NMS suppression.
        relevant_classes:    Set of COCO class IDs to keep.  ``None`` means
            keep all classes.
        class_names:         Mapping from class ID to human-readable label.
        corridor_fraction:   Path corridor width as a fraction of image width.
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.45,
        relevant_classes: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
        corridor_fraction: float = 0.4,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._relevant_classes = set(relevant_classes) if relevant_classes else None
        self._class_names = class_names or {}
        self._corridor_fraction = corridor_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def postprocess(
        self,
        raw_output: np.ndarray,
        scale_factor: float,
        original_shape: Tuple[int, int, int],
    ) -> List[DetectedObject]:
        """Decode raw ONNX output into a filtered, NMS-suppressed list.

        Args:
            raw_output:     Model output tensor of shape
                ``(1, num_boxes, 4 + num_classes)``.
            scale_factor:   Multiply model-space coords by this to recover
                original image pixel coordinates.
            original_shape: ``(H, W, C)`` of the original input image.

        Returns:
            A list of :class:`DetectedObject` instances after confidence
            filtering, class filtering, and greedy NMS.
        """
        if raw_output.ndim != 3 or raw_output.shape[0] != 1:
            log.warning("Unexpected raw_output shape: %s", raw_output.shape)
            return []

        boxes_raw = raw_output[0]          # (num_boxes, 4 + num_classes)
        image_w = original_shape[1]

        candidates: List[DetectedObject] = []

        for row in boxes_raw:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self._conf_threshold:
                continue
            if self._relevant_classes and class_id not in self._relevant_classes:
                continue

            x1 = int(row[0] * scale_factor)
            y1 = int(row[1] * scale_factor)
            x2 = int(row[2] * scale_factor)
            y2 = int(row[3] * scale_factor)

            # Clamp to image bounds
            orig_h, orig_w = original_shape[:2]
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(x1 + 1, min(x2, orig_w))
            y2 = max(y1 + 1, min(y2, orig_h))

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            in_corridor = bbox.is_in_corridor(image_w, self._corridor_fraction)
            class_name = self._class_names.get(class_id, str(class_id))

            candidates.append(
                DetectedObject(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    is_path_obstacle=in_corridor,
                )
            )

        return self._apply_nms(candidates)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_nms(
        self, detections: List[DetectedObject]
    ) -> List[DetectedObject]:
        """Apply greedy NMS to a list of detections.

        Detections are sorted by confidence descending.  Any detection that
        has IoU > ``nms_iou_threshold`` with an already-kept detection is
        suppressed.

        Args:
            detections: Candidate detections before suppression.

        Returns:
            Filtered list with overlapping boxes removed.
        """
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[DetectedObject] = []

        for candidate in sorted_dets:
            suppressed = False
            for accepted in kept:
                if self._compute_iou(candidate.bbox, accepted.bbox) > self._nms_iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(candidate)

        return kept

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute intersection-over-union between two bounding boxes.

        Args:
            box1: First bounding box.
            box2: Second bounding box.

        Returns:
            IoU value in ``[0, 1]``.  Returns ``0.0`` when there is no
            intersection.
        """
        inter_x1 = max(box1.x1, box2.x1)
        inter_y1 = max(box1.y1, box2.y1)
        inter_x2 = min(box1.x2, box2.x2)
        inter_y2 = min(box1.y2, box2.y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = box1.area() + box2.area() - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------


class ObjectDetector:
    """High-level wrapper around the NanoDet-Plus-m ONNX model.

    Handles config loading, model initialisation, preprocessing, ONNX
    inference, and postprocessing in a single ``detect()`` call.

    Args:
        config_path: Path to the scene understanding YAML config file.
            Reads the ``object_detector`` section.

    Example::

        detector = ObjectDetector("configs/scene_understanding.yaml")
        result   = detector.detect(bgr_frame)
        if result.nearest_obstacle():
            print("Obstacle detected — consider stopping")
    """

    def __init__(self, config_path: str = "configs/scene_understanding.yaml") -> None:
        self._model_loaded = False
        self._session: Optional[ort.InferenceSession] = None  # type: ignore[type-arg]
        self._input_name: str = ""

        try:
            with open(config_path, encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh)
            cfg = full_cfg.get("object_detector", full_cfg)
        except FileNotFoundError:
            log.error("Config not found: %s", config_path)
            cfg = {}

        input_size: Tuple[int, int] = tuple(cfg.get("input_size", [416, 416]))  # type: ignore[assignment]
        conf_thr    = float(cfg.get("confidence_threshold", 0.5))
        nms_thr     = float(cfg.get("nms_iou_threshold", 0.45))
        corridor    = float(cfg.get("path_corridor_fraction", 0.4))

        # class_names may be stored as int keys after YAML parsing
        raw_names   = cfg.get("class_names", {})
        class_names: Dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}

        relevant = cfg.get("relevant_classes")
        if relevant:
            relevant = [int(c) for c in relevant]

        self._preprocessor  = NanoDetPreprocessor(input_size=input_size)
        self._postprocessor = NanoDetPostprocessor(
            conf_threshold=conf_thr,
            nms_iou_threshold=nms_thr,
            relevant_classes=relevant,
            class_names=class_names,
            corridor_fraction=corridor,
        )

        onnx_path = cfg.get("onnx_path", "models/object_detector.onnx")
        self._load_model(onnx_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> ObjectDetectionResult:
        """Run detection on a single BGR camera frame.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            An :class:`ObjectDetectionResult`.  When the model is not loaded
            or an error occurs, returns an empty result with
            ``inference_time_ms = 0``.
        """
        if not self._model_loaded:
            return ObjectDetectionResult(image_shape=image.shape[:3] if image.ndim == 3 else (0, 0, 0))  # type: ignore[arg-type]

        t0 = time.perf_counter()
        try:
            blob, scale_factor = self._preprocessor.preprocess(image)
            raw = self._session.run(None, {self._input_name: blob})[0]
            detections = self._postprocessor.postprocess(
                raw, scale_factor, image.shape  # type: ignore[arg-type]
            )
        except Exception as exc:
            log.error("Inference error: %s", exc, exc_info=True)
            return ObjectDetectionResult(image_shape=image.shape[:3])  # type: ignore[arg-type]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        path_obstacles = [d for d in detections if d.is_path_obstacle]

        return ObjectDetectionResult(
            detections=detections,
            path_obstacles=path_obstacles,
            inference_time_ms=elapsed_ms,
            image_shape=image.shape,  # type: ignore[arg-type]
        )

    def detect_from_path(self, image_path: str) -> ObjectDetectionResult:
        """Load an image from disk and run detection.

        Args:
            image_path: Filesystem path to a BGR image file.

        Returns:
            An :class:`ObjectDetectionResult`.  Returns an empty result if
            the image cannot be read.
        """
        image = cv2.imread(image_path)
        if image is None:
            log.warning("Could not read image: %s", image_path)
            return ObjectDetectionResult()
        return self.detect(image)

    def is_ready(self) -> bool:
        """Return ``True`` when the ONNX model has been loaded successfully."""
        return self._model_loaded

    def warmup(self) -> float:
        """Run one inference pass on a blank image to warm up the ONNX session.

        Returns:
            Inference time of the warmup pass in milliseconds.  Returns ``0.0``
            if the model is not loaded.
        """
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detect(blank)
        return result.inference_time_ms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, onnx_path: str) -> None:
        """Load the ONNX model from disk.

        Sets ``_model_loaded = True`` on success.  Logs a warning and leaves
        the detector in degraded mode if the file is missing or ONNX Runtime
        is unavailable.

        Args:
            onnx_path: Filesystem path to the ``.onnx`` weight file.
        """
        if not _ORT_AVAILABLE:
            log.warning("onnxruntime not installed — object detector disabled.")
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
            log.info("NanoDet model loaded: %s", onnx_path)
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

    # Allow running as  python -m app.scene_understanding.object_detector
    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    detector = ObjectDetector("configs/scene_understanding.yaml")
    log.info("Model loaded: %s", detector.is_ready())
    if not detector.is_ready():
        log.info("Expected — ONNX weights not yet downloaded")

    images = sorted(glob.glob("rgb/rgb_image_*.png"))[:5]
    if not images:
        log.warning("No images found in rgb/. Run from the project root.")
        sys.exit(0)

    for img_path in images:
        result = detector.detect_from_path(img_path)
        print(
            f"{Path(img_path).name:25s}  "
            f"detections={len(result.detections):2d}  "
            f"path_obstacles={len(result.path_obstacles):2d}  "
            f"inference={result.inference_time_ms:6.1f}ms"
        )
        nearest = result.nearest_obstacle()
        if nearest:
            print(f"  └─ nearest obstacle: {nearest}")
