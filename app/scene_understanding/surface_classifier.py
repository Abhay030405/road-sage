"""Road surface classifier for RoadSage scene understanding.

Classifies the road surface directly ahead of the vehicle into one of four
categories: clean, pothole, speed_breaker, or waterlogged.  Runs in CPU-only
mode via ONNX Runtime using a MobileNetV2-based backbone.

This model is **optional** in Phase 3 — the rest of the pipeline degrades
gracefully when the ONNX weights are not present.

Usage::

    classifier = SurfaceClassifier("configs/scene_understanding.yaml")

    # Classify the full frame (uses a bottom-centre road patch internally)
    result = classifier.classify_center_patch(bgr_frame)
    print(result.surface_class.value, result.confidence)

    if result.is_hazard():
        print("Road hazard detected — reduce speed")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
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

# ImageNet normalisation constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# SurfaceClass
# ---------------------------------------------------------------------------


class SurfaceClass(str, Enum):
    """Road surface condition labels.

    Values are lowercase strings suitable for logging and JSON serialisation.
    """

    CLEAN        = "clean"
    POTHOLE      = "pothole"
    SPEED_BREAKER = "speed_breaker"
    WATERLOGGED  = "waterlogged"
    UNKNOWN      = "unknown"


# ---------------------------------------------------------------------------
# SurfaceClassificationResult
# ---------------------------------------------------------------------------


@dataclass
class SurfaceClassificationResult:
    """Output of a single road surface classification pass.

    Attributes:
        surface_class:     Predicted :class:`SurfaceClass`.
        confidence:        Softmax probability of the predicted class in
            ``[0, 1]``.
        inference_time_ms: Wall-clock duration of the full ``classify()``
            call in milliseconds.
    """

    surface_class: SurfaceClass = SurfaceClass.UNKNOWN
    confidence: float = 0.0
    inference_time_ms: float = 0.0

    def is_hazard(self) -> bool:
        """Return ``True`` for hazardous surface conditions.

        A surface is considered a hazard when it is a pothole or waterlogged,
        as both conditions can cause the vehicle to lose traction or sustain
        damage.

        Returns:
            ``True`` for :attr:`SurfaceClass.POTHOLE` and
            :attr:`SurfaceClass.WATERLOGGED`; ``False`` otherwise.
        """
        return self.surface_class in (SurfaceClass.POTHOLE, SurfaceClass.WATERLOGGED)

    def __repr__(self) -> str:
        hazard = " [HAZARD]" if self.is_hazard() else ""
        return (
            f"SurfaceClassificationResult("
            f"{self.surface_class.value}, conf={self.confidence:.2f}{hazard})"
        )


# ---------------------------------------------------------------------------
# SurfaceClassifier
# ---------------------------------------------------------------------------


class SurfaceClassifier:
    """High-level wrapper around the MobileNetV2 surface classifier ONNX model.

    Reads the ``surface_classifier`` section of the scene-understanding YAML
    config, optionally loads the ONNX model, and classifies BGR images.  The
    model is optional — when the ONNX file is absent the classifier returns
    :attr:`SurfaceClass.UNKNOWN` with zero confidence for every input.

    Args:
        config_path: Path to ``configs/scene_understanding.yaml``.

    Example::

        classifier = SurfaceClassifier()
        result = classifier.classify_center_patch(bgr_frame)
        if result.is_hazard():
            print(f"Hazard: {result.surface_class.value}")
    """

    _DEFAULT_CLASSES: List[str] = [
        "clean",
        "pothole",
        "speed_breaker",
        "waterlogged",
    ]

    def __init__(self, config_path: str = "configs/scene_understanding.yaml") -> None:
        self._model_loaded = False
        self._session: Optional[ort.InferenceSession] = None  # type: ignore[type-arg]
        self._input_name: str = ""
        self._input_size: Tuple[int, int] = (224, 224)
        self._conf_threshold: float = 0.6
        self._classes: List[str] = self._DEFAULT_CLASSES

        try:
            with open(config_path, encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh)
            cfg = full_cfg.get("surface_classifier", full_cfg)
        except FileNotFoundError:
            log.error("Config not found: %s", config_path)
            cfg = {}

        raw_size = cfg.get("input_size", [224, 224])
        self._input_size = (int(raw_size[0]), int(raw_size[1]))
        self._conf_threshold = float(cfg.get("confidence_threshold", 0.6))

        raw_classes = cfg.get("classes", self._DEFAULT_CLASSES)
        self._classes = [str(c) for c in raw_classes]

        # Build label → SurfaceClass mapping, falling back to UNKNOWN
        self._label_map: Dict[str, SurfaceClass] = {}
        for label in self._classes:
            try:
                self._label_map[label] = SurfaceClass(label)
            except ValueError:
                self._label_map[label] = SurfaceClass.UNKNOWN

        onnx_path = cfg.get("onnx_path", "models/surface_classifier.onnx")
        self._load_model(onnx_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image: np.ndarray) -> SurfaceClassificationResult:
        """Classify the road surface in a BGR image patch.

        Args:
            image: BGR uint8 array of any size.  Resized to ``input_size``
                internally.

        Returns:
            A :class:`SurfaceClassificationResult`.  Returns
            ``SurfaceClass.UNKNOWN`` with ``confidence=0.0`` when the model
            is not loaded.
        """
        if not self._model_loaded:
            return SurfaceClassificationResult()

        t0 = time.perf_counter()
        try:
            blob = self._preprocess(image)
            raw = self._session.run(None, {self._input_name: blob})[0]  # (1, num_classes)
            probs = self._softmax(raw[0])
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
        except Exception as exc:
            log.error("Surface classification error: %s", exc, exc_info=True)
            return SurfaceClassificationResult()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if confidence < self._conf_threshold:
            return SurfaceClassificationResult(
                surface_class=SurfaceClass.UNKNOWN,
                confidence=confidence,
                inference_time_ms=elapsed_ms,
            )

        label = self._classes[class_idx] if class_idx < len(self._classes) else "unknown"
        surface = self._label_map.get(label, SurfaceClass.UNKNOWN)

        return SurfaceClassificationResult(
            surface_class=surface,
            confidence=confidence,
            inference_time_ms=elapsed_ms,
        )

    def classify_center_patch(self, image: np.ndarray) -> SurfaceClassificationResult:
        """Extract the near-road region and classify the surface.

        Crops the bottom 40 % of the frame centred horizontally — this region
        captures the road surface directly ahead of the vehicle, which is the
        most relevant area for hazard detection.

        Args:
            image: Full BGR camera frame.

        Returns:
            A :class:`SurfaceClassificationResult` for the cropped region.
        """
        h, w = image.shape[:2]
        y_start = int(h * 0.6)       # bottom 40 % of the frame
        x_margin = int(w * 0.2)      # discard the outer 20 % on each side
        patch = image[y_start:h, x_margin: w - x_margin]
        if patch.size == 0:
            return SurfaceClassificationResult()
        return self.classify(patch)

    def is_ready(self) -> bool:
        """Return ``True`` when the ONNX model has been loaded successfully."""
        return self._model_loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalise, and format an image for ONNX inference.

        Args:
            image: BGR uint8 array.

        Returns:
            Float32 NCHW array of shape ``(1, 3, H, W)``.
        """
        target_w, target_h = self._input_size
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over a 1-D logits array.

        Args:
            logits: 1-D float array of raw class scores.

        Returns:
            Probability distribution with the same shape.
        """
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _load_model(self, onnx_path: str) -> None:
        """Load the ONNX model from disk.

        Logs an *info* message (not warning) when the file is absent because
        the surface classifier is optional in Phase 3.

        Args:
            onnx_path: Filesystem path to the ``.onnx`` weight file.
        """
        if not _ORT_AVAILABLE:
            log.info("onnxruntime not installed — surface classifier disabled.")
            return

        if not Path(onnx_path).exists():
            log.info(
                "Surface classifier ONNX not found at '%s' — running without it. "
                "Download with: bash models/download_models.sh",
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
            log.info("Surface classifier model loaded: %s", onnx_path)
        except Exception as exc:
            log.error("Failed to load ONNX model '%s': %s", onnx_path, exc)
