"""
app.decision.ml_fallback
=========================

MobileNetV3-Small decision CNN wrapped for CPU ONNX inference.

This module is the lowest-priority layer of the decision engine — invoked
only when the geometric pipeline returns ``None`` (insufficient lane evidence).
It supports both deterministic single-pass inference and Monte Carlo Dropout
uncertainty estimation via additive input noise.

No training code lives here.  Training is handled by
``training/trainers/train_decision.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from app.decision import DecisionPath, DecisionResult, DriveCommand

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class MLFallbackPreprocessor:
    """Converts a raw BGR image to a normalised NCHW float32 tensor.

    Args:
        input_size: ``(height, width)`` tuple for the model's expected input.
    """

    def __init__(self, input_size: Tuple[int, int] = (224, 224)) -> None:
        self._input_size = input_size  # (H, W)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a BGR image for the decision CNN.

        Steps:

        1. BGR → RGB.
        2. Resize to ``input_size`` with bilinear interpolation.
        3. Scale to [0, 1] and apply ImageNet normalisation.
        4. Reshape to NCHW float32 tensor.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            Float32 NCHW array of shape ``(1, 3, H, W)``.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = self._input_size
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        # HWC → CHW → NCHW
        return normalised.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class MLFallbackModel:
    """ONNX-backed MobileNetV3-Small decision classifier.

    Gracefully degrades when the model weights are not present — all
    prediction methods return safe defaults (STOP with zero confidence).

    Args:
        config_path: Path to ``configs/decision_engine.yaml``.
    """

    _CLASS_NAMES: List[str] = ["FORWARD", "LEFT", "RIGHT", "STOP"]

    def __init__(self, config_path: str = "configs/decision_engine.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        ml_cfg = raw["ml_fallback"]
        self._model_path: str = ml_cfg["model_path"]
        h, w = ml_cfg["input_size"]
        self._input_size: Tuple[int, int] = (int(h), int(w))
        self._num_classes: int = int(ml_cfg.get("num_classes", 4))
        self._mc_dropout_samples: int = int(ml_cfg.get("mc_dropout_samples", 10))
        self._uncertainty_threshold: float = float(
            ml_cfg.get("uncertainty_threshold", 0.3)
        )

        self._preprocessor = MLFallbackPreprocessor(self._input_size)
        self._session = None
        self._input_name: Optional[str] = None
        self._model_loaded: bool = False

        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the ONNX model; log a warning if weights are missing."""
        if not Path(self._model_path).exists():
            logger.warning(
                "ML fallback model not found at '%s'. "
                "Run training/trainers/train_decision.py to generate weights.",
                self._model_path,
            )
            return

        try:
            import onnxruntime as ort  # type: ignore[import]

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3  # ERROR only
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._model_loaded = True
            logger.info(
                "MLFallbackModel loaded: %s  input_name=%s",
                self._model_path,
                self._input_name,
            )
        except Exception as exc:
            logger.warning("Failed to load ML fallback model: %s", exc)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over the last axis."""
        shifted = logits - logits.max()
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum()

    def _run_session(self, blob: np.ndarray) -> np.ndarray:
        """Run one ONNX forward pass and return the softmax probabilities.

        Args:
            blob: NCHW float32 tensor of shape ``(1, 3, H, W)``.

        Returns:
            Float32 softmax array of shape ``(num_classes,)``.
        """
        outputs = self._session.run(None, {self._input_name: blob})
        logits = outputs[0].squeeze()        # (num_classes,)
        return self._softmax(logits).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, image: np.ndarray
    ) -> Tuple[DriveCommand, float, List[float]]:
        """Run a single deterministic forward pass.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            A tuple ``(command, confidence, softmax_probs)`` where
            *softmax_probs* is a list of length ``num_classes``.
            Returns ``(STOP, 0.0, [0.25, …])`` when the model is not loaded
            or an inference error occurs.
        """
        _safe = (DriveCommand.STOP, 0.0, [1.0 / self._num_classes] * self._num_classes)

        if not self._model_loaded:
            return _safe

        try:
            blob = self._preprocessor.preprocess(image)
            softmax = self._run_session(blob)
            class_id = int(np.argmax(softmax))
            confidence = float(softmax[class_id])
            command = DriveCommand.from_int(class_id)
            return command, confidence, softmax.tolist()
        except Exception as exc:
            logger.error("MLFallbackModel.predict error: %s", exc)
            return _safe

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        n_samples: int = 10,
    ) -> Tuple[DriveCommand, float, float, List[float]]:
        """Run MC Dropout inference via additive Gaussian input noise.

        Each of the ``n_samples`` forward passes receives a slightly
        different noisy version of the input tensor (σ = 0.05), simulating
        the stochasticity of dropout-enabled inference without requiring
        training-mode dropout layers in the ONNX graph.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.
            n_samples: Number of stochastic forward passes.

        Returns:
            A tuple ``(command, confidence, uncertainty, mean_softmax)``
            where *uncertainty* is the mean standard deviation across classes
            and *mean_softmax* is the averaged probability vector.
            Returns safe defaults ``(STOP, 0.0, 1.0, [0.25, …])`` when the
            model is not loaded.
        """
        _safe = (
            DriveCommand.STOP,
            0.0,
            1.0,
            [1.0 / self._num_classes] * self._num_classes,
        )

        if not self._model_loaded:
            return _safe

        try:
            base_blob = self._preprocessor.preprocess(image)  # (1, 3, H, W)
            softmax_samples: List[np.ndarray] = []

            for _ in range(n_samples):
                noise = np.random.normal(0, 0.05, base_blob.shape).astype(np.float32)
                noisy_blob = base_blob + noise
                softmax_samples.append(self._run_session(noisy_blob))

            arr = np.stack(softmax_samples, axis=0)         # (N, C)
            mean_softmax = arr.mean(axis=0)                  # (C,)
            std_softmax = arr.std(axis=0)                    # (C,)

            uncertainty = float(np.mean(std_softmax))
            class_id = int(np.argmax(mean_softmax))
            confidence = float(mean_softmax[class_id])
            command = DriveCommand.from_int(class_id)

            return command, confidence, uncertainty, mean_softmax.tolist()

        except Exception as exc:
            logger.error("MLFallbackModel.predict_with_uncertainty error: %s", exc)
            return _safe

    def predict_to_decision_result(
        self,
        image: np.ndarray,
        use_mc_dropout: bool = True,
    ) -> DecisionResult:
        """Produce a full :class:`~app.decision.DecisionResult` from an image.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.
            use_mc_dropout: When ``True``, use multi-sample uncertainty
                estimation.  When ``False``, run a single deterministic pass.

        Returns:
            A :class:`~app.decision.DecisionResult` with
            :attr:`~app.decision.DecisionPath.ML_FALLBACK` path.
        """
        if use_mc_dropout:
            command, confidence, uncertainty, mean_softmax = (
                self.predict_with_uncertainty(image, n_samples=self._mc_dropout_samples)
            )
            softmax_list = mean_softmax
        else:
            command, confidence, softmax_list = self.predict(image)

        return DecisionResult(
            command=command,
            confidence=confidence,
            decision_path=DecisionPath.ML_FALLBACK,
            ml_softmax=softmax_list,
        )

    def is_ready(self) -> bool:
        """Return ``True`` when the ONNX model is loaded and ready."""
        return self._model_loaded

    def warmup(self) -> float:
        """Run a single blank-image forward pass and return latency in ms.

        Returns:
            Inference time in milliseconds, or ``0.0`` if the model is
            not loaded.
        """
        if not self._model_loaded:
            return 0.0
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        self.predict(blank)
        return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import sys

    logging.basicConfig(level=logging.INFO)

    model = MLFallbackModel()
    print(f"Model loaded: {model.is_ready()}")

    if not model.is_ready():
        print("Expected — train the model first in Phase 4.3")

    rgb_images = sorted(glob.glob("rgb/rgb_image_*.png"))[:3]
    if not rgb_images:
        print("No RGB images found in rgb/ — skipping image tests.")
    else:
        for img_path in rgb_images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Could not read {img_path}")
                continue
            result = model.predict_to_decision_result(img, use_mc_dropout=False)
            print(
                f"  {Path(img_path).name}  →  {result.command.value}"
                f"  conf={result.confidence:.3f}"
                f"  path={result.decision_path.value}"
            )

    print()
    print("ML fallback pipeline ready (model weights pending training)")