"""
app.explainability.gradcam
===========================

Gradient-weighted Class Activation Mapping (GradCAM) for the RoadSage
MobileNetV3-Small decision CNN.

GradCAM highlights which spatial regions of an input frame most influenced
the model's predicted driving command, making the system's decisions
interpretable to judges and domain experts.

Architecture note
-----------------
MobileNetV3-Small (torchvision) layout::

    MobileNetV3
      features[0..12]   — convolutional backbone
      avgpool           — AdaptiveAvgPool2d(1)
      classifier[0..3]  — FC head

The default target layer is ``features.12``, the last convolutional block
before global average pooling.  Its 7×7 spatial feature maps (for a 224×224
input) carry high-level semantic information while still retaining enough
spatial resolution for a meaningful heatmap.

Graceful degradation
--------------------
All classes degrade gracefully when PyTorch or the model weights are
unavailable.  :func:`generate_gradcam_placeholder_result` produces a
Gaussian-blob heatmap that substitutes for a real GradCAM result during
development or when only ONNX weights have been exported.

Typical usage::

    from app.explainability.gradcam import GradCAM, GradCAMManager

    gcam = GradCAM()                                  # loads models/decision_cnn.pth
    result = gcam.generate(frame, target_class=0)     # 0 = FORWARD
    cv2.imwrite("gradcam.jpg", result.overlay)

    manager = GradCAMManager()
    b64 = manager.maybe_generate(frame, predicted_command="LEFT")
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_COMMAND_TO_CLASS: dict[str, int] = {
    "FORWARD": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "STOP": 3,
}

_GRADCAM_CONFIG_KEY = "gradcam"
_ML_FALLBACK_CONFIG_KEY = "ml_fallback"
_DEFAULT_PTH_PATH = "models/decision_cnn.pth"
_DEFAULT_EVERY_N = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GradCAMResult:
    """Output of a single GradCAM pass.

    Attributes:
        heatmap: Float32 array of shape ``(H, W)`` with values in ``[0, 1]``.
            Higher values indicate regions that most influenced the prediction.
        overlay: BGR uint8 array of shape ``(H, W, 3)``.  The JET colormap
            heatmap blended onto the original image at ``alpha`` transparency.
        predicted_class: Integer class index ``{0=FORWARD, 1=LEFT, 2=RIGHT, 3=STOP}``.
        predicted_class_name: Human-readable command string.
        confidence: Softmax probability of *predicted_class* in ``[0, 1]``.
        layer_name: Name of the convolutional layer whose activations were used.
    """

    heatmap: np.ndarray
    overlay: np.ndarray
    predicted_class: int
    predicted_class_name: str
    confidence: float
    layer_name: str


# ---------------------------------------------------------------------------
# Placeholder (no weights required)
# ---------------------------------------------------------------------------


def generate_gradcam_placeholder_result(image: np.ndarray) -> GradCAMResult:
    """Produce a fake GradCAM result using a centred Gaussian blob.

    Used when model weights are not available (development / CI).  The
    heatmap is a normalised 2-D Gaussian centred at the image midpoint with
    σ = min(H, W) / 4.

    Args:
        image: BGR uint8 source frame of any resolution.

    Returns:
        :class:`GradCAMResult` with ``predicted_class=0`` (FORWARD),
        ``confidence=0.0``, and ``layer_name="placeholder"``.
    """
    h, w = image.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    sigma = min(h, w) / 4.0
    gaussian = np.exp(
        -((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma ** 2)
    )
    heatmap = (gaussian / (gaussian.max() + 1e-8)).astype(np.float32)

    colormap_bgr = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image, 0.6, colormap_bgr, 0.4, 0)

    return GradCAMResult(
        heatmap=heatmap,
        overlay=overlay,
        predicted_class=0,
        predicted_class_name="FORWARD",
        confidence=0.0,
        layer_name="placeholder",
    )


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


class GradCAM:
    """GradCAM explainability for the RoadSage MobileNetV3-Small decision CNN.

    Registers forward and backward hooks on *target_layer_name* to capture
    intermediate activations and their gradients during inference.  The
    weighted activation map is upsampled to the original image resolution
    and overlaid as a JET heatmap.

    Args:
        model_path: Path to the PyTorch ``.pth`` checkpoint.  If the file
            does not exist, the instance operates in placeholder mode.
        target_layer_name: Dot-separated attribute path of the convolutional
            layer to hook, e.g. ``"features.12"``.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    _CLASS_NAMES = ["FORWARD", "LEFT", "RIGHT", "STOP"]

    def __init__(
        self,
        model_path: str = _DEFAULT_PTH_PATH,
        target_layer_name: str = "features.12",
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._target_layer_name = target_layer_name
        self._device_str = device
        self._model_loaded = False
        self._model = None
        self._activations: Optional[object] = None
        self._gradients: Optional[object] = None

        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the PyTorch MobileNetV3-Small checkpoint.

        Logs a warning (rather than raising) when weights are missing so the
        rest of the pipeline can still operate in placeholder mode.
        """
        if not Path(self._model_path).exists():
            logger.warning(
                "GradCAM: model not found at '%s'. Placeholder heatmaps will be used.",
                self._model_path,
            )
            return

        try:
            import torch  # type: ignore[import]
            import torchvision.models as tvm  # type: ignore[import]

            self._device = torch.device(self._device_str)

            model = tvm.mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, 4)

            state = torch.load(self._model_path, map_location=self._device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)

            model.to(self._device)
            model.eval()
            self._model = model

            target_layer = model
            for attr in self._target_layer_name.split("."):
                target_layer = getattr(target_layer, attr)

            def _forward_hook(module, inp, output):
                self._activations = output.detach()

            def _backward_hook(module, grad_input, grad_output):
                self._gradients = grad_output[0].detach()

            target_layer.register_forward_hook(_forward_hook)
            target_layer.register_full_backward_hook(_backward_hook)

            self._model_loaded = True
            logger.info(
                "GradCAM loaded: %s  target_layer=%s",
                self._model_path,
                self._target_layer_name,
            )

        except Exception as exc:
            logger.warning("GradCAM: failed to load model — %s", exc)
            self._model_loaded = False

    def _preprocess(self, image: np.ndarray):
        """Preprocess a BGR image into a normalised NCHW PyTorch tensor.

        Steps: BGR→RGB, resize to 224×224, ImageNet normalisation, add batch dim.

        Args:
            image: BGR uint8 array of shape ``(H, W, 3)``.

        Returns:
            ``torch.Tensor`` of shape ``(1, 3, 224, 224)`` on ``self._device``.
        """
        import torch  # type: ignore[import]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` when the PyTorch model is loaded and hooks are registered."""
        return self._model_loaded

    def generate(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
    ) -> GradCAMResult:
        """Compute a GradCAM heatmap and blend it onto *image*.

        When *target_class* is ``None`` the predicted class (argmax of logits)
        is used as the back-propagation target.

        Args:
            image: BGR uint8 source frame.
            target_class: Integer class index to explain; ``None`` uses argmax.
            alpha: Heatmap overlay opacity in ``[0, 1]``.

        Returns:
            :class:`GradCAMResult` with heatmap, overlay, class info, and
            the name of the hooked layer.
        """
        if not self._model_loaded:
            return generate_gradcam_placeholder_result(image)

        import torch  # type: ignore[import]

        h_orig, w_orig = image.shape[:2]
        input_tensor = self._preprocess(image)

        self._model.zero_grad()
        logits = self._model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        confidence = float(torch.softmax(logits, dim=1)[0, target_class].item())
        class_name = (
            self._CLASS_NAMES[target_class]
            if target_class < len(self._CLASS_NAMES)
            else str(target_class)
        )

        self._model.zero_grad()
        logits[0, target_class].backward()

        weights = self._gradients.mean(dim=(2, 3))   # (1, C)
        activations = self._activations[0]            # (C, H_feat, W_feat)
        _, act_h, act_w = activations.shape
        cam = torch.zeros(act_h, act_w, device=self._device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy().astype(np.float32)
        cam_np = cv2.resize(cam_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        cam_min, cam_max = cam_np.min(), cam_np.max()
        heatmap = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)

        colormap_bgr = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(image, 1.0 - alpha, colormap_bgr, alpha, 0)

        return GradCAMResult(
            heatmap=heatmap,
            overlay=overlay,
            predicted_class=target_class,
            predicted_class_name=class_name,
            confidence=confidence,
            layer_name=self._target_layer_name,
        )

    def generate_base64(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
    ) -> str:
        """Generate a GradCAM overlay and return it as a base64-encoded JPEG.

        Args:
            image: BGR uint8 source frame.
            target_class: Class index to explain (``None`` = predicted class).

        Returns:
            Base64-encoded JPEG string of the overlay image.
        """
        result = self.generate(image, target_class=target_class)
        _, buf = cv2.imencode(".jpg", result.overlay)
        return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# GradCAMManager
# ---------------------------------------------------------------------------


class GradCAMManager:
    """Frame-skipping wrapper around :class:`GradCAM` for streaming performance.

    Throttles GradCAM generation to every *N* frames so latency impact is
    minimal during live inference.  STOP commands are always explained
    because safety decisions warrant scrutiny.

    Args:
        config_path: Path to ``configs/decision_engine.yaml``.  The
            ``gradcam.every_n_frames`` key controls the skip frequency
            (defaults to 5 when not present).
    """

    def __init__(self, config_path: str = "configs/decision_engine.yaml") -> None:
        self._every_n: int = _DEFAULT_EVERY_N
        pytorch_path: str = _DEFAULT_PTH_PATH

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            self._every_n = int(
                raw.get(_GRADCAM_CONFIG_KEY, {}).get("every_n_frames", _DEFAULT_EVERY_N)
            )
            pytorch_path = raw.get(_ML_FALLBACK_CONFIG_KEY, {}).get(
                "pytorch_path", _DEFAULT_PTH_PATH
            )
        except FileNotFoundError:
            logger.warning(
                "GradCAMManager: config not found at '%s', using defaults.", config_path
            )
        except Exception as exc:
            logger.warning("GradCAMManager: config load error — %s. Using defaults.", exc)

        self._gradcam = GradCAM(model_path=pytorch_path)
        self._frame_count: int = 0

        logger.info(
            "GradCAMManager ready: every_n=%d  model_ready=%s",
            self._every_n,
            self._gradcam.is_ready(),
        )

    def maybe_generate(
        self,
        image: np.ndarray,
        predicted_command: str,
    ) -> Optional[str]:
        """Conditionally generate a GradCAM base64 string based on frame count.

        Args:
            image: BGR uint8 source frame.
            predicted_command: Driving command string — ``"FORWARD"``,
                ``"LEFT"``, ``"RIGHT"``, or ``"STOP"``.

        Returns:
            Base64-encoded JPEG GradCAM overlay, or ``None`` if this frame
            is being skipped.
        """
        self._frame_count += 1

        is_stop = predicted_command.upper() == "STOP"
        due_for_gradcam = (self._frame_count % self._every_n) == 0

        if not (due_for_gradcam or is_stop):
            return None

        class_id = _COMMAND_TO_CLASS.get(predicted_command.upper(), 0)
        return self._gradcam.generate_base64(image, target_class=class_id)


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import os

    logging.basicConfig(level=logging.INFO)

    gradcam = GradCAM()
    print(f"GradCAM model loaded: {gradcam.is_ready()}")

    os.makedirs("outputs", exist_ok=True)

    images = sorted(glob.glob("rgb/rgb_image_*.png"))[:3]
    if not images:
        print("No images found in rgb/ — generating placeholder from blank frame.")
        images_data = [(np.zeros((480, 640, 3), dtype=np.uint8), "blank")]
    else:
        images_data = [(cv2.imread(p), Path(p).name) for p in images]

    for n, (img, name) in enumerate(images_data):
        if img is None:
            print(f"  Could not read {name}, skipping.")
            continue
        result = gradcam.generate(img)
        out_path = f"outputs/gradcam_{n}.jpg"
        cv2.imwrite(out_path, result.overlay)
        print(
            f"  [{n}] {name} → {result.predicted_class_name} "
            f"(conf={result.confidence:.3f}) layer={result.layer_name}"
        )

    print("GradCAM overlays saved to outputs/")
