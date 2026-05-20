"""Export RoadSage PyTorch model checkpoints to ONNX format.

Converts `.pth` weight files to ONNX graphs compatible with ONNX Runtime
(CPU mode, opset 17).  After export the script verifies the graph loads
cleanly and prints a latency comparison between PyTorch and ONNX Runtime so
you can confirm the expected speed-up on the target hardware.

Supported models
----------------
* ``lane_detector``  — UFLDv2 ResNet-18 (input 1×3×288×800)
* ``depth_estimator`` — MiDaS Small (input 1×3×256×256)
* ``decision_cnn``   — MobileNetV3-Small (input 1×3×224×224)
* ``all``            — exports all three in sequence

Usage::

    python training/scripts/export_onnx.py --model lane_detector
    python training/scripts/export_onnx.py --model all
    python training/scripts/export_onnx.py \\
        --model lane_detector \\
        --input_pth  models/tusimple_res18.pth \\
        --output_onnx models/lane_detector.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred so the script can print --help without
# requiring a full CUDA/PyTorch install.
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        log.error("PyTorch is not installed. Run: pip install torch")
        sys.exit(1)


def _require_ort():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        log.error("ONNX Runtime is not installed. Run: pip install onnxruntime")
        sys.exit(1)


def _require_onnx():
    try:
        import onnx
        return onnx
    except ImportError:
        log.error("ONNX is not installed. Run: pip install onnx")
        sys.exit(1)


# ---------------------------------------------------------------------------
# ONNX verification helper
# ---------------------------------------------------------------------------


def verify_onnx(onnx_path: str, dummy_input_shape: Tuple[int, ...]) -> None:
    """Load an ONNX model, run one inference pass, and print the output shape.

    Args:
        onnx_path:         Path to the ``.onnx`` file to verify.
        dummy_input_shape: Shape of the random float32 input tensor
                           (e.g. ``(1, 3, 288, 800)``).

    Raises:
        SystemExit: On any ONNX load or inference error.
    """
    ort = _require_ort()
    onnx_lib = _require_onnx()

    log.info("Verifying ONNX model: %s", onnx_path)

    # Graph-level check
    try:
        model_proto = onnx_lib.load(onnx_path)
        onnx_lib.checker.check_model(model_proto)
    except Exception as exc:  # noqa: BLE001
        log.error("ONNX graph check failed: %s", exc)
        sys.exit(1)

    # Runtime inference check
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3   # suppress ORT verbose output
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to create ORT session: %s", exc)
        sys.exit(1)

    input_name  = session.get_inputs()[0].name
    dummy_input = np.random.randn(*dummy_input_shape).astype(np.float32)

    # Warm-up
    session.run(None, {input_name: dummy_input})

    # Timed run
    t0      = time.perf_counter()
    outputs = session.run(None, {input_name: dummy_input})
    ort_ms  = (time.perf_counter() - t0) * 1000.0

    log.info("ONNX Runtime inference OK  —  output shapes: %s",
             [o.shape for o in outputs])
    log.info("ONNX Runtime latency (single pass, CPU): %.2f ms", ort_ms)


# ---------------------------------------------------------------------------
# UFLDv2 ResNet-18 — lane detector
# ---------------------------------------------------------------------------


def export_lane_detector(pth_path: str, onnx_path: str) -> None:
    """Export the UFLDv2 ResNet-18 lane detector to ONNX.

    The full UFLDv2 architecture is implemented in Phase 4.  This function
    uses a ResNet-18 feature extractor as a structural placeholder that
    matches the expected input/output contract so that downstream ONNX
    Runtime integration can be developed and tested before the real weights
    are available.

    Args:
        pth_path:  Path to the PyTorch checkpoint (``.pth``).
        onnx_path: Destination path for the exported ``.onnx`` file.

    Raises:
        SystemExit: When the checkpoint file is missing or export fails.
    """
    torch = _require_torch()

    pth_file  = Path(pth_path)
    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    INPUT_SHAPE = (1, 3, 288, 800)   # UFLDv2 canonical input size

    # ---- Load / build model ----
    if pth_file.exists():
        log.info("Loading checkpoint: %s", pth_file)
        try:
            import torchvision.models as tvm
            # Structural placeholder — actual UFLDv2 head added in Phase 4
            model = tvm.resnet18(weights=None)
            state = torch.load(str(pth_file), map_location="cpu")
            # Accept raw state-dict or Lightning-style checkpoint dict
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning("Missing keys in checkpoint (%d) — placeholder arch mismatch "
                            "is expected before Phase 4.", len(missing))
            log.info("Checkpoint loaded.")
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load checkpoint: %s", exc)
            sys.exit(1)
    else:
        log.warning(
            "Checkpoint not found at '%s'.  Exporting placeholder ResNet-18 "
            "architecture for integration testing.", pth_file
        )
        try:
            import torchvision.models as tvm
            model = tvm.resnet18(weights=None)
        except ImportError:
            log.error("torchvision is required for export. pip install torchvision")
            sys.exit(1)

    model.eval()

    dummy_input = torch.randn(*INPUT_SHAPE)

    # ---- PyTorch baseline latency ----
    with torch.no_grad():
        torch.jit.trace(model, dummy_input)   # warm-up trace
        t0 = time.perf_counter()
        _ = model(dummy_input)
        pt_ms = (time.perf_counter() - t0) * 1000.0
    log.info("PyTorch inference latency (CPU): %.2f ms", pt_ms)

    # ---- Export ----
    log.info("Exporting to ONNX: %s  (opset 17) …", onnx_file)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
    log.info("ONNX export complete: %s", onnx_file)

    # ---- Verify ----
    verify_onnx(str(onnx_file), INPUT_SHAPE)

    log.info(
        "Lane detector exported successfully.  "
        "PyTorch: %.2f ms  |  ONNX Runtime: (see above)", pt_ms
    )


# ---------------------------------------------------------------------------
# MiDaS Small — depth estimator
# ---------------------------------------------------------------------------


def export_depth_estimator(pth_path: str, onnx_path: str) -> None:
    """Export MiDaS Small depth estimator to ONNX.

    Args:
        pth_path:  Path to the MiDaS ``.pth`` / ``.pt`` checkpoint.
        onnx_path: Destination path for the ``.onnx`` file.

    Raises:
        SystemExit: When export fails.
    """
    torch = _require_torch()

    pth_file  = Path(pth_path)
    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    INPUT_SHAPE = (1, 3, 256, 256)   # MiDaS Small canonical size

    log.info("Loading MiDaS Small …")
    try:
        model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small",
            pretrained=(not pth_file.exists()),
        )
        if pth_file.exists():
            state = torch.load(str(pth_file), map_location="cpu")
            model.load_state_dict(state, strict=False)
            log.info("Loaded weights from %s", pth_file)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load MiDaS model: %s", exc)
        sys.exit(1)

    model.eval()
    dummy_input = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(dummy_input)
        pt_ms = (time.perf_counter() - t0) * 1000.0
    log.info("PyTorch inference latency (CPU): %.2f ms", pt_ms)

    log.info("Exporting to ONNX: %s  (opset 17) …", onnx_file)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(onnx_file),
            opset_version=17,
            input_names=["input"],
            output_names=["depth"],
            dynamic_axes={"input": {0: "batch_size"}, "depth": {0: "batch_size"}},
        )
    log.info("ONNX export complete: %s", onnx_file)
    verify_onnx(str(onnx_file), INPUT_SHAPE)
    log.info("Depth estimator exported.  PyTorch: %.2f ms | ONNX Runtime: (see above)", pt_ms)


# ---------------------------------------------------------------------------
# MobileNetV3-Small — decision CNN
# ---------------------------------------------------------------------------


def export_decision_cnn(pth_path: str, onnx_path: str) -> None:
    """Export the MobileNetV3-Small decision CNN to ONNX.

    Args:
        pth_path:  Path to the Phase-4 trained ``.pth`` checkpoint.
        onnx_path: Destination path for the ``.onnx`` file.

    Raises:
        SystemExit: When the checkpoint is missing or export fails.
    """
    torch = _require_torch()

    pth_file  = Path(pth_path)
    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    INPUT_SHAPE = (1, 3, 224, 224)   # MobileNetV3 canonical size
    NUM_CLASSES = 4                   # FORWARD / LEFT / RIGHT / STOP

    if not pth_file.exists():
        log.error(
            "Decision CNN checkpoint not found at '%s'.\n"
            "Train the model first via: "
            "python training/scripts/train_decision_cnn.py --config configs/production.yaml",
            pth_file,
        )
        sys.exit(1)

    log.info("Loading Decision CNN checkpoint: %s", pth_file)
    try:
        import torchvision.models as tvm
        model = tvm.mobilenet_v3_small(weights=None)
        # Replace classifier head to match the 4-class decision output
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)
        state = torch.load(str(pth_file), map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=True)
        log.info("Checkpoint loaded.")
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load Decision CNN: %s", exc)
        sys.exit(1)

    model.eval()
    dummy_input = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(dummy_input)
        pt_ms = (time.perf_counter() - t0) * 1000.0
    log.info("PyTorch inference latency (CPU): %.2f ms", pt_ms)

    log.info("Exporting to ONNX: %s  (opset 17) …", onnx_file)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(onnx_file),
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
    log.info("ONNX export complete: %s", onnx_file)
    verify_onnx(str(onnx_file), INPUT_SHAPE)
    log.info("Decision CNN exported.  PyTorch: %.2f ms | ONNX Runtime: (see above)", pt_ms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODEL_DEFAULTS = {
    "lane_detector":   ("models/lane_detector.pth",   "models/lane_detector.onnx"),
    "depth_estimator": ("models/depth_estimator.pth", "models/depth_estimator.onnx"),
    "decision_cnn":    ("models/decision_cnn.pth",    "models/decision_cnn.onnx"),
}

_EXPORTERS = {
    "lane_detector":   export_lane_detector,
    "depth_estimator": export_depth_estimator,
    "decision_cnn":    export_decision_cnn,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export RoadSage PyTorch checkpoints to ONNX (CPU, opset 17).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        choices=["lane_detector", "depth_estimator", "decision_cnn", "all"],
        required=True,
        help="Which model to export.",
    )
    p.add_argument(
        "--input_pth",
        default=None,
        help="Path to the .pth checkpoint.  Defaults to models/<model>.pth.",
    )
    p.add_argument(
        "--output_onnx",
        default=None,
        help="Destination .onnx path.  Defaults to models/<model>.onnx.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    targets: list[str] = (
        list(_MODEL_DEFAULTS.keys()) if args.model == "all" else [args.model]
    )

    for model_name in targets:
        default_pth, default_onnx = _MODEL_DEFAULTS[model_name]
        pth_path   = args.input_pth   or default_pth
        onnx_path  = args.output_onnx or default_onnx

        log.info("=" * 60)
        log.info("Exporting: %s", model_name)
        log.info("  Input  : %s", pth_path)
        log.info("  Output : %s", onnx_path)
        log.info("=" * 60)

        exporter = _EXPORTERS[model_name]
        exporter(pth_path, onnx_path)

    log.info("All requested exports complete.")
