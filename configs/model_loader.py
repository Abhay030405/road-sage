import os

from dotenv import load_dotenv

load_dotenv()

DEVICE: str = os.getenv("DEVICE", "cpu").lower()
USE_ONNX: bool = os.getenv("USE_ONNX", "true").lower() in ("1", "true", "yes")
UFLD_MODEL: str = os.getenv("UFLD_MODEL", "ufldv2_resnet18")
FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "mobilenetv3_small")
DETECTOR_MODEL: str = os.getenv("DETECTOR_MODEL", "nanodet_plus_m")
DEPTH_MODEL: str = os.getenv("DEPTH_MODEL", "midas_small")

if DEVICE == "cuda":
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "[model_loader] WARNING: DEVICE=cuda requested but no CUDA device found. "
                "Falling back to cpu + ONNX mode."
            )
            DEVICE = "cpu"
            USE_ONNX = True
    except ImportError:
        print(
            "[model_loader] WARNING: DEVICE=cuda requested but torch is not installed. "
            "Falling back to cpu + ONNX mode."
        )
        DEVICE = "cpu"
        USE_ONNX = True
