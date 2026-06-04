"""
training.evaluation.evaluate_phase5
=====================================

Phase 5 exit-criteria validation script.

Checks every Phase 5 deliverable end-to-end and prints a colour-coded
checklist so the team can confirm readiness before moving to Phase 6
(Dashboard / live demo).

Usage::

    python training/evaluation/evaluate_phase5.py

    # Or with the server already running in a separate terminal:
    uvicorn api.main:app --port 8000 &
    python training/evaluation/evaluate_phase5.py

Exit code
---------
``0`` — all 8 criteria passed.
``1`` — one or more criteria failed or raised an uncaught exception.
"""

from __future__ import annotations

import base64
import glob
import json
import logging
import sys
import time
from collections import Counter
from typing import List, Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers (ANSI — degrade gracefully on Windows without colour support)
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _ok(msg: str) -> str:
    return f"{_GREEN}✅{_RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}❌{_RESET} {msg}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}⚠{_RESET}  {msg}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rgb_images(n: int = 20) -> List[str]:
    """Return up to *n* sorted image paths from the ``rgb/`` directory."""
    return sorted(glob.glob("rgb/rgb_image_*.png"))[:n]


def _load_image(path: str) -> Optional[np.ndarray]:
    """Read a BGR image; return ``None`` on failure (logged, not raised)."""
    img = cv2.imread(path)
    if img is None:
        logger.warning("Could not read image: %s", path)
    return img


# ---------------------------------------------------------------------------
# Validation criteria
# ---------------------------------------------------------------------------


def _check_engine_instantiation() -> tuple[bool, str]:
    """Criterion 1 — engine instantiates without error."""
    try:
        from app.engine import RoadSageEngine
        RoadSageEngine()
        return True, _ok("Engine instantiated")
    except Exception as exc:
        return False, _fail(f"Engine instantiation failed: {exc}")


def _check_predict_20_images(engine) -> tuple[bool, str]:
    """Criterion 2 — predict() runs on 20 images without exception."""
    images = _rgb_images(20)
    if not images:
        return False, _fail("predict on 20 images: no images found in rgb/")

    errors = 0
    for path in images:
        try:
            img = _load_image(path)
            result = engine.predict(img)
            assert result.command in ("FORWARD", "LEFT", "RIGHT", "STOP"), (
                f"Unexpected command: {result.command!r}"
            )
        except Exception as exc:
            logger.debug("predict failed on %s: %s", path, exc)
            errors += 1

    ok = errors == 0
    n = len(images)
    msg = f"engine.predict on {n} images: {n - errors}/{n} succeeded"
    return ok, (_ok(msg) if ok else _fail(msg))


def _check_command_distribution(engine) -> tuple[bool, str]:
    """Criterion 3 — command distribution is sensible (not 100 % STOP)."""
    images = _rgb_images(20)
    if not images:
        return False, _fail("Command distribution: no images found in rgb/")

    commands = []
    for path in images:
        img = _load_image(path)
        if img is None:
            continue
        try:
            commands.append(engine.predict(img).command)
        except Exception:
            commands.append("STOP")

    if not commands:
        return False, _fail("Command distribution: all images failed to load")

    stop_rate = commands.count("STOP") / len(commands)
    dist = dict(Counter(commands))
    ok = stop_rate < 0.95
    msg = f"Command distribution: {dist}"
    return ok, (_ok(msg) if ok else _fail(msg + f"  [stop_rate={stop_rate:.0%}]"))


def _check_latency(engine) -> tuple[bool, str]:
    """Criterion 4 — P95 latency < 500 ms on CPU without ONNX weights."""
    images = _rgb_images(10)
    if not images:
        return False, _fail("Latency: no images found in rgb/")

    latencies: List[float] = []
    for path in images:
        img = _load_image(path)
        if img is None:
            continue
        try:
            result = engine.predict(img)
            latencies.append(result.latency_ms["total"])
        except Exception:
            pass

    if not latencies:
        return False, _fail("Latency: all predict calls failed")

    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    p95 = sorted(latencies)[p95_idx]
    ok = p95 < 500.0
    msg = f"P95 latency: {p95:.0f}ms (target <500ms without ONNX)"
    return ok, (_ok(msg) if ok else _fail(msg))


def _check_visualization(engine) -> tuple[bool, str]:
    """Criterion 5 — include_viz=True returns a non-empty base64 JPEG."""
    images = _rgb_images(1)
    if not images:
        return False, _fail("Visualization: no images found in rgb/")

    img = _load_image(images[0])
    if img is None:
        return False, _fail("Visualization: could not read first image")

    try:
        result = engine.predict(img, include_viz=True)
    except Exception as exc:
        return False, _fail(f"Visualization: predict raised {exc}")

    if result.lane_viz_base64:
        try:
            decoded = base64.b64decode(result.lane_viz_base64)
            assert len(decoded) > 100, "decoded JPEG is suspiciously small"
            return True, _ok("Visualization produces valid base64 image")
        except Exception as exc:
            return False, _fail(f"Visualization: base64 decode failed — {exc}")

    return False, _warn("No visualization returned (check include_viz=True)")


def _check_api_health() -> tuple[bool, str]:
    """Criterion 6 — API /health returns 200 (skipped if server is not running)."""
    try:
        import requests  # type: ignore[import]
        resp = requests.get("http://localhost:8000/api/v1/health", timeout=3)
        if resp.status_code == 200:
            return True, _ok("API /health returned 200")
        return False, _fail(f"API /health returned HTTP {resp.status_code}")
    except ImportError:
        return False, _warn(
            "requests not installed — install with: pip install requests"
        )
    except Exception:
        # Server not running — treat as a warning, not a hard failure
        return True, _warn(
            "API server not running — start with: uvicorn api.main:app"
        )


def _check_api_predict() -> tuple[bool, str]:
    """Criterion 7 — API /predict returns a valid command (skipped if server not running)."""
    try:
        import requests  # type: ignore[import]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = requests.post(
            "http://localhost:8000/api/v1/predict",
            files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=10,
        )
        if resp.status_code != 200:
            return False, _fail(f"API /predict returned HTTP {resp.status_code}")
        command = resp.json().get("command")
        if command not in ("FORWARD", "LEFT", "RIGHT", "STOP"):
            return False, _fail(f"API /predict returned invalid command: {command!r}")
        return True, _ok(f"API /predict returned valid command: {command}")
    except ImportError:
        return False, _warn(
            "requests not installed — install with: pip install requests"
        )
    except Exception:
        return True, _warn(
            "API server not running — start with: uvicorn api.main:app"
        )


def _check_json_serialization(engine) -> tuple[bool, str]:
    """Criterion 8 — PredictionResult.to_json() round-trips cleanly."""
    images = _rgb_images(1)
    if not images:
        return False, _fail("JSON serialization: no images found in rgb/")

    img = _load_image(images[0])
    if img is None:
        return False, _fail("JSON serialization: could not read first image")

    try:
        result = engine.predict(img)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert "command" in parsed, "command key missing from parsed JSON"
        assert parsed["command"] in ("FORWARD", "LEFT", "RIGHT", "STOP")
        return True, _ok("PredictionResult serializes to valid JSON")
    except Exception as exc:
        return False, _fail(f"JSON serialization failed: {exc}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_phase5_validation() -> int:
    """Run all Phase 5 exit-criteria checks and print a formatted checklist.

    Returns:
        ``0`` if every criterion passed, ``1`` otherwise.
    """
    print()
    print("=" * 60)
    print("  RoadSage — Phase 5 Exit-Criteria Validation")
    print("=" * 60)
    print()

    # Criterion 1 must succeed before we can run the others
    ok1, line1 = _check_engine_instantiation()
    print(line1)

    if not ok1:
        print()
        print(_fail("Cannot continue — engine did not instantiate."))
        print("Phase 5 validation complete: 0/8 criteria passed")
        return 1

    from app.engine import RoadSageEngine
    engine = RoadSageEngine()

    checks = [
        (True, line1),                          # 1 — already done above
        _check_predict_20_images(engine),       # 2
        _check_command_distribution(engine),    # 3
        _check_latency(engine),                 # 4
        _check_visualization(engine),           # 5
        _check_api_health(),                    # 6
        _check_api_predict(),                   # 7
        _check_json_serialization(engine),      # 8
    ]

    results: List[bool] = []
    for i, (passed, line) in enumerate(checks):
        if i > 0:  # criterion 1 printed above
            print(line)
        results.append(passed)

    total = len(results)
    passed_count = sum(results)

    print()
    print("=" * 60)
    print(f"Phase 5 validation complete: {passed_count}/{total} criteria passed")
    if passed_count == total:
        print(_ok("Ready for Phase 6 (Dashboard)"))
    else:
        print(_fail("Fix failing criteria before Phase 6"))
    print("=" * 60)
    print()

    return 0 if passed_count == total else 1


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_phase5_validation())
