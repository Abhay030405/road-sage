"""
training.evaluation.evaluate_phase7
=====================================

Phase 7 exit-criteria validation script.

Runs 11 checks that together confirm RoadSage is production-ready before the
live demo.  Each check is printed with a ✅ / ❌ / ⚠️  symbol so the team
can quickly scan the output.

Usage::

    python training/evaluation/evaluate_phase7.py

    # Or via Makefile:
    make phase7-check

Exit code
---------
``0`` — all mandatory criteria passed (ONNX presence is advisory only).
``1`` — one or more criteria failed.
"""

from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def print_criterion(
    label: str,
    passed: Optional[bool],
    detail: str = "",
) -> None:
    """Print a single criterion line with a pass/fail/skip symbol.

    Parameters
    ----------
    label:
        Short human-readable criterion name.
    passed:
        ``True`` → ✅ pass, ``False`` → ❌ fail, ``None`` → ⚠️  skip/N-A.
    detail:
        Optional one-line explanation printed on an indented second line.
    """
    if passed is True:
        symbol = "✅"        # ✅
    elif passed is False:
        symbol = "❌"        # ❌
    else:
        symbol = "⚠️ " # ⚠️

    print(f"  {symbol}  {label}")
    if detail:
        print(f"       {detail}")


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def run_phase7_validation() -> int:
    """Run all 11 Phase 7 exit criteria and print a summary.

    Returns
    -------
    int
        ``0`` if no mandatory criteria failed, ``1`` otherwise.
    """

    print()
    print("━" * 60)
    print("  RoadSage — Phase 7 Exit Criteria Validation")
    print("━" * 60)
    print()

    results: list[Optional[bool]] = []

    # We hold a shared engine reference so model weights are loaded once.
    engine = None

    # ------------------------------------------------------------------ #
    # CRITERION 1 — Full pytest suite                                     #
    # ------------------------------------------------------------------ #
    print("[ 1/11 ] Running pytest suite ...")
    try:
        proc = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
        )
        passed = proc.returncode == 0
        output_lines = proc.stdout.strip().split("\n")
        summary = output_lines[-1] if output_lines else "no output"
        print_criterion("All pytest tests passing", passed, summary)
        results.append(passed)
    except FileNotFoundError:
        print_criterion("All pytest tests passing", None, "pytest not found — skipping")
        results.append(None)

    # ------------------------------------------------------------------ #
    # CRITERION 2 — Ruff lint                                             #
    # ------------------------------------------------------------------ #
    print("[ 2/11 ] Running ruff ...")
    try:
        proc = subprocess.run(
            ["ruff", "check", "."],
            capture_output=True,
            text=True,
        )
        passed = proc.returncode == 0
        print_criterion(
            "Zero ruff lint errors",
            passed,
            "Clean" if passed else proc.stdout[:200].strip(),
        )
        results.append(passed)
    except FileNotFoundError:
        print_criterion("Zero ruff lint errors", None, "ruff not installed — skipping")
        results.append(None)

    # ------------------------------------------------------------------ #
    # CRITERION 3 — Engine instantiates and predicts                     #
    # ------------------------------------------------------------------ #
    print("[ 3/11 ] Engine instantiation ...")
    try:
        from app.engine import RoadSageEngine

        engine = RoadSageEngine()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        r = engine.predict(blank)
        assert r.command in {"FORWARD", "LEFT", "RIGHT", "STOP"}
        print_criterion(
            "Engine instantiates and predicts",
            True,
            f"command={r.command}  conf={r.confidence:.2f}",
        )
        results.append(True)
    except Exception as exc:
        print_criterion("Engine instantiates and predicts", False, str(exc))
        results.append(False)
        engine = None

    # ------------------------------------------------------------------ #
    # CRITERION 4 — No crashes on 20 real images                         #
    # ------------------------------------------------------------------ #
    print("[ 4/11 ] No-crash test on 20 MNNIT images ...")
    if engine is None:
        print_criterion(
            "No crashes on 20 MNNIT images",
            None,
            "Skipped — engine not available",
        )
        results.append(None)
    else:
        try:
            image_paths = sorted(glob.glob("rgb/rgb_image_*.png"))[:20]
            if not image_paths:
                print_criterion(
                    "No crashes on 20 MNNIT images",
                    None,
                    "No images found in rgb/ — skipping",
                )
                results.append(None)
            else:
                errors = 0
                for p in image_paths:
                    try:
                        engine.predict(cv2.imread(p))
                    except Exception:
                        errors += 1
                passed = errors == 0
                print_criterion(
                    "No crashes on 20 MNNIT images",
                    passed,
                    f"{len(image_paths) - errors}/{len(image_paths)} succeeded",
                )
                results.append(passed)
        except Exception as exc:
            print_criterion("No crashes on 20 MNNIT images", False, str(exc))
            results.append(False)

    # ------------------------------------------------------------------ #
    # CRITERION 5 — Latency benchmark                                    #
    # ------------------------------------------------------------------ #
    print("[ 5/11 ] Latency benchmark ...")
    if engine is None:
        print_criterion("Latency benchmark", None, "Skipped — engine not available")
        results.append(None)
    else:
        try:
            image_paths = sorted(glob.glob("rgb/rgb_image_*.png"))[:20]
            if not image_paths:
                print_criterion(
                    "Latency benchmark", None, "No images in rgb/ — skipping"
                )
                results.append(None)
            else:
                latencies: list[float] = []
                for p in image_paths:
                    r = engine.predict(cv2.imread(p))
                    latencies.append(r.latency_ms["total"])

                p95 = sorted(latencies)[int(len(latencies) * 0.95)]
                avg = sum(latencies) / len(latencies)

                # With ONNX weights: target 100 ms; without: 500 ms
                has_onnx = Path("models/lane_detector.onnx").exists()
                target_ms = 100 if has_onnx else 500
                note = "" if has_onnx else " (install ONNX weights for <100 ms target)"
                passed = p95 < target_ms

                print_criterion(
                    f"P95 latency < {target_ms} ms{note}",
                    passed,
                    f"P95={p95:.0f} ms  avg={avg:.0f} ms",
                )
                results.append(passed)
        except Exception as exc:
            print_criterion("Latency benchmark", False, str(exc))
            results.append(False)

    # ------------------------------------------------------------------ #
    # CRITERION 6 — PredictionResult JSON serialisation                  #
    # ------------------------------------------------------------------ #
    print("[ 6/11 ] JSON serialisation ...")
    if engine is None:
        print_criterion(
            "PredictionResult JSON serializable",
            None,
            "Skipped — engine not available",
        )
        results.append(None)
    else:
        try:
            import json

            image_paths = sorted(glob.glob("rgb/rgb_image_*.png"))
            if not image_paths:
                # Fall back to blank frame
                r = engine.predict(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                r = engine.predict(cv2.imread(image_paths[0]))

            parsed = json.loads(r.to_json())
            assert "command" in parsed and "confidence" in parsed
            preview = list(parsed.keys())[:6]
            print_criterion(
                "PredictionResult JSON serializable",
                True,
                f"keys: {preview} ...",
            )
            results.append(True)
        except Exception as exc:
            print_criterion("PredictionResult JSON serializable", False, str(exc))
            results.append(False)

    # ------------------------------------------------------------------ #
    # CRITERION 7 — All config files valid                               #
    # ------------------------------------------------------------------ #
    print("[ 7/11 ] Config file validation ...")
    try:
        import yaml

        from app.utils.config_validator import load_and_validate_config

        load_and_validate_config("configs/production.yaml")
        yaml.safe_load(open("configs/lane_detection.yaml"))
        yaml.safe_load(open("configs/scene_understanding.yaml"))
        yaml.safe_load(open("configs/decision_engine.yaml"))
        print_criterion("All config files valid", True, "4/4 configs parsed")
        results.append(True)
    except Exception as exc:
        print_criterion("All config files valid", False, str(exc))
        results.append(False)

    # ------------------------------------------------------------------ #
    # CRITERION 8 — Geometric decision logic                             #
    # ------------------------------------------------------------------ #
    print("[ 8/11 ] Geometric decision logic ...")
    try:
        from app.decision.geometric_logic import GeometricConfig, compute_geometric_decision
        from app.lane_detection.lane_geometry import LaneGeometry
        from app.scene_understanding import SceneContext

        geo = LaneGeometry(
            offset=0.4,
            curvature=0.001,
            offset_m=0.4,
            curvature_inv_m=0.001,
            left_lane_detected=True,
            right_lane_detected=True,
            left_lane_confidence=0.90,
            right_lane_confidence=0.88,
            lane_geometry_valid=True,
            vanishing_point_x=320,
            road_width_m=3.5,
            center_lane_detected=False,
            left_coeffs=None,
            right_coeffs=None,
        )
        scene = SceneContext()   # clear scene, no hazard
        cfg = GeometricConfig()  # defaults: offset_threshold=0.3

        decision = compute_geometric_decision(geo, scene, cfg)
        assert decision is not None, "Expected a DecisionResult, got None"
        assert decision.command.value == "LEFT", (
            f"offset=+0.4 m should map to LEFT, got {decision.command.value}"
        )
        print_criterion(
            "Geometric decision logic (offset → LEFT)",
            True,
            f"offset=+0.40 m → {decision.command.value}",
        )
        results.append(True)
    except Exception as exc:
        print_criterion("Geometric decision logic", False, str(exc))
        results.append(False)

    # ------------------------------------------------------------------ #
    # CRITERION 9 — Safety gate always STOPs on hazard                   #
    # ------------------------------------------------------------------ #
    print("[ 9/11 ] Safety gate — STOP on hazard ...")
    try:
        from app.decision import DecisionPath, DecisionResult, DriveCommand
        from app.decision.safety_gate import SafetyGate
        from app.scene_understanding import SceneContext

        gate = SafetyGate("configs/decision_engine.yaml")

        scene_with_hazard = SceneContext(
            immediate_hazard=True,
            hazard_reason="unit-test hazard",
        )
        proposed = DecisionResult(
            command=DriveCommand.FORWARD,
            confidence=0.9,
            decision_path=DecisionPath.GEOMETRIC,
        )
        final = gate.evaluate(scene_with_hazard, proposed)
        assert final.command == DriveCommand.STOP, (
            f"Expected STOP, got {final.command.value}"
        )
        print_criterion(
            "Safety gate STOP on hazard",
            True,
            "FORWARD → STOP when immediate_hazard=True",
        )
        results.append(True)
    except Exception as exc:
        print_criterion("Safety gate STOP on hazard", False, str(exc))
        results.append(False)

    # ------------------------------------------------------------------ #
    # CRITERION 10 — Docker Compose config valid                          #
    # ------------------------------------------------------------------ #
    print("[ 10/11 ] Docker Compose validation ...")
    try:
        proc = subprocess.run(
            ["docker", "compose", "config", "--quiet"],
            capture_output=True,
            text=True,
        )
        passed = proc.returncode == 0
        print_criterion(
            "docker-compose.yml is valid",
            passed,
            "Valid" if passed else proc.stderr[:120].strip(),
        )
        results.append(passed)
    except FileNotFoundError:
        print_criterion(
            "docker-compose.yml is valid",
            None,
            "Docker not installed — skipping",
        )
        results.append(None)

    # ------------------------------------------------------------------ #
    # CRITERION 11 — ONNX model files present (advisory)                 #
    # ------------------------------------------------------------------ #
    print("[ 11/11 ] ONNX model files ...")
    onnx_files = {
        "Lane Detector": "models/lane_detector.onnx",
        "Depth Estimator": "models/depth_estimator.onnx",
        "Decision CNN": "models/decision_cnn.onnx",
    }
    all_present = True
    for name, path in onnx_files.items():
        exists = Path(path).exists()
        if not exists:
            all_present = False
        print_criterion(
            f"ONNX: {name}",
            exists,
            path if exists else f"Missing — run training/scripts/export_onnx.py",
        )
    # ONNX presence is advisory: counts in the summary but never a hard fail
    results.append(True if all_present else None)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    passed_count = sum(1 for r in results if r is True)
    failed_count = sum(1 for r in results if r is False)
    skipped_count = sum(1 for r in results if r is None)
    total_count = len(results)

    print()
    print("━" * 60)
    print(
        f"  Phase 7 Validation: "
        f"{passed_count}/{total_count} criteria passed"
        + (f"  ({skipped_count} skipped)" if skipped_count else "")
    )
    print()

    if failed_count == 0:
        print("  \U0001f680  RoadSage is production-ready. Ship it.")
    elif failed_count <= 2:
        print("  ⚠️   Minor issues — review failures before demo.")
    else:
        print("  ❌  Fix failing criteria before demo.")

    print("━" * 60)
    print()

    return 1 if failed_count > 0 else 0


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_phase7_validation())
