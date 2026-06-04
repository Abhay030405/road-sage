"""
training.evaluation.evaluate_decision
=======================================

Evaluates the full RoadSage decision pipeline on MNNIT images.

Supports two evaluation modes:

* **Automatic** (no ground truth) — runs every image through the engine and
  reports command distribution, latency percentiles, and decision-path
  breakdown.
* **Manual** (GT-annotated) — loads human-annotated ground truth from a JSON
  file and computes per-class accuracy, STOP precision/recall, and a
  confusion matrix.

A template generator is included to bootstrap the annotation workflow.

Usage::

    # Auto-evaluate first 200 images
    python training/evaluation/evaluate_decision.py --source data/mnnit/rgb

    # Evaluate against annotated ground truth and save confusion matrix
    python training/evaluation/evaluate_decision.py \\
        --source data/mnnit/rgb \\
        --gt data/mnnit/ground_truth.json \\
        --output-dir outputs/decision_eval/

    # Create annotation template (fill in gt_command, then pass via --gt)
    python training/evaluation/evaluate_decision.py --create-gt-template

    # Latency benchmark (500 frames by default)
    python training/evaluation/evaluate_decision.py --benchmark --n-frames 100

Exit code
---------
``0`` always — this script is diagnostic, not a hard gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from app.engine import RoadSageEngine

log = logging.getLogger(__name__)

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

_COMMANDS = ["FORWARD", "LEFT", "RIGHT", "STOP"]

# Evaluation thresholds
_ACCURACY_THRESHOLD = 0.88        # >= 88 % overall accuracy
_STOP_PRECISION_THRESHOLD = 0.99  # >= 99 % STOP precision (safety-critical)
_P95_LATENCY_MS = 100.0           # < 100 ms P95 latency


# ===========================================================================
# 1. GroundTruthSample
# ===========================================================================


@dataclass
class GroundTruthSample:
    """One human-annotated image with its expected command.

    Attributes
    ----------
    image_path:
        Path to the image file (absolute or relative to working directory).
    gt_command:
        Manually annotated expected driving command.
        One of ``"FORWARD"``, ``"LEFT"``, ``"RIGHT"``, ``"STOP"``.
    annotator_notes:
        Optional free-text notes from the annotator (e.g. "junction ahead",
        "pedestrian visible").
    """

    image_path: str
    gt_command: str
    annotator_notes: str = ""


# ===========================================================================
# 2. DecisionEvalResult
# ===========================================================================


@dataclass
class DecisionEvalResult:
    """Evaluation result for a single image through the decision pipeline.

    Attributes
    ----------
    image_path:
        Path to the source image.
    predicted_command:
        Command produced by the engine (``"FORWARD"``, ``"LEFT"``,
        ``"RIGHT"``, or ``"STOP"``).
    gt_command:
        Ground-truth command from human annotation, or ``None`` when
        running in automatic mode.
    correct:
        ``True`` when ``predicted_command == gt_command``, ``None`` when
        no ground truth is available.
    confidence:
        Fused scalar confidence in ``[0, 1]``.
    decision_path:
        Name of the sub-system that produced the command (e.g.
        ``"geometric"``, ``"ml_fallback"``, ``"safety_gate"``).
    latency_ms:
        Total wall-clock inference time in milliseconds.
    hazard_detected:
        ``True`` when the safety gate triggered a stop.
    """

    image_path: str
    predicted_command: str
    gt_command: Optional[str]
    correct: Optional[bool]
    confidence: float
    decision_path: str
    latency_ms: float
    hazard_detected: bool


# ===========================================================================
# 3. DecisionEvalStats
# ===========================================================================


@dataclass
class DecisionEvalStats:
    """Aggregate statistics across all evaluated images.

    Attributes
    ----------
    total_evaluated:
        Number of images successfully processed.
    correct:
        Number of correctly predicted commands (requires ground truth).
    accuracy:
        Overall accuracy ``correct / total_evaluated`` (0.0 when no GT).
    per_class_accuracy:
        Per-command accuracy dict, e.g. ``{"FORWARD": 0.92, ...}``.
        Empty when no ground truth is available.
    stop_precision:
        Of all *predicted* STOP commands, the fraction that matched GT STOP.
        Safety-critical metric.  NaN when no GT.
    stop_recall:
        Of all *GT* STOP commands, the fraction that were predicted as STOP.
        NaN when no GT.
    avg_confidence:
        Mean fused confidence across all images.
    avg_latency_ms:
        Mean total inference latency.
    p95_latency_ms:
        95th-percentile total inference latency.
    decision_path_distribution:
        Count of how many frames used each decision sub-system.
    command_distribution:
        Count of each predicted command across all images.
    """

    total_evaluated: int
    correct: int
    accuracy: float
    per_class_accuracy: Dict[str, float]
    stop_precision: float
    stop_recall: float
    avg_confidence: float
    avg_latency_ms: float
    p95_latency_ms: float
    decision_path_distribution: Dict[str, int]
    command_distribution: Dict[str, int]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted evaluation report with PASS/FAIL criteria.

        Criteria
        --------
        * Accuracy >= 88 %  (N/A when no GT)
        * STOP precision >= 99 %  (N/A when no GT)
        * P95 latency < 100 ms
        * No pipeline crashes (always PASS when this method is reached)
        """

        def _pf(ok: bool, na: bool = False) -> str:
            if na:
                return f"{_YELLOW}N/A{_RESET}"
            tag = "PASS" if ok else "FAIL"
            colour = _GREEN if ok else _RED
            return f"{colour}{tag}{_RESET}"

        has_gt = self.correct > 0 or self.accuracy > 0.0
        acc_ok = self.accuracy >= _ACCURACY_THRESHOLD
        stop_ok = (
            not (self.stop_precision != self.stop_precision)  # not NaN
            and self.stop_precision >= _STOP_PRECISION_THRESHOLD
        )
        lat_ok = self.p95_latency_ms < _P95_LATENCY_MS

        divider = "=" * 60
        print(divider)
        print("  Decision Pipeline — Evaluation Report")
        print(divider)
        print(f"  Total evaluated  : {self.total_evaluated}")
        if has_gt:
            print(f"  Correct          : {self.correct}")
            print(f"  Accuracy         : {self.accuracy * 100:.1f} %")
            print()
            print("  Per-class accuracy:")
            for cmd in _COMMANDS:
                acc = self.per_class_accuracy.get(cmd, float("nan"))
                print(f"    {cmd:>8} : {acc * 100:.1f} %")
            print()
            print(f"  STOP precision   : {self.stop_precision * 100:.1f} %")
            print(f"  STOP recall      : {self.stop_recall * 100:.1f} %")
        print()
        print("  Command distribution:")
        for cmd in _COMMANDS:
            cnt = self.command_distribution.get(cmd, 0)
            pct = cnt / self.total_evaluated * 100 if self.total_evaluated else 0
            print(f"    {cmd:>8} : {cnt:>5}  ({pct:.1f} %)")
        print()
        print("  Decision path distribution:")
        for path, cnt in sorted(self.decision_path_distribution.items(),
                                key=lambda x: -x[1]):
            pct = cnt / self.total_evaluated * 100 if self.total_evaluated else 0
            print(f"    {path:>20} : {cnt:>5}  ({pct:.1f} %)")
        print()
        print(f"  Avg confidence   : {self.avg_confidence:.4f}")
        print(f"  Avg latency      : {self.avg_latency_ms:.1f} ms")
        print(f"  P95 latency      : {self.p95_latency_ms:.1f} ms")
        print(divider)
        print(f"  Accuracy >= {_ACCURACY_THRESHOLD*100:.0f}%          "
              f": {_pf(acc_ok, na=not has_gt)}")
        print(f"  STOP precision >= {_STOP_PRECISION_THRESHOLD*100:.0f}% "
              f": {_pf(stop_ok, na=not has_gt)}")
        print(f"  P95 latency < {_P95_LATENCY_MS:.0f} ms       "
              f": {_pf(lat_ok)}  (got {self.p95_latency_ms:.1f} ms)")
        print(f"  No pipeline crashes     : {_GREEN}PASS{_RESET}")
        print(divider)

        # Overall verdict only when GT is available
        if has_gt:
            all_pass = acc_ok and stop_ok and lat_ok
            tag = f"{_GREEN}ALL PASS{_RESET}" if all_pass else f"{_RED}SOME FAIL{_RESET}"
            print(f"  Overall: {tag}")
            print(divider)


# ===========================================================================
# 4. load_ground_truth
# ===========================================================================


def load_ground_truth(gt_path: str) -> List[GroundTruthSample]:
    """Load human-annotated ground truth from a JSON file.

    The file must contain a JSON array where each element has at minimum
    ``"image_path"`` and ``"gt_command"`` keys.  An optional ``"notes"``
    key is loaded into :attr:`~GroundTruthSample.annotator_notes`.

    Parameters
    ----------
    gt_path:
        Path to the ground-truth JSON file.

    Returns
    -------
    List[GroundTruthSample]
        Parsed samples, or an empty list when the file is not found.
    """

    p = Path(gt_path)
    if not p.exists():
        log.warning("Ground truth file not found: %s", gt_path)
        return []

    with open(p, "r") as fh:
        raw = json.load(fh)

    samples: List[GroundTruthSample] = []
    for entry in raw:
        cmd = str(entry.get("gt_command", "")).strip().upper()
        if cmd not in _COMMANDS:
            log.warning(
                "Skipping entry with invalid gt_command %r: %s",
                cmd, entry.get("image_path", "?"),
            )
            continue
        samples.append(
            GroundTruthSample(
                image_path=str(entry["image_path"]),
                gt_command=cmd,
                annotator_notes=str(entry.get("notes", "")),
            )
        )

    log.info("Loaded %d ground-truth samples from %s", len(samples), gt_path)
    return samples


# ===========================================================================
# 5. create_gt_template
# ===========================================================================


def create_gt_template(
    image_dir: str = "rgb",
    output_path: str = "data/mnnit/ground_truth_template.json",
    n_samples: int = 100,
) -> None:
    """Create a blank annotation template for manual ground-truth labelling.

    Samples ``n_samples`` images uniformly from ``image_dir``, writes a JSON
    file with empty ``gt_command`` fields, and prints instructions.

    Parameters
    ----------
    image_dir:
        Directory containing ``rgb_image_*.png`` frames.
    output_path:
        Destination path for the template JSON file.
    n_samples:
        Number of images to include in the template.
    """

    image_root = Path(image_dir)
    all_images = sorted(image_root.glob("rgb_image_*.png"))
    if not all_images:
        log.warning("No images found in %s — template will be empty.", image_dir)

    step = max(1, len(all_images) // n_samples)
    sampled = all_images[::step][:n_samples]

    template = [
        {"image_path": str(p), "gt_command": "", "notes": ""}
        for p in sampled
    ]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(template, fh, indent=2)

    print(f"Template created at {out_path}")
    print("Open the file and fill in gt_command for each image.")
    print("Commands: FORWARD | LEFT | RIGHT | STOP")


# ===========================================================================
# 6. evaluate_pipeline
# ===========================================================================


def evaluate_pipeline(
    image_dir: str = "rgb",
    gt_path: Optional[str] = None,
    max_images: int = 200,
    output_dir: str = "outputs/decision_eval/",
) -> DecisionEvalStats:
    """Evaluate the full RoadSage decision pipeline on MNNIT images.

    When ground truth is supplied, accuracy, per-class metrics, and STOP
    precision/recall are computed.  Without ground truth the function
    reports command distribution, latency, and decision-path breakdown only.

    Per-image results are written to ``output_dir/results.jsonl``.

    Parameters
    ----------
    image_dir:
        Directory containing ``rgb_image_*.png`` frames.
    gt_path:
        Optional path to a ground-truth JSON file produced by
        :func:`create_gt_template` (after annotation).
    max_images:
        Maximum number of images to evaluate when no ground truth is
        provided.  Ignored when GT is available (all GT images are used).
    output_dir:
        Directory where ``results.jsonl`` and optional plots are written.

    Returns
    -------
    DecisionEvalStats
        Aggregate metrics across all evaluated images.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load engine & GT                                                    #
    # ------------------------------------------------------------------ #
    log.info("Loading RoadSageEngine ...")
    engine = RoadSageEngine()

    gt_samples: List[GroundTruthSample] = []
    if gt_path:
        gt_samples = load_ground_truth(gt_path)

    # ------------------------------------------------------------------ #
    # Build image list                                                    #
    # ------------------------------------------------------------------ #
    if gt_samples:
        # Use only GT-annotated images
        image_paths = [Path(s.image_path) for s in gt_samples]
        gt_map = {s.image_path: s for s in gt_samples}
        log.info("Evaluating %d ground-truth images.", len(image_paths))
    else:
        image_root = Path(image_dir)
        image_paths = sorted(image_root.glob("rgb_image_*.png"))[:max_images]
        gt_map = {}
        log.info(
            "No ground truth — evaluating first %d images from %s.",
            len(image_paths), image_dir,
        )

    # ------------------------------------------------------------------ #
    # Per-image loop                                                      #
    # ------------------------------------------------------------------ #
    results: List[DecisionEvalResult] = []
    jsonl_path = out_path / "results.jsonl"

    with open(jsonl_path, "w") as jsonl_fh:
        for img_path in tqdm(image_paths, desc="Evaluating", unit="img"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                log.warning("Cannot read %s — skipping.", img_path)
                continue

            pred = engine.predict(frame)

            # Match against GT
            key = str(img_path)
            gt_entry = gt_map.get(key)
            gt_cmd = gt_entry.gt_command if gt_entry else None
            correct = (pred.command == gt_cmd) if gt_cmd else None

            result = DecisionEvalResult(
                image_path=key,
                predicted_command=pred.command,
                gt_command=gt_cmd,
                correct=correct,
                confidence=pred.confidence,
                decision_path=pred.decision_path,
                latency_ms=pred.latency_ms.get("total", 0.0),
                hazard_detected=pred.hazard_detected,
            )
            results.append(result)

            jsonl_fh.write(
                json.dumps({
                    "image_path": key,
                    "predicted_command": pred.command,
                    "gt_command": gt_cmd,
                    "correct": correct,
                    "confidence": pred.confidence,
                    "decision_path": pred.decision_path,
                    "latency_ms": pred.latency_ms.get("total", 0.0),
                    "hazard_detected": pred.hazard_detected,
                }) + "\n"
            )

    log.info("Results written to %s", jsonl_path)

    return _compute_decision_stats(results)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------


def _compute_decision_stats(results: List[DecisionEvalResult]) -> DecisionEvalStats:
    """Compute aggregate DecisionEvalStats from per-image result list."""

    n = len(results)
    if n == 0:
        return DecisionEvalStats(
            total_evaluated=0,
            correct=0,
            accuracy=0.0,
            per_class_accuracy={},
            stop_precision=float("nan"),
            stop_recall=float("nan"),
            avg_confidence=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            decision_path_distribution={},
            command_distribution={},
        )

    latencies = [r.latency_ms for r in results]
    confidences = [r.confidence for r in results]

    # Command distribution
    from collections import Counter
    cmd_dist = dict(Counter(r.predicted_command for r in results))
    path_dist = dict(Counter(r.decision_path for r in results))

    # Accuracy metrics (only when GT is available)
    gt_results = [r for r in results if r.gt_command is not None]
    n_correct = sum(1 for r in gt_results if r.correct)
    accuracy = n_correct / len(gt_results) if gt_results else 0.0

    per_class: Dict[str, float] = {}
    if gt_results:
        for cmd in _COMMANDS:
            class_gt = [r for r in gt_results if r.gt_command == cmd]
            if class_gt:
                per_class[cmd] = sum(1 for r in class_gt if r.correct) / len(class_gt)

    # STOP precision / recall
    if gt_results:
        pred_stop = [r for r in gt_results if r.predicted_command == "STOP"]
        gt_stop = [r for r in gt_results if r.gt_command == "STOP"]
        stop_precision = (
            sum(1 for r in pred_stop if r.gt_command == "STOP") / len(pred_stop)
            if pred_stop else float("nan")
        )
        stop_recall = (
            sum(1 for r in gt_stop if r.predicted_command == "STOP") / len(gt_stop)
            if gt_stop else float("nan")
        )
    else:
        stop_precision = float("nan")
        stop_recall = float("nan")

    return DecisionEvalStats(
        total_evaluated=n,
        correct=n_correct,
        accuracy=accuracy,
        per_class_accuracy=per_class,
        stop_precision=stop_precision,
        stop_recall=stop_recall,
        avg_confidence=float(np.mean(confidences)),
        avg_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        decision_path_distribution=path_dist,
        command_distribution=cmd_dist,
    )


# ===========================================================================
# 7. plot_confusion_matrix
# ===========================================================================


def plot_confusion_matrix(
    results: List[DecisionEvalResult],
    output_dir: str = "outputs/decision_eval/",
) -> None:
    """Plot and save a 4x4 confusion matrix from GT-annotated results.

    Rows represent ground-truth labels; columns represent predicted labels.
    The diagonal (correct predictions) is highlighted in green; off-diagonal
    cells use a blue colour scale proportional to count.

    The figure is saved to ``output_dir/confusion_matrix.png``.

    Parameters
    ----------
    results:
        List of :class:`DecisionEvalResult` objects.  Results without a
        ``gt_command`` are silently ignored.
    output_dir:
        Directory where ``confusion_matrix.png`` is written.
    """

    gt_results = [r for r in results if r.gt_command is not None]
    if not gt_results:
        log.warning("No ground-truth results — skipping confusion matrix.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_cls = len(_COMMANDS)
    cmd_idx = {c: i for i, c in enumerate(_COMMANDS)}
    matrix = np.zeros((n_cls, n_cls), dtype=int)

    for r in gt_results:
        if r.gt_command in cmd_idx and r.predicted_command in cmd_idx:
            matrix[cmd_idx[r.gt_command], cmd_idx[r.predicted_command]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))

    # Base colour: blues
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)

    # Highlight diagonal green
    for i in range(n_cls):
        for j in range(n_cls):
            colour = "darkgreen" if i == j else "navy"
            alpha = 1.0 if matrix[i, j] > 0 else 0.4
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color=colour, fontsize=12, fontweight="bold",
                alpha=alpha,
            )

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(_COMMANDS, rotation=30, ha="right")
    ax.set_yticklabels(_COMMANDS)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Decision Pipeline — Confusion Matrix", fontsize=13,
                 fontweight="bold")

    plt.tight_layout()

    plot_file = out_path / "confusion_matrix.png"
    plt.savefig(str(plot_file), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved to {plot_file}")


# ===========================================================================
# 8. run_latency_benchmark
# ===========================================================================


def run_latency_benchmark(
    n_frames: int = 500,
    image_dir: str = "rgb",
) -> dict:
    """Benchmark end-to-end and per-component pipeline latency.

    Loads :class:`~app.engine.RoadSageEngine` once, then calls
    :meth:`~app.engine.RoadSageEngine.predict` on ``n_frames`` frames
    (cycling through available images when the directory has fewer frames
    than requested).

    Parameters
    ----------
    n_frames:
        Total number of frames to process.
    image_dir:
        Directory containing ``rgb_image_*.png`` frames.

    Returns
    -------
    dict
        Per-component P50 and P95 latencies plus overall mean::

            {
                "n_frames": int,
                "lane_p50": float, "lane_p95": float,
                "scene_p50": float, "scene_p95": float,
                "decision_p50": float, "decision_p95": float,
                "total_p50": float, "total_p95": float,
                "total_mean": float,
            }
    """

    image_root = Path(image_dir)
    all_images = sorted(image_root.glob("rgb_image_*.png"))
    if not all_images:
        log.error("No images found in %s for benchmark.", image_dir)
        return {"error": "no images found"}

    log.info("Loading RoadSageEngine for benchmark ...")
    engine = RoadSageEngine()

    lane_ms_list: List[float] = []
    scene_ms_list: List[float] = []
    decision_ms_list: List[float] = []
    total_ms_list: List[float] = []

    for i in tqdm(range(n_frames), desc="Benchmarking", unit="frame"):
        img_path = all_images[i % len(all_images)]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        result = engine.predict(frame)
        lat = result.latency_ms
        lane_ms_list.append(lat.get("lane", 0.0))
        scene_ms_list.append(lat.get("scene", 0.0))
        decision_ms_list.append(lat.get("decision", 0.0))
        total_ms_list.append(lat.get("total", 0.0))

    def _p(arr: List[float], pct: int) -> float:
        return float(np.percentile(arr, pct)) if arr else 0.0

    total_p95 = _p(total_ms_list, 95)
    result_dict = {
        "n_frames": n_frames,
        "lane_p50": _p(lane_ms_list, 50),
        "lane_p95": _p(lane_ms_list, 95),
        "scene_p50": _p(scene_ms_list, 50),
        "scene_p95": _p(scene_ms_list, 95),
        "decision_p50": _p(decision_ms_list, 50),
        "decision_p95": _p(decision_ms_list, 95),
        "total_p50": _p(total_ms_list, 50),
        "total_p95": total_p95,
        "total_mean": float(np.mean(total_ms_list)) if total_ms_list else 0.0,
    }

    # PASS / WARN / FAIL verdict
    if total_p95 < 100.0:
        verdict = f"{_GREEN}PASS{_RESET}  (P95 = {total_p95:.1f} ms < 100 ms)"
    elif total_p95 < 200.0:
        verdict = f"{_YELLOW}WARN{_RESET}  (P95 = {total_p95:.1f} ms, target < 100 ms)"
    else:
        verdict = f"{_RED}FAIL{_RESET}  (P95 = {total_p95:.1f} ms > 200 ms)"

    print("\nLatency Benchmark Results")
    print(f"  Frames         : {n_frames}")
    print(f"  Lane   P50/P95 : {result_dict['lane_p50']:.1f} / {result_dict['lane_p95']:.1f} ms")
    print(f"  Scene  P50/P95 : {result_dict['scene_p50']:.1f} / {result_dict['scene_p95']:.1f} ms")
    print(f"  Decision P50/P95: {result_dict['decision_p50']:.1f} / {result_dict['decision_p95']:.1f} ms")
    print(f"  Total  P50/P95 : {result_dict['total_p50']:.1f} / {result_dict['total_p95']:.1f} ms")
    print(f"  Total  mean    : {result_dict['total_mean']:.1f} ms")
    print(f"  Verdict        : {verdict}")

    return result_dict


# ===========================================================================
# 9. __main__
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the RoadSage decision pipeline on MNNIT images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        default="rgb",
        metavar="DIR",
        help="Directory containing rgb_image_*.png frames.",
    )
    p.add_argument(
        "--gt",
        default=None,
        metavar="JSON",
        help="Ground-truth annotation file (JSON array).  Optional.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=200,
        metavar="N",
        help="Max images to evaluate when no GT is provided.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark after evaluation.",
    )
    p.add_argument(
        "--n-frames",
        type=int,
        default=100,
        metavar="N",
        help="Number of frames for the latency benchmark.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/decision_eval/",
        metavar="DIR",
        help="Directory for results.jsonl, confusion matrix, and plots.",
    )
    p.add_argument(
        "--create-gt-template",
        action="store_true",
        help="Create a blank annotation template JSON file and exit.",
    )
    p.add_argument(
        "--template-output",
        default="data/mnnit/ground_truth_template.json",
        metavar="JSON",
        help="Output path for the GT annotation template.",
    )
    p.add_argument(
        "--template-n",
        type=int,
        default=100,
        metavar="N",
        help="Number of images to include in the annotation template.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the decision evaluation script.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv``).

    Returns
    -------
    int
        Always ``0`` — this script is diagnostic, not a hard gate.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------ #
    # Template creation mode                                              #
    # ------------------------------------------------------------------ #
    if args.create_gt_template:
        create_gt_template(
            image_dir=args.source,
            output_path=args.template_output,
            n_samples=args.template_n,
        )
        return 0

    # ------------------------------------------------------------------ #
    # Main evaluation                                                     #
    # ------------------------------------------------------------------ #
    stats = evaluate_pipeline(
        image_dir=args.source,
        gt_path=args.gt,
        max_images=args.max_images,
        output_dir=args.output_dir,
    )
    stats.print_report()

    # ------------------------------------------------------------------ #
    # Confusion matrix (only when GT was provided)                        #
    # ------------------------------------------------------------------ #
    if args.gt:
        # Re-load results from JSONL to build the confusion matrix
        jsonl_path = Path(args.output_dir) / "results.jsonl"
        if jsonl_path.exists():
            cm_results: List[DecisionEvalResult] = []
            with open(jsonl_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cm_results.append(
                        DecisionEvalResult(
                            image_path=rec["image_path"],
                            predicted_command=rec["predicted_command"],
                            gt_command=rec.get("gt_command"),
                            correct=rec.get("correct"),
                            confidence=rec["confidence"],
                            decision_path=rec["decision_path"],
                            latency_ms=rec["latency_ms"],
                            hazard_detected=rec["hazard_detected"],
                        )
                    )
            plot_confusion_matrix(cm_results, output_dir=args.output_dir)

    # ------------------------------------------------------------------ #
    # Latency benchmark                                                   #
    # ------------------------------------------------------------------ #
    if args.benchmark:
        print()
        run_latency_benchmark(n_frames=args.n_frames, image_dir=args.source)

    return 0


if __name__ == "__main__":
    sys.exit(main())
