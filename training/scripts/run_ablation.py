"""
training.scripts.run_ablation
==============================

Ablation study: measures the contribution of each RoadSage pipeline
component by running with one component disabled at a time and comparing
the resulting command distributions, stability, and safety metrics.

The five variants are:

============================================  ==================================
Variant name                                  What is disabled
============================================  ==================================
Baseline                                      Nothing — all components active
No Scene Understanding                        Scene analyzer returns empty ctx
No BEV Transform                              BEV warp is bypassed (identity)
No Confidence Gate                            min_confidence forced to 0.0
No Temporal Smoothing                         Temporal buffer cleared each frame
============================================  ==================================

**Interpretation guide**

* A large drop in STOP rate vs. Baseline when scene understanding is removed
  shows how much the scene module contributes to hazard avoidance.
* A spike in flicker count without temporal smoothing quantifies jitter.
* A rise in unsafe commands without the confidence gate shows how many
  uncertain frames would have produced actions in production.

Usage::

    python training/scripts/run_ablation.py \\
        --source data/mnnit/rgb \\
        --n-images 50 \\
        --output-dir outputs/ablation/

Exit code
---------
``0`` always — this script is diagnostic, not a hard gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
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
from app.scene_understanding import SceneContext

log = logging.getLogger(__name__)

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

# Fixed scale factor applied to raw pixel offset when BEV is disabled.
# Converts approximate pixel distance from image centre to a rough metre
# estimate without a calibrated perspective warp.
_RAW_PIXEL_TO_METRE = 0.003


# ===========================================================================
# 1. AblationConfig
# ===========================================================================


@dataclass
class AblationConfig:
    """Configuration for one ablation variant.

    Each flag disables a single pipeline component; all others remain active.

    Attributes
    ----------
    name:
        Human-readable label for this variant (used in tables and plots).
    disable_scene:
        When ``True``, replace the scene analyzer's output with an empty
        :class:`~app.scene_understanding.SceneContext` so the decision
        engine sees no obstacles or surface hazards.
    disable_bev:
        When ``True``, bypass the Bird's-Eye-View perspective warp and
        substitute a raw-pixel-offset estimate scaled by
        ``_RAW_PIXEL_TO_METRE``.
    disable_confidence_gate:
        When ``True``, set the safety gate's ``min_confidence`` threshold
        to ``0.0`` so low-confidence predictions are never forced to STOP.
    disable_temporal_smoothing:
        When ``True``, clear the engine's temporal buffer after each frame
        so consecutive predictions are fully independent (no vote smoothing).
    description:
        Optional extra context about the variant, used in JSON output.
    """

    name: str
    disable_scene: bool = False
    disable_bev: bool = False
    disable_confidence_gate: bool = False
    disable_temporal_smoothing: bool = False
    description: str = ""


# ===========================================================================
# 2. AblationResult
# ===========================================================================


@dataclass
class AblationResult:
    """Aggregate metrics for one ablation variant.

    Attributes
    ----------
    config:
        The :class:`AblationConfig` that produced this result.
    total_frames:
        Number of frames successfully processed.
    command_distribution:
        Count of each predicted command across all frames.
    stop_rate:
        Fraction of frames where command was STOP.
    forward_rate:
        Fraction of frames where command was FORWARD.
    avg_confidence:
        Mean fused confidence across all frames.
    avg_latency_ms:
        Mean total inference latency in milliseconds.
    flicker_count:
        Number of three-frame windows where all three consecutive commands
        were different (``cmd[i-2] != cmd[i-1] != cmd[i]`` and
        ``cmd[i-2] != cmd[i]``), indicating unstable oscillation.
    unsafe_commands:
        Frames where ``confidence < 0.6`` and ``command != "STOP"``.
        These are frames that would have issued a directional command despite
        high uncertainty.
    """

    config: AblationConfig
    total_frames: int
    command_distribution: Dict[str, int]
    stop_rate: float
    forward_rate: float
    avg_confidence: float
    avg_latency_ms: float
    flicker_count: int
    unsafe_commands: int


# ===========================================================================
# 3. run_ablation_variant
# ===========================================================================


def run_ablation_variant(
    images: List[np.ndarray],
    engine: RoadSageEngine,
    config: AblationConfig,
) -> AblationResult:
    """Run the pipeline on a list of frames with one component disabled.

    Applies monkey-patches to ``engine``'s internal components for the
    duration of this function, then restores originals before returning.

    Parameters
    ----------
    images:
        Pre-loaded BGR frames to process.
    engine:
        A fully initialised :class:`~app.engine.RoadSageEngine` instance.
        Internal state (temporal buffer, latency history) is reset before
        processing to ensure results are independent across variants.
    config:
        Specifies which component to disable for this variant.

    Returns
    -------
    AblationResult
        Aggregate metrics for this variant.
    """

    # ------------------------------------------------------------------ #
    # Save originals                                                      #
    # ------------------------------------------------------------------ #
    original_analyze = engine._scene_analyzer.analyze
    original_transform_image = engine._bev.transform_image
    original_transform_points = engine._bev.transform_points
    original_min_confidence = engine._safety_gate._config.min_confidence

    # Reset engine state so variants don't contaminate each other
    engine._temporal_buffer._buffer.clear()
    engine._latency_history.clear()

    # ------------------------------------------------------------------ #
    # Apply patches                                                       #
    # ------------------------------------------------------------------ #
    if config.disable_scene:
        # Replace scene analyzer output with a fully empty, hazard-free context
        engine._scene_analyzer.analyze = lambda img: SceneContext()

    if config.disable_bev:
        # Replace BEV warp with identity — geometry computer works in raw
        # image space. transform_image returns the frame unchanged;
        # transform_points returns points unchanged.
        engine._bev.transform_image = lambda img: img
        engine._bev.transform_points = lambda pts: pts

    if config.disable_confidence_gate:
        # Disable the confidence threshold — never force STOP due to
        # low confidence (scene hazards still trigger STOP).
        engine._safety_gate._config.min_confidence = 0.0

    # ------------------------------------------------------------------ #
    # Per-frame inference                                                 #
    # ------------------------------------------------------------------ #
    commands: List[str] = []
    confidences: List[float] = []
    latencies: List[float] = []

    for frame in tqdm(images, desc=f"  {config.name}", unit="frame", leave=False):
        result = engine.predict(frame)

        commands.append(result.command)
        confidences.append(result.confidence)
        latencies.append(result.latency_ms.get("total", 0.0))

        if config.disable_temporal_smoothing:
            # Prevent history from influencing the next frame
            engine._temporal_buffer._buffer.clear()

    # ------------------------------------------------------------------ #
    # Restore originals                                                   #
    # ------------------------------------------------------------------ #
    engine._scene_analyzer.analyze = original_analyze
    engine._bev.transform_image = original_transform_image
    engine._bev.transform_points = original_transform_points
    engine._safety_gate._config.min_confidence = original_min_confidence
    engine._temporal_buffer._buffer.clear()

    # ------------------------------------------------------------------ #
    # Compute metrics                                                     #
    # ------------------------------------------------------------------ #
    n = len(commands)
    if n == 0:
        return AblationResult(
            config=config,
            total_frames=0,
            command_distribution={},
            stop_rate=0.0,
            forward_rate=0.0,
            avg_confidence=0.0,
            avg_latency_ms=0.0,
            flicker_count=0,
            unsafe_commands=0,
        )

    cmd_dist = dict(Counter(commands))

    # Flicker: three consecutive ALL-DIFFERENT commands
    flicker = 0
    for i in range(2, n):
        a, b, c = commands[i - 2], commands[i - 1], commands[i]
        if a != b and b != c and a != c:
            flicker += 1

    # Unsafe: confidence < 0.6 AND not STOP
    unsafe = sum(
        1 for cmd, conf in zip(commands, confidences)
        if conf < 0.6 and cmd != "STOP"
    )

    return AblationResult(
        config=config,
        total_frames=n,
        command_distribution=cmd_dist,
        stop_rate=cmd_dist.get("STOP", 0) / n,
        forward_rate=cmd_dist.get("FORWARD", 0) / n,
        avg_confidence=float(np.mean(confidences)),
        avg_latency_ms=float(np.mean(latencies)),
        flicker_count=flicker,
        unsafe_commands=unsafe,
    )


# ===========================================================================
# 4. run_full_ablation
# ===========================================================================


def run_full_ablation(
    image_dir: str = "rgb",
    n_images: int = 50,
    output_dir: str = "outputs/ablation/",
) -> List[AblationResult]:
    """Run all five ablation variants and print a comparison table.

    Variants
    --------
    0. **Baseline** — all components active.
    1. **No Scene Understanding** — scene analyzer returns empty context.
    2. **No BEV Transform** — BEV warp replaced by identity.
    3. **No Confidence Gate** — min_confidence forced to 0.0.
    4. **No Temporal Smoothing** — temporal buffer cleared after each frame.

    Parameters
    ----------
    image_dir:
        Directory containing ``rgb_image_*.png`` frames.
    n_images:
        Number of images to load.  All five variants run on the same set.
    output_dir:
        Directory where ``ablation_results.json`` is written.

    Returns
    -------
    List[AblationResult]
        One result per variant, in the order listed above.
    """

    # ------------------------------------------------------------------ #
    # Define variants                                                     #
    # ------------------------------------------------------------------ #
    variants = [
        AblationConfig(
            "Baseline",
            description="All components enabled — reference point.",
        ),
        AblationConfig(
            "No Scene Understanding",
            disable_scene=True,
            description="Scene analyzer replaced with empty SceneContext.",
        ),
        AblationConfig(
            "No BEV Transform",
            disable_bev=True,
            description="BEV perspective warp bypassed; raw pixel geometry used.",
        ),
        AblationConfig(
            "No Confidence Gate",
            disable_confidence_gate=True,
            description="Safety gate min_confidence set to 0.0.",
        ),
        AblationConfig(
            "No Temporal Smoothing",
            disable_temporal_smoothing=True,
            description="Temporal buffer cleared after each frame.",
        ),
    ]

    # ------------------------------------------------------------------ #
    # Load images                                                         #
    # ------------------------------------------------------------------ #
    image_root = Path(image_dir)
    all_paths = sorted(image_root.glob("rgb_image_*.png"))[:n_images]
    if not all_paths:
        log.error("No images found in %s — aborting ablation.", image_dir)
        return []

    log.info("Loading %d images ...", len(all_paths))
    images: List[np.ndarray] = []
    for p in all_paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            images.append(frame)
        else:
            log.warning("Could not read %s — skipping.", p)

    if not images:
        log.error("No images could be read.")
        return []

    log.info("Loaded %d frames.  Running ablation study ...", len(images))

    # ------------------------------------------------------------------ #
    # Load engine once                                                    #
    # ------------------------------------------------------------------ #
    log.info("Loading RoadSageEngine ...")
    engine = RoadSageEngine()

    # ------------------------------------------------------------------ #
    # Run variants                                                        #
    # ------------------------------------------------------------------ #
    results: List[AblationResult] = []
    for cfg in variants:
        log.info("Running variant: %s", cfg.name)
        result = run_ablation_variant(images, engine, cfg)
        results.append(result)

    # ------------------------------------------------------------------ #
    # Print comparison table                                              #
    # ------------------------------------------------------------------ #
    _print_comparison_table(results)

    # ------------------------------------------------------------------ #
    # Save JSON                                                           #
    # ------------------------------------------------------------------ #
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "ablation_results.json"
    serialisable = []
    for r in results:
        serialisable.append({
            "name": r.config.name,
            "description": r.config.description,
            "disable_scene": r.config.disable_scene,
            "disable_bev": r.config.disable_bev,
            "disable_confidence_gate": r.config.disable_confidence_gate,
            "disable_temporal_smoothing": r.config.disable_temporal_smoothing,
            "total_frames": r.total_frames,
            "command_distribution": r.command_distribution,
            "stop_rate": round(r.stop_rate, 4),
            "forward_rate": round(r.forward_rate, 4),
            "avg_confidence": round(r.avg_confidence, 4),
            "avg_latency_ms": round(r.avg_latency_ms, 2),
            "flicker_count": r.flicker_count,
            "unsafe_commands": r.unsafe_commands,
        })

    with open(json_file, "w") as fh:
        json.dump(serialisable, fh, indent=2)

    log.info("Ablation results saved to %s", json_file)
    return results


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------


def _print_comparison_table(results: List[AblationResult]) -> None:
    """Print a formatted comparison table of ablation results."""

    header = (
        f"{'Variant':<25} | {'STOP%':>6} | {'FWD%':>5} | "
        f"{'Avg Conf':>8} | {'Latency':>8} | {'Flicker':>7} | {'Unsafe':>6}"
    )
    separator = "-" * len(header)

    print()
    print(separator)
    print(header)
    print(separator)

    baseline = results[0] if results else None

    for r in results:
        stop_pct = r.stop_rate * 100
        fwd_pct = r.forward_rate * 100

        # Highlight regressions vs. baseline in yellow
        if baseline and r.config.name != "Baseline":
            stop_delta = r.stop_rate - baseline.stop_rate
            unsafe_delta = r.unsafe_commands - baseline.unsafe_commands
            flicker_delta = r.flicker_count - baseline.flicker_count
            stop_col = _YELLOW if abs(stop_delta) > 0.05 else ""
            unsafe_col = _YELLOW if unsafe_delta > 2 else ""
            flicker_col = _YELLOW if flicker_delta > 2 else ""
        else:
            stop_col = unsafe_col = flicker_col = ""

        print(
            f"{r.config.name:<25} | "
            f"{stop_col}{stop_pct:5.1f}%{_RESET} | "
            f"{fwd_pct:5.1f}% | "
            f"{r.avg_confidence:8.3f} | "
            f"{r.avg_latency_ms:6.1f} ms | "
            f"{flicker_col}{r.flicker_count:7d}{_RESET} | "
            f"{unsafe_col}{r.unsafe_commands:6d}{_RESET}"
        )

    print(separator)
    print()


# ===========================================================================
# 5. plot_ablation_results
# ===========================================================================


def plot_ablation_results(
    results: List[AblationResult],
    output_dir: str = "outputs/ablation/",
) -> None:
    """Create and save a grouped bar chart comparing ablation variants.

    Three groups of bars are plotted side by side for each variant:

    * **STOP rate** (higher = more conservative / safer)
    * **Flicker count** (lower = more stable)
    * **Unsafe commands** (lower = safer)

    The figure is saved to ``output_dir/ablation_comparison.png``.

    Parameters
    ----------
    results:
        List of :class:`AblationResult` objects from
        :func:`run_full_ablation`.
    output_dir:
        Directory where ``ablation_comparison.png`` is written.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    names = [r.config.name for r in results]
    stop_rates = [r.stop_rate * 100 for r in results]
    flicker_counts = [float(r.flicker_count) for r in results]
    unsafe_counts = [float(r.unsafe_commands) for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Ablation Study — Component Contribution", fontsize=14, fontweight="bold"
    )

    # ---- Panel 1: STOP rate ----
    ax = axes[0]
    bars = ax.bar(x, stop_rates, color="steelblue", edgecolor="white", width=0.6)
    if results:
        baseline_stop = stop_rates[0]
        ax.axhline(baseline_stop, color="black", linestyle="--", linewidth=1.0,
                   label=f"Baseline ({baseline_stop:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("STOP rate (%)")
    ax.set_title("STOP Rate\n(higher = more conservative)")
    ax.legend(fontsize=7)
    _annotate_bars(ax, bars, fmt="{:.1f}%")

    # ---- Panel 2: Flicker count ----
    ax = axes[1]
    bars = ax.bar(x, flicker_counts, color="darkorange", edgecolor="white", width=0.6)
    if results:
        ax.axhline(flicker_counts[0], color="black", linestyle="--", linewidth=1.0,
                   label=f"Baseline ({flicker_counts[0]:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Flicker count")
    ax.set_title("Flicker Count\n(lower = more stable)")
    ax.legend(fontsize=7)
    _annotate_bars(ax, bars, fmt="{:.0f}")

    # ---- Panel 3: Unsafe commands ----
    ax = axes[2]
    bars = ax.bar(x, unsafe_counts, color="crimson", edgecolor="white", width=0.6)
    if results:
        ax.axhline(unsafe_counts[0], color="black", linestyle="--", linewidth=1.0,
                   label=f"Baseline ({unsafe_counts[0]:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Unsafe command count")
    ax.set_title("Unsafe Commands\n(lower = safer)")
    ax.legend(fontsize=7)
    _annotate_bars(ax, bars, fmt="{:.0f}")

    plt.tight_layout()

    plot_file = out_path / "ablation_comparison.png"
    plt.savefig(str(plot_file), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Ablation plot saved to {plot_file}")


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    """Add value labels above each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(height * 0.02, 0.3),
            fmt.format(height),
            ha="center", va="bottom", fontsize=7,
        )


# ===========================================================================
# 6. __main__
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation study: measure RoadSage component contributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        default="rgb",
        metavar="DIR",
        help="Directory containing rgb_image_*.png frames.",
    )
    p.add_argument(
        "--n-images",
        type=int,
        default=50,
        metavar="N",
        help="Number of images to run per variant.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/ablation/",
        metavar="DIR",
        help="Directory for JSON results and the comparison plot.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ablation study script.

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

    results = run_full_ablation(
        image_dir=args.source,
        n_images=args.n_images,
        output_dir=args.output_dir,
    )

    if not results:
        log.error("No ablation results produced — check image directory.")
        return 0

    plot_ablation_results(results, output_dir=args.output_dir)

    print(f"Ablation study complete. Results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
