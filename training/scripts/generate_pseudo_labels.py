"""
training/scripts/generate_pseudo_labels.py
===========================================

Runs the pretrained UFLDv2 lane detector + geometric decision engine on all
MNNIT RGB images and saves driving-command pseudo-labels for frames where
the lane confidence is high enough.

Output format: JSONL — one :class:`PseudoLabel` JSON object per line.

Typical usage::

    python training/scripts/generate_pseudo_labels.py \\
        --source rgb \\
        --output data/mnnit/pseudo_labels/labels.jsonl \\
        --min-confidence 0.85

The script degrades gracefully when the ONNX lane detector model is absent:
it prints a DRY RUN message explaining what would happen, then exits cleanly.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

from app.decision.geometric_logic import GeometricConfig, compute_geometric_decision
from app.decision import DecisionPath
from app.lane_detection.bev_transform import BEVConfig, BEVTransform
from app.lane_detection.lane_geometry import LaneGeometryComputer
from app.lane_detection.ufld_model import UFLDLaneDetector
from app.scene_understanding import SceneContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PseudoLabel:
    """A single pseudo-labeled driving command for one image frame.

    Attributes:
        image_path: Relative path to the source image.
        command: One of ``"FORWARD"``, ``"LEFT"``, ``"RIGHT"``, ``"STOP"``.
        confidence: Geometric decision confidence in [0, 1].
        lane_confidence: Per-frame lane detection confidence in [0, 1].
        offset_m: Signed lateral offset from lane centre in metres.
        curvature_inv_m: Signed road curvature in m⁻¹.
        decision_path: Which sub-system produced the command
            (``"geometric"`` or ``"single_lane"``).
        iteration: Self-training iteration that generated this label
            (starts at 1).
    """

    image_path: str
    command: str
    confidence: float
    lane_confidence: float
    offset_m: float
    curvature_inv_m: float
    decision_path: str
    iteration: int = 1

    def to_jsonl_line(self) -> str:
        """Serialise this label as a single JSON line (no trailing newline).

        Returns:
            A compact JSON string suitable for writing to a JSONL file.
        """
        return json.dumps(asdict(self))


@dataclass
class PseudoLabelStats:
    """Aggregated statistics for one pseudo-labeling run.

    Attributes:
        total_images: Number of images found in the source directory.
        processed: Number of images that were successfully read.
        skipped_no_lanes: Images skipped due to missing or invalid geometry.
        skipped_low_confidence: Images skipped due to confidence below threshold.
        accepted: Images that received a pseudo-label.
        command_distribution: Count of each accepted command.
        coverage_percent: ``accepted / total_images * 100``.
    """

    total_images: int = 0
    processed: int = 0
    skipped_no_lanes: int = 0
    skipped_low_confidence: int = 0
    accepted: int = 0
    command_distribution: Dict[str, int] = field(
        default_factory=lambda: {"FORWARD": 0, "LEFT": 0, "RIGHT": 0, "STOP": 0}
    )
    coverage_percent: float = 0.0

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print()
        print("=" * 52)
        print("  Pseudo-Labeling Summary")
        print("=" * 52)
        print(f"  Total images found    : {self.total_images:>6}")
        print(f"  Successfully read     : {self.processed:>6}")
        print(f"  Accepted              : {self.accepted:>6}")
        print(f"  Skipped (no lanes)    : {self.skipped_no_lanes:>6}")
        print(f"  Skipped (low conf)    : {self.skipped_low_confidence:>6}")
        print(f"  Coverage              : {self.coverage_percent:>5.1f}% of images pseudo-labeled")
        print()
        print("  Command distribution:")
        if self.accepted > 0:
            for cmd, count in self.command_distribution.items():
                pct = count / self.accepted * 100
                bar = "#" * int(pct / 5)
                print(f"    {cmd:<10} {count:>6}  ({pct:5.1f}%)  {bar}")
        else:
            print("    (no labels accepted)")
        print("=" * 52)
        print()


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    image_dir: str,
    output_path: str,
    config_path: str = "configs/decision_engine.yaml",
    lane_config_path: str = "configs/lane_detection.yaml",
    min_lane_confidence: float = 0.85,
    iteration: int = 1,
) -> PseudoLabelStats:
    """Run the lane detector + geometric engine on all MNNIT images and write pseudo-labels.

    Pseudo-labels are written **incrementally** — each accepted frame is
    flushed to the output JSONL file immediately, so a partial run can be
    resumed or interrupted without losing work.

    Only images that pass **all** of the following filters receive a label:

    1. The image can be read by OpenCV.
    2. At least one lane is detected with confidence ≥ ``min_lane_confidence``.
    3. ``geometry.lane_geometry_valid`` is ``True``.
    4. :func:`~app.decision.geometric_logic.compute_geometric_decision` returns
       a non-``None`` result via the GEOMETRIC or SINGLE_LANE path.

    Args:
        image_dir: Directory containing source images (``rgb_image_*.png``).
        output_path: Destination JSONL file path.  Opened in append mode so
            repeated runs add to the file rather than overwriting it.
        config_path: Path to ``configs/decision_engine.yaml``.
        lane_config_path: Path to ``configs/lane_detection.yaml``.
        min_lane_confidence: Minimum lane detection confidence required to
            accept a frame.
        iteration: Self-training iteration index written into each label.

    Returns:
        A :class:`PseudoLabelStats` instance summarising the run.
    """
    try:
        from tqdm import tqdm
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False
        logger.warning("tqdm not installed — progress bar disabled.")

    # ------------------------------------------------------------------
    # Load configurations
    # ------------------------------------------------------------------
    with open(config_path, "r", encoding="utf-8") as fh:
        decision_cfg = yaml.safe_load(fh)

    with open(lane_config_path, "r", encoding="utf-8") as fh:
        lane_cfg = yaml.safe_load(fh)

    geo_config = GeometricConfig.from_yaml(config_path)
    confidence_cfg = lane_cfg.get("confidence", {})

    # ------------------------------------------------------------------
    # Instantiate pipeline components
    # ------------------------------------------------------------------
    detector = UFLDLaneDetector(config_path=lane_config_path)
    bev_config = BEVConfig.from_yaml(lane_config_path)
    bev = BEVTransform(bev_config)
    geometry_computer = LaneGeometryComputer(bev, confidence_cfg)

    # Empty scene context: no obstacle detection during pseudo-labeling
    empty_scene = SceneContext()

    # ------------------------------------------------------------------
    # Discover images
    # ------------------------------------------------------------------
    pattern = os.path.join(image_dir, "rgb_image_*.png")
    image_paths = sorted(glob.glob(pattern))

    stats = PseudoLabelStats(total_images=len(image_paths))
    if not image_paths:
        logger.warning("No images found matching: %s", pattern)
        return stats

    logger.info("Found %d images in '%s'", len(image_paths), image_dir)

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Process images
    # ------------------------------------------------------------------
    iterable = tqdm(image_paths, desc="Pseudo-labeling", unit="img") if _use_tqdm else image_paths

    with open(output_path, "a", encoding="utf-8") as out_fh:
        for img_path in iterable:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            stats.processed += 1

            # Run lane detector
            detection = detector.predict(image)

            # Filter: no lanes or low confidence
            if detection.no_lanes_detected or not detection.confidence:
                stats.skipped_no_lanes += 1
                continue

            lane_conf = max(detection.confidence)
            if lane_conf < min_lane_confidence:
                stats.skipped_low_confidence += 1
                continue

            # Compute geometry
            geometry = geometry_computer.compute(detection)

            if not geometry.lane_geometry_valid:
                stats.skipped_no_lanes += 1
                continue

            # Geometric decision
            result = compute_geometric_decision(geometry, empty_scene, geo_config)

            if result is None:
                stats.skipped_no_lanes += 1
                continue

            # Only accept geometric or single-lane path labels
            if result.decision_path not in (
                DecisionPath.GEOMETRIC,
                DecisionPath.SINGLE_LANE,
            ):
                stats.skipped_no_lanes += 1
                continue

            # Write pseudo-label
            label = PseudoLabel(
                image_path=os.path.relpath(img_path),
                command=result.command.value,
                confidence=round(result.confidence, 4),
                lane_confidence=round(float(lane_conf), 4),
                offset_m=round(geometry.offset_m, 4),
                curvature_inv_m=round(geometry.curvature_inv_m, 6),
                decision_path=result.decision_path.value,
                iteration=iteration,
            )
            out_fh.write(label.to_jsonl_line() + "\n")
            out_fh.flush()

            stats.accepted += 1
            stats.command_distribution[result.command.value] += 1

    # ------------------------------------------------------------------
    # Finalise stats
    # ------------------------------------------------------------------
    stats.coverage_percent = (
        stats.accepted / stats.total_images * 100.0 if stats.total_images > 0 else 0.0
    )
    return stats


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def _dry_run_demo(image_dir: str, lane_config_path: str, config_path: str) -> None:
    """Show pipeline structure without running inference."""
    print()
    print("  DRY RUN — showing pipeline structure only.")
    print()
    try:
        bev_config = BEVConfig.from_yaml(lane_config_path)
        geo_config = GeometricConfig.from_yaml(config_path)
        print(f"  BEVConfig loaded         : output={bev_config.output_width}x{bev_config.output_height}  ppm={bev_config.pixels_per_meter}")
        print(f"  GeometricConfig loaded   : offset_threshold={geo_config.offset_threshold}  curve_threshold={geo_config.curve_threshold}")
        images = sorted(glob.glob(os.path.join(image_dir, "rgb_image_*.png")))
        print(f"  Images found             : {len(images)} in '{image_dir}/'")
        print()
        print("  Pipeline order: UFLDLaneDetector → LaneGeometryComputer → compute_geometric_decision → PseudoLabel")
        print()
        print("  To activate: run  bash models/download_models.sh  then re-run this script.")
    except Exception as exc:
        print(f"  Could not complete dry-run demo: {exc}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate driving-command pseudo-labels from MNNIT RGB images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="rgb",
                        help="Source image directory (contains rgb_image_*.png)")
    parser.add_argument("--output", default="data/mnnit/pseudo_labels/labels.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--min-confidence", type=float, default=0.85,
                        help="Minimum lane detection confidence to accept a frame")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Self-training iteration number to embed in labels")
    parser.add_argument("--config", default="configs/decision_engine.yaml",
                        help="Decision engine config path")
    parser.add_argument("--lane-config", default="configs/lane_detection.yaml",
                        help="Lane detection config path")
    args = parser.parse_args()

    # Check whether ONNX model is available
    detector_check = UFLDLaneDetector(config_path=args.lane_config)
    if not detector_check.is_ready():
        print()
        print("Lane detector ONNX not found. Download from models/download_models.sh first.")
        print("Running in DRY RUN mode — showing pipeline structure only.")
        _dry_run_demo(args.source, args.lane_config, args.config)
    else:
        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        print(f"Starting pseudo-label generation: source='{args.source}'")
        print(f"  output='{args.output}'  min_confidence={args.min_confidence}  iteration={args.iteration}")

        stats = generate_pseudo_labels(
            image_dir=args.source,
            output_path=args.output,
            config_path=args.config,
            lane_config_path=args.lane_config,
            min_lane_confidence=args.min_confidence,
            iteration=args.iteration,
        )

        stats.print_summary()
        print(f"Pseudo-labels written to: {args.output}")
