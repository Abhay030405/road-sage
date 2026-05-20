"""Filter raw MNNIT campus road images through 4 quality gates.

Pipeline order per image:
    blur → brightness → road_coverage → duplicate

Passing images are copied to the verified output directory.

Usage:
    python training/scripts/filter_dataset.py
    python training/scripts/filter_dataset.py --source rgb --output data/mnnit/verified
    python training/scripts/filter_dataset.py --source rgb --output data/mnnit/verified --config configs/production.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Quality gate functions
# ---------------------------------------------------------------------------


def check_blur(image: np.ndarray, threshold: float = 50.0) -> tuple[bool, float]:
    """Check whether an image is sharp enough using Laplacian variance.

    Blurry images have low Laplacian variance because their high-frequency
    edge content has been suppressed.

    Args:
        image: BGR image array as returned by ``cv2.imread``.
        threshold: Minimum Laplacian variance required to pass.  Images with
            variance below this value are considered too blurry.

    Returns:
        A ``(passes, score)`` tuple where *passes* is ``True`` when the image
        is sharp enough and *score* is the raw Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score >= threshold, score


def check_brightness(
    image: np.ndarray,
    min_val: float = 30.0,
    max_val: float = 220.0,
) -> tuple[bool, float]:
    """Check whether an image has acceptable overall brightness.

    Very dark images (underexposed) and very bright images (overexposed or
    washed-out) are both rejected.

    Args:
        image: BGR image array as returned by ``cv2.imread``.
        min_val: Minimum acceptable mean grayscale pixel value (0–255).
        max_val: Maximum acceptable mean grayscale pixel value (0–255).

    Returns:
        A ``(passes, mean_brightness)`` tuple where *passes* is ``True`` when
        the mean brightness falls within ``[min_val, max_val]``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    passes = min_val <= mean_brightness <= max_val
    return passes, mean_brightness


def check_road_coverage(
    image: np.ndarray,
    min_ratio: float = 0.20,
) -> tuple[bool, float]:
    """Check whether enough of the image is covered by road-like pixels.

    Road/asphalt surfaces appear as low-saturation, mid-value gray tones in
    HSV space.  The mask targets:

    * Hue:        0–180 (any hue; saturation distinguishes road from sky)
    * Saturation: 0–50  (low saturation → gray/asphalt, not coloured objects)
    * Value:      30–200 (excludes pitch-black shadows and blown-out regions)

    Args:
        image: BGR image array as returned by ``cv2.imread``.
        min_ratio: Minimum fraction of pixels that must match the road mask.

    Returns:
        A ``(passes, road_ratio)`` tuple where *passes* is ``True`` when the
        road pixel ratio meets *min_ratio*.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 30], dtype=np.uint8)
    upper = np.array([180, 50, 200], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    road_ratio = float(mask.sum() / 255) / float(image.shape[0] * image.shape[1])
    passes = road_ratio >= min_ratio
    return passes, road_ratio


def check_duplicate(
    image_path: str,
    existing_hashes: list[imagehash.ImageHash],
    threshold: float = 0.98,
) -> tuple[bool, str]:
    """Check whether an image is a near-duplicate of one already accepted.

    Uses perceptual hashing (pHash) so that minor JPEG re-encodings, small
    crops, or brightness tweaks do not produce false negatives.

    Similarity is derived from the Hamming distance between 64-bit pHashes::

        similarity = 1 - (hash_distance / 64.0)

    Args:
        image_path: Filesystem path to the candidate image file.
        existing_hashes: List of ``imagehash.ImageHash`` objects computed from
            previously accepted images.  This list is mutated by the caller
            (a new hash is appended when the image passes all gates).
        threshold: Similarity above which the image is considered a duplicate
            and rejected.  ``0.98`` means the images differ by at most ~1 bit.

    Returns:
        A ``(is_duplicate, hash_str)`` tuple.  *is_duplicate* is ``True`` when
        the image is too similar to an already-accepted image.  *hash_str* is
        the hex string representation of this image's pHash.
    """
    pil_img = Image.open(image_path)
    current_hash = imagehash.phash(pil_img)
    hash_str = str(current_hash)

    for existing in existing_hashes:
        distance = current_hash - existing
        similarity = 1.0 - (distance / 64.0)
        if similarity > threshold:
            return True, hash_str

    return False, hash_str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_filter_pipeline(
    source_dir: str,
    output_dir: str,
    config: dict,
) -> dict:
    """Run all four quality gates over every image in *source_dir*.

    Images that pass all gates are copied to *output_dir*.  Rejected images are
    logged at DEBUG level with the name of the failing gate.

    Gate order: **blur → brightness → road_coverage → duplicate**

    Thresholds are read from *config* under the ``quality_filters`` key.  Any
    threshold not present in *config* falls back to the function defaults.

    Args:
        source_dir: Root directory to search recursively for images.
        output_dir: Destination directory for accepted images.  Created if it
            does not already exist.
        config: Loaded YAML configuration dictionary.  Relevant section::

            quality_filters:
              blur_laplacian_threshold: 50
              brightness_min: 30
              brightness_max: 220
              road_coverage_min: 0.20
              dedup_similarity_threshold: 0.98

    Returns:
        A statistics dictionary with the following keys:

        * ``total_processed``   – total images examined
        * ``passed``            – images copied to *output_dir*
        * ``rejected_blur``     – rejected by the blur gate
        * ``rejected_brightness``      – rejected by the brightness gate
        * ``rejected_road_coverage``   – rejected by the road coverage gate
        * ``rejected_duplicate``       – rejected as near-duplicates
        * ``acceptance_rate``   – ``passed / total_processed`` (0.0 if none processed)
    """
    qf = config.get("quality_filters", {})
    blur_threshold = float(qf.get("blur_laplacian_threshold", 50))
    brightness_min = float(qf.get("brightness_min", 30))
    brightness_max = float(qf.get("brightness_max", 220))
    road_min = float(qf.get("road_coverage_min", 0.20))
    dedup_threshold = float(qf.get("dedup_similarity_threshold", 0.98))

    os.makedirs(output_dir, exist_ok=True)

    # Collect all candidate image paths up front for an accurate progress bar.
    image_paths: list[Path] = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if Path(fname).suffix.lower() in _IMAGE_EXTS:
                image_paths.append(Path(root) / fname)

    stats = {
        "total_processed": 0,
        "passed": 0,
        "rejected_blur": 0,
        "rejected_brightness": 0,
        "rejected_road_coverage": 0,
        "rejected_duplicate": 0,
        "acceptance_rate": 0.0,
    }

    existing_hashes: list[imagehash.ImageHash] = []

    for img_path in tqdm(image_paths, desc="Filtering images", unit="img"):
        stats["total_processed"] += 1
        name = img_path.name

        # --- Load image -------------------------------------------------------
        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("Could not read '%s' — skipping.", name)
            continue

        # --- Gate 1: blur -----------------------------------------------------
        passes, blur_score = check_blur(image, threshold=blur_threshold)
        if not passes:
            stats["rejected_blur"] += 1
            log.debug("REJECT blur   | score=%.1f < %.1f | %s", blur_score, blur_threshold, name)
            continue

        # --- Gate 2: brightness -----------------------------------------------
        passes, brightness = check_brightness(image, min_val=brightness_min, max_val=brightness_max)
        if not passes:
            stats["rejected_brightness"] += 1
            log.debug(
                "REJECT bright | mean=%.1f not in [%.0f, %.0f] | %s",
                brightness, brightness_min, brightness_max, name,
            )
            continue

        # --- Gate 3: road coverage --------------------------------------------
        passes, road_ratio = check_road_coverage(image, min_ratio=road_min)
        if not passes:
            stats["rejected_road_coverage"] += 1
            log.debug("REJECT road   | ratio=%.3f < %.2f | %s", road_ratio, road_min, name)
            continue

        # --- Gate 4: duplicate ------------------------------------------------
        is_dup, hash_str = check_duplicate(str(img_path), existing_hashes, threshold=dedup_threshold)
        if is_dup:
            stats["rejected_duplicate"] += 1
            log.debug("REJECT dup    | hash=%s | %s", hash_str, name)
            continue

        # --- All gates passed -------------------------------------------------
        existing_hashes.append(imagehash.hex_to_hash(hash_str))
        dest = Path(output_dir) / name
        shutil.copy2(str(img_path), str(dest))
        stats["passed"] += 1
        log.debug("ACCEPT        | blur=%.1f bright=%.1f road=%.3f | %s", blur_score, brightness, road_ratio, name)

    total = stats["total_processed"]
    stats["acceptance_rate"] = stats["passed"] / total if total > 0 else 0.0

    # --- Summary table --------------------------------------------------------
    rejected_total = total - stats["passed"]
    print("\n" + "=" * 52)
    print(f"{'FILTER PIPELINE SUMMARY':^52}")
    print("=" * 52)
    print(f"  {'Source directory':<30} {source_dir}")
    print(f"  {'Output directory':<30} {output_dir}")
    print("-" * 52)
    print(f"  {'Total images processed':<30} {total:>8}")
    print(f"  {'Passed':<30} {stats['passed']:>8}  ({stats['acceptance_rate']:.1%})")
    print(f"  {'Rejected (total)':<30} {rejected_total:>8}")
    print(f"    {'-> blur':<28} {stats['rejected_blur']:>8}")
    print(f"    {'-> brightness':<28} {stats['rejected_brightness']:>8}")
    print(f"    {'-> road coverage':<28} {stats['rejected_road_coverage']:>8}")
    print(f"    {'-> duplicate':<28} {stats['rejected_duplicate']:>8}")
    print("=" * 52 + "\n")

    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter MNNIT campus road images through 4 quality gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="rgb",
        help="Directory containing raw source images.",
    )
    parser.add_argument(
        "--output",
        default="data/mnnit/verified",
        help="Destination directory for images that pass all gates.",
    )
    parser.add_argument(
        "--config",
        default="configs/production.yaml",
        help="Path to the RoadSage YAML config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level rejection messages.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        log.info("Loaded config from '%s'.", config_path)
    else:
        log.warning("Config file '%s' not found — using default thresholds.", config_path)

    stats = run_filter_pipeline(
        source_dir=args.source,
        output_dir=args.output,
        config=config,
    )
