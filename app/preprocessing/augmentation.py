"""Image augmentation pipeline for MNNIT campus road images.

MNNIT-specific conditions simulated:
  - Morning haze / fog (RandomFog)
  - Tree-cast shadows on road surface (RandomShadow)
  - Variable lighting from sunrise through afternoon (RandomBrightnessContrast, CLAHE)
  - Rain streaks during monsoon (RandomRain)
  - Slight camera shake / motion blur (GaussianBlur)
  - Perspective shifts from potholes / speed bumps (Perspective)

Usage::

    augmentor = RoadSageAugmentor(mode="train")
    aug_image = augmentor.augment(bgr_image)

    # with lane points
    aug_image, aug_points = augmentor.augment_with_lanes(bgr_image, lane_points)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml

try:
    import albumentations as A
except ImportError as _albu_err:
    raise ImportError(
        "albumentations is required for the augmentation pipeline. "
        "Install it with:  pip install albumentations"
    ) from _albu_err

log = logging.getLogger(__name__)

# Inference resize target matches lane-detector input_size in production.yaml
_INFERENCE_W, _INFERENCE_H = 800, 288

# ImageNet normalisation constants (used for all modes)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Mirrors the ``augmentation`` section of ``configs/production.yaml``.

    All probability fields control whether the corresponding transform is
    applied to any given image (Bernoulli trial per image).

    Attributes:
        enabled: Master switch.  When ``False`` only Normalize is applied
            even in training mode.
        clahe_clip_limit: Contrast limit for CLAHE histogram clipping.
        clahe_tile_grid_size: Tile grid dimensions ``(rows, cols)`` for CLAHE.
        clahe_probability: Probability of applying CLAHE.
        random_shadow_probability: Probability of adding a synthetic shadow.
        brightness_limit: Max absolute brightness shift (±fraction of 255).
        contrast_limit: Max absolute contrast shift.
        brightness_contrast_probability: Probability of applying brightness/contrast jitter.
        gaussian_blur_limit: Maximum kernel size for Gaussian blur (odd int).
        gaussian_blur_probability: Probability of applying Gaussian blur.
        sharpen_probability: Probability of applying unsharp-mask sharpening.
        horizontal_flip_probability: Probability of a left-right flip.
        perspective_scale: Maximum perspective warp magnitude (fraction of image size).
        perspective_probability: Probability of applying perspective warp.
        random_rain_probability: Probability of overlaying synthetic rain.
        random_fog_probability: Probability of overlaying synthetic fog.
    """

    enabled: bool = True

    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = field(default_factory=lambda: (8, 8))
    clahe_probability: float = 0.7

    # Shadow
    random_shadow_probability: float = 0.5

    # Brightness / contrast
    brightness_limit: float = 0.3
    contrast_limit: float = 0.3
    brightness_contrast_probability: float = 0.5

    # Blur / sharpen
    gaussian_blur_limit: int = 7
    gaussian_blur_probability: float = 0.3
    sharpen_probability: float = 0.2

    # Geometric
    horizontal_flip_probability: float = 0.5
    perspective_scale: float = 0.1
    perspective_probability: float = 0.4

    # Weather
    random_rain_probability: float = 0.2
    random_fog_probability: float = 0.2

    @classmethod
    def from_yaml_section(cls, aug_cfg: dict) -> "AugmentationConfig":
        """Construct an :class:`AugmentationConfig` from the ``augmentation`` dict.

        Args:
            aug_cfg: The value of ``config["augmentation"]`` after
                ``yaml.safe_load``.

        Returns:
            A populated :class:`AugmentationConfig` instance.
        """
        tgs = aug_cfg.get("clahe", {}).get("tile_grid_size", [8, 8])
        return cls(
            enabled=aug_cfg.get("enabled", True),
            clahe_clip_limit=aug_cfg.get("clahe", {}).get("clip_limit", 2.0),
            clahe_tile_grid_size=(int(tgs[0]), int(tgs[1])),
            clahe_probability=aug_cfg.get("clahe", {}).get("probability", 0.7),
            random_shadow_probability=aug_cfg.get("random_shadow", {}).get("probability", 0.5),
            brightness_limit=aug_cfg.get("random_brightness_contrast", {}).get("brightness_limit", 0.3),
            contrast_limit=aug_cfg.get("random_brightness_contrast", {}).get("contrast_limit", 0.3),
            brightness_contrast_probability=aug_cfg.get("random_brightness_contrast", {}).get("probability", 0.5),
            gaussian_blur_limit=aug_cfg.get("gaussian_blur", {}).get("blur_limit", 7),
            gaussian_blur_probability=aug_cfg.get("gaussian_blur", {}).get("probability", 0.3),
            horizontal_flip_probability=aug_cfg.get("horizontal_flip", {}).get("probability", 0.5),
            perspective_scale=aug_cfg.get("perspective_transform", {}).get("scale", 0.1),
            perspective_probability=aug_cfg.get("perspective_transform", {}).get("probability", 0.4),
            random_rain_probability=aug_cfg.get("random_rain", {}).get("probability", 0.2),
            random_fog_probability=aug_cfg.get("random_fog", {}).get("probability", 0.2),
        )


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_training_augmentation(
    config: AugmentationConfig,
    with_keypoints: bool = False,
) -> "A.Compose":
    """Build the full training augmentation pipeline from *config*.

    Transform order is deliberate: photometric transforms (CLAHE, shadow,
    brightness) run before geometric ones (flip, perspective) so that
    photometric artefacts are not distorted by subsequent warps.  Weather
    effects (rain, fog) are applied last to overlay on the already-warped
    image.  Normalize is always the final step.

    When ``config.enabled`` is ``False`` the pipeline contains only
    Normalize, making it safe to use in ablation studies without changing
    call sites.

    Args:
        config: Populated :class:`AugmentationConfig` instance.
        with_keypoints: When ``True``, attach
            ``albumentations.KeypointParams`` so the pipeline also transforms
            ``(x, y)`` keypoints (e.g. lane points) in sync with the image.
            Pass ``keypoints=[(x, y), ...]`` to the returned pipeline.

    Returns:
        An ``albumentations.Compose`` pipeline ready to call with
        ``pipeline(image=bgr_array)["image"]``.
    """
    kp_params = (
        A.KeypointParams(format="xy", remove_invisible=False)
        if with_keypoints
        else None
    )

    if not config.enabled:
        log.info("Augmentation disabled — returning Normalize-only pipeline.")
        return A.Compose(
            [A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)],
            keypoint_params=kp_params,
        )

    # GaussianBlur blur_limit must be an odd int >= 3
    blur_limit = max(3, config.gaussian_blur_limit)
    if blur_limit % 2 == 0:
        blur_limit += 1

    transforms = [
        # --- Photometric -------------------------------------------------------
        A.CLAHE(
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
            p=config.clahe_probability,
        ),
        A.RandomShadow(p=config.random_shadow_probability),
        A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit,
            contrast_limit=config.contrast_limit,
            p=config.brightness_contrast_probability,
        ),
        # --- Blur / sharpen ----------------------------------------------------
        A.GaussianBlur(blur_limit=blur_limit, p=config.gaussian_blur_probability),
        A.Sharpen(p=config.sharpen_probability),
        # --- Geometric ---------------------------------------------------------
        A.HorizontalFlip(p=config.horizontal_flip_probability),
        A.Perspective(scale=config.perspective_scale, p=config.perspective_probability),
        # --- Weather -----------------------------------------------------------
        A.RandomRain(p=config.random_rain_probability),
        A.RandomFog(p=config.random_fog_probability),
        # --- Normalise (always last) -------------------------------------------
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return A.Compose(transforms, keypoint_params=kp_params)


def build_inference_augmentation() -> "A.Compose":
    """Build the inference-time augmentation pipeline.

    Resizes to the lane-detector input resolution (800 × 288) and normalises.
    No random transforms are applied so results are fully deterministic.

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose([
        A.Resize(height=_INFERENCE_H, width=_INFERENCE_W),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_validation_augmentation() -> "A.Compose":
    """Build the validation-time augmentation pipeline.

    Applies only Normalize so that validation metrics reflect the true
    distribution of the data without any stochastic perturbation.

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose([
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# High-level augmentor class
# ---------------------------------------------------------------------------

_Mode = Literal["train", "val", "inference"]


class RoadSageAugmentor:
    """High-level interface for RoadSage image augmentation.

    Selects the correct pipeline for the requested *mode* and exposes simple
    ``augment`` / ``augment_with_lanes`` methods so callers need not interact
    with albumentations directly.

    Args:
        config_path: Path to a RoadSage YAML config file.  If the file cannot
            be found, default :class:`AugmentationConfig` values are used and
            a warning is emitted.
        mode: One of ``"train"``, ``"val"``, or ``"inference"``.

    Raises:
        ValueError: When *mode* is not one of the three accepted strings.

    Example::

        augmentor = RoadSageAugmentor(mode="train")
        aug = augmentor.augment(bgr_image)
        aug, pts = augmentor.augment_with_lanes(bgr_image, lane_points)
    """

    _VALID_MODES: frozenset[str] = frozenset({"train", "val", "inference"})

    def __init__(
        self,
        config_path: str = "configs/production.yaml",
        mode: _Mode = "train",
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {sorted(self._VALID_MODES)}."
            )
        self.mode = mode
        self._aug_config = self._load_aug_config(config_path)
        self._pipeline = self._build_pipeline()
        # Track whether the pipeline contains HorizontalFlip so
        # augment_with_lanes can mirror x-coordinates correctly.
        self._has_hflip = (
            mode == "train" and self._aug_config.horizontal_flip_probability > 0.0
        )
        log.info("RoadSageAugmentor ready (mode=%s).", mode)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_aug_config(config_path: str) -> AugmentationConfig:
        """Load augmentation settings from *config_path*.

        Falls back to default :class:`AugmentationConfig` values if the file
        is absent or the ``augmentation`` key is missing.
        """
        path = Path(config_path)
        if not path.exists():
            log.warning(
                "Config file '%s' not found — using default augmentation settings.",
                config_path,
            )
            return AugmentationConfig()

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        aug_section = raw.get("augmentation")
        if aug_section is None:
            log.warning(
                "No 'augmentation' key in '%s' — using defaults.", config_path
            )
            return AugmentationConfig()

        return AugmentationConfig.from_yaml_section(aug_section)

    def _build_pipeline(self) -> "A.Compose":
        if self.mode == "train":
            return build_training_augmentation(self._aug_config)
        if self.mode == "val":
            return build_validation_augmentation()
        return build_inference_augmentation()  # "inference"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply the pipeline to *image* and return the augmented array.

        Args:
            image: Input image in BGR format as returned by ``cv2.imread``.
                The array is converted to RGB before passing to albumentations
                and converted back to BGR on return so callers using OpenCV
                see a consistent colour channel order.

        Returns:
            Augmented image array in BGR format, dtype ``float32`` (normalised).
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._pipeline(image=rgb)["image"]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def augment_with_lanes(
        self,
        image: np.ndarray,
        lane_points: list[tuple[float, float]],
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Apply the pipeline to *image* and transform *lane_points* accordingly.

        Lane points are pixel coordinates ``(x, y)`` in the original image
        space.  This method manually mirrors x-coordinates when a
        :class:`~albumentations.HorizontalFlip` is actually triggered, keeping
        lane annotations consistent with the augmented image.

        The approach samples the flip decision independently by comparing a
        fresh ``random.random()`` draw against the configured flip probability,
        which matches how albumentations internally decides whether to flip.
        Both the image augmentation and the coordinate transform use the same
        draw, ensuring they are always in sync.

        Args:
            image: BGR input image array.
            lane_points: List of ``(x, y)`` pixel coordinate pairs describing
                lane positions on the original image.

        Returns:
            A ``(augmented_image, transformed_points)`` tuple.  Points are
            returned as a new list; the input list is not mutated.

        Note:
            Perspective warps are not currently propagated to lane points
            because the warp matrix is not exposed by albumentations without
            a custom transform.  For geometry-critical applications, replace
            :class:`~albumentations.Perspective` with a manual
            ``cv2.getPerspectiveTransform`` call.
        """
        img_w = image.shape[1]

        # Decide flip before running the full pipeline so we can mirror points.
        do_flip = (
            self._has_hflip
            and random.random() < self._aug_config.horizontal_flip_probability
        )

        if do_flip:
            flipped_image = cv2.flip(image, 1)
            aug_image = self.augment(flipped_image)
            transformed = [(img_w - 1 - x, y) for x, y in lane_points]
        else:
            aug_image = self.augment(image)
            transformed = list(lane_points)

        return aug_image, transformed


# ---------------------------------------------------------------------------
# Demo / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for the demo: pip install matplotlib")
        sys.exit(1)

    SOURCE_DIR = "rgb"
    OUTPUT_PATH = Path("outputs/augmentation_preview.png")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Collect images
    image_paths: list[Path] = [
        p for p in Path(SOURCE_DIR).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not image_paths:
        log.error("No images found in '%s'.", SOURCE_DIR)
        sys.exit(1)

    samples = random.sample(image_paths, min(5, len(image_paths)))
    augmentor = RoadSageAugmentor(mode="train")

    n_rows = len(samples)
    n_cols = 4  # original + 3 augmented versions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1:
        axes = [axes]

    for row_idx, img_path in enumerate(samples):
        original_bgr = cv2.imread(str(img_path))
        if original_bgr is None:
            log.warning("Could not read '%s' — skipping.", img_path.name)
            continue

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        axes[row_idx][0].imshow(original_rgb)
        axes[row_idx][0].set_title(f"Original\n{img_path.name}", fontsize=7)
        axes[row_idx][0].axis("off")

        for aug_idx in range(1, 4):
            aug_bgr = augmentor.augment(original_bgr)
            # Denormalise for display: reverse ImageNet norm → clip to [0, 1]
            mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
            std = np.array(_IMAGENET_STD, dtype=np.float32)
            aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            display = np.clip(aug_rgb * std + mean, 0.0, 1.0)

            axes[row_idx][aug_idx].imshow(display)
            axes[row_idx][aug_idx].set_title(f"Aug {aug_idx}", fontsize=7)
            axes[row_idx][aug_idx].axis("off")
            print(f"  [{img_path.name}] aug{aug_idx} applied.")

    fig.suptitle("RoadSage Augmentation Preview — MNNIT Campus", fontsize=11, y=1.01)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    print(f"\nSaved preview grid to '{OUTPUT_PATH}'.")
    plt.show()
