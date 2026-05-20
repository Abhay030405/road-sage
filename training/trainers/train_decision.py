"""
training/trainers/train_decision.py
=====================================

Trains a MobileNetV3-Small decision CNN on pseudo-labeled MNNIT images.

Handles class imbalance with ``WeightedRandomSampler`` and
``CrossEntropyLoss`` class weights, supports early stopping, and exports
the final model to ONNX for deployment in :mod:`app.decision.ml_fallback`.

Typical usage::

    # First generate labels:
    python -m training.scripts.generate_pseudo_labels

    # Then train:
    python -m training.trainers.train_decision

    # Dry-run (dataset stats + model summary, no training):
    python -m training.trainers.train_decision --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------

CLASS_MAP: Dict[str, int] = {"FORWARD": 0, "LEFT": 1, "RIGHT": 2, "STOP": 3}
CLASS_NAMES: List[str] = ["FORWARD", "LEFT", "RIGHT", "STOP"]
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PseudoLabelDataset:
    """PyTorch Dataset backed by a JSONL pseudo-label file.

    Reads all labels from *jsonl_path*, performs a stratified train/val/test
    split, and serves ``(image_tensor, class_id)`` tuples.

    Args:
        jsonl_path: Path to the JSONL file produced by
            :func:`~training.scripts.generate_pseudo_labels.generate_pseudo_labels`.
        image_base_dir: Root directory relative to which ``image_path``
            fields in the JSONL are resolved.
        transform: Optional ``torchvision.transforms`` pipeline applied to
            each image.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        seed: Random seed for reproducible splits.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_base_dir: str,
        transform=None,
        split: str = "train",
        seed: int = 42,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        import torch  # noqa: F401 — deferred so non-torch envs can import module

        self._transform = transform
        self._image_base_dir = image_base_dir

        # Read all records
        records: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            raise ValueError(f"No records found in {jsonl_path!r}")

        # Stratified split by command
        try:
            from sklearn.model_selection import train_test_split

            labels = [CLASS_MAP[r["command"]] for r in records]
            test_frac = 1.0 - train_frac - val_frac

            train_val_recs, test_recs, train_val_lbl, _ = train_test_split(
                records, labels, test_size=test_frac, random_state=seed, stratify=labels
            )
            relative_val = val_frac / (train_frac + val_frac)
            train_val_lbl2 = [CLASS_MAP[r["command"]] for r in train_val_recs]
            train_recs, val_recs = train_test_split(
                train_val_recs, test_size=relative_val, random_state=seed,
                stratify=train_val_lbl2
            )
        except ImportError:
            logger.warning("scikit-learn not available — using sequential split.")
            n = len(records)
            rng = random.Random(seed)
            indices = list(range(n))
            rng.shuffle(indices)
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            train_recs = [records[i] for i in indices[:n_train]]
            val_recs = [records[i] for i in indices[n_train: n_train + n_val]]
            test_recs = [records[i] for i in indices[n_train + n_val:]]

        split_map = {"train": train_recs, "val": val_recs, "test": test_recs}
        if split not in split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        self._records = split_map[split]
        self._split = split

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        """Return ``(image_tensor, class_id)`` for record at *idx*.

        Args:
            idx: Index into the split's record list.

        Returns:
            Tuple of ``(torch.Tensor, int)``.

        Raises:
            RuntimeError: If the image cannot be loaded.
        """
        import torch
        from PIL import Image as PILImage

        rec = self._records[idx]
        img_path = os.path.join(self._image_base_dir, rec["image_path"])
        if not os.path.exists(img_path):
            img_path = rec["image_path"]  # try as absolute / project-root-relative

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Could not load image: {img_path!r}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)

        if self._transform is not None:
            tensor = self._transform(pil_img)
        else:
            tensor = torch.from_numpy(
                np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )

        class_id = CLASS_MAP[rec["command"]]
        return tensor, class_id

    def get_class_weights(self):
        """Compute inverse-frequency class weights for this split.

        Returns:
            ``torch.FloatTensor`` of shape ``(NUM_CLASSES,)``.  Classes with
            zero samples receive weight 0.

        Example::

            weights = train_dataset.get_class_weights()
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
        """
        import torch

        counts = [0] * NUM_CLASSES
        for rec in self._records:
            counts[CLASS_MAP[rec["command"]]] += 1

        total = sum(counts)
        weights = [
            (total / (NUM_CLASSES * c)) if c > 0 else 0.0
            for c in counts
        ]
        logger.info(
            "Class distribution (%s): %s",
            self._split,
            {CLASS_NAMES[i]: counts[i] for i in range(NUM_CLASSES)},
        )
        return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model():
    """Build a pretrained MobileNetV3-Small with a custom 4-class head.

    Architecture of the replacement classifier::

        Linear(576 → 128) → Hardswish → Dropout(0.3) → Linear(128 → 4)

    Returns:
        A ``torchvision.models.MobileNetV3`` module ready for fine-tuning.
    """
    import torch.nn as nn
    import torchvision.models as models

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    # Replace the original three-layer classifier
    model.classifier = nn.Sequential(
        nn.Linear(576, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(128, NUM_CLASSES),
    )
    return model


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def build_transforms():
    """Return ``(train_transform, val_transform)`` torchvision pipelines.

    Training pipeline includes random crop, horizontal flip, and colour
    jitter for regularisation.  Validation pipeline applies only
    deterministic resizing and centre crop.

    Returns:
        Tuple of ``(train_transform, val_transform)``.
    """
    from torchvision import transforms

    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """Run one full training pass over *loader*.

    Args:
        model: PyTorch module in training mode.
        loader: ``DataLoader`` for the training split.
        optimizer: Gradient-based optimiser.
        criterion: Loss function.
        device: ``torch.device`` to run on.

    Returns:
        Mean per-sample loss for this epoch.
    """
    import torch

    model.train()
    total_loss = 0.0
    n_batches = 0

    try:
        from tqdm import tqdm
        iterable = tqdm(loader, desc="  train", leave=False)
    except ImportError:
        iterable = loader

    for images, labels in iterable:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    """Evaluate *model* on *loader* without gradient computation.

    Args:
        model: PyTorch module.
        loader: ``DataLoader`` for the evaluation split.
        criterion: Loss function.
        device: ``torch.device``.

    Returns:
        ``(mean_loss, accuracy_percent)`` tuple.
    """
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    mean_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_decision_cnn(
    config_path: str = "configs/decision_engine.yaml",
    jsonl_path: str = "data/mnnit/pseudo_labels/labels.jsonl",
    output_dir: str = "models/",
) -> None:
    """Full training pipeline: data → model → train → evaluate → export.

    Steps:

    1. Load hyperparameters from *config_path* (``training`` section).
    2. Build stratified train / val / test datasets from *jsonl_path*.
    3. Create ``WeightedRandomSampler`` to mitigate class imbalance.
    4. Build MobileNetV3-Small model and move it to CPU.
    5. Train with ``CrossEntropyLoss`` + label smoothing for ``epochs``
       epochs, applying ``CosineAnnealingLR`` scheduling.
    6. Early-stopping based on validation loss with ``patience`` epochs.
    7. Reload the best checkpoint and evaluate on the test set.
    8. Export to ONNX (``opset_version=17``) and verify with OnnxRuntime.

    Args:
        config_path: Path to ``configs/decision_engine.yaml``.
        jsonl_path: Path to the JSONL pseudo-label file.
        output_dir: Directory to write model checkpoints and the ONNX file.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    with open(config_path, "r", encoding="utf-8") as fh:
        full_cfg = yaml.safe_load(fh)
    cfg = full_cfg["training"]

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    epochs = int(cfg.get("epochs", 50))
    patience = int(cfg.get("early_stopping_patience", 10))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    train_frac = float(cfg.get("train_split", 0.80))
    val_frac = float(cfg.get("val_split", 0.10))
    num_workers = int(cfg.get("num_workers", 0))  # 0 avoids Windows multiprocessing issues

    device = torch.device("cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = output_path / "decision_cnn_best.pth"
    onnx_path = output_path / "decision_cnn.onnx"

    train_transform, val_transform = build_transforms()

    logger.info("Loading datasets from %s", jsonl_path)
    train_dataset = PseudoLabelDataset(
        jsonl_path, image_base_dir=".",
        transform=train_transform, split="train",
        seed=seed, train_frac=train_frac, val_frac=val_frac,
    )
    val_dataset = PseudoLabelDataset(
        jsonl_path, image_base_dir=".",
        transform=val_transform, split="val",
        seed=seed, train_frac=train_frac, val_frac=val_frac,
    )
    test_dataset = PseudoLabelDataset(
        jsonl_path, image_base_dir=".",
        transform=val_transform, split="test",
        seed=seed, train_frac=train_frac, val_frac=val_frac,
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    # Weighted sampler (per-sample weights derived from class weights)
    class_weights = train_dataset.get_class_weights()
    sample_weights = torch.FloatTensor([
        class_weights[CLASS_MAP[train_dataset._records[i]["command"]]]
        for i in range(len(train_dataset))
    ])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # Model
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: MobileNetV3-Small  trainable params=%d", total_params)

    # Loss with class weighting + label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_val_acc = 0.0

    logger.info("Starting training for up to %d epochs (patience=%d)", epochs, patience)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.1f%%",
            epoch, epochs, train_loss, val_loss, val_acc,
        )
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info("  → Checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test accuracy: %.1f%%", test_acc)

    # Export to ONNX
    model.eval()
    dummy_input = torch.zeros(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info("ONNX exported to %s", onnx_path)

    # Verify with OnnxRuntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 3, 224, 224), dtype=np.float32)
        out = sess.run(None, {"input": dummy_np})
        assert out[0].shape == (1, NUM_CLASSES), f"Unexpected output shape: {out[0].shape}"
        logger.info("ONNX verification passed: output shape %s", out[0].shape)
    except Exception as exc:
        logger.warning("ONNX verification failed: %s", exc)

    print()
    print(f"Training complete. Best val accuracy: {best_val_acc:.1f}%. Test accuracy: {test_acc:.1f}%")
    print(f"ONNX model saved to {onnx_path}")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train the MobileNetV3-Small decision CNN on pseudo-labeled MNNIT data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/decision_engine.yaml",
                        help="Decision engine config path")
    parser.add_argument("--jsonl", default="data/mnnit/pseudo_labels/labels.jsonl",
                        help="Pseudo-label JSONL file")
    parser.add_argument("--output-dir", default="models/",
                        help="Directory to save checkpoints and ONNX model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dataset stats and model summary without training")
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if not Path(jsonl_path).exists() or Path(jsonl_path).stat().st_size == 0:
        print(f"No pseudo-labels found at {jsonl_path!r}")
        print("Run:  python -m training.scripts.generate_pseudo_labels  first.")
        raise SystemExit(0)

    if args.dry_run:
        import torch
        train_transform, val_transform = build_transforms()
        ds = PseudoLabelDataset(
            jsonl_path, image_base_dir=".",
            transform=train_transform, split="train",
        )
        weights = ds.get_class_weights()
        print(f"\nDataset size (train split): {len(ds)} samples")
        print(f"Class weights: {dict(zip(CLASS_NAMES, weights.tolist()))}")
        model = build_model()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"MobileNetV3-Small trainable parameters: {total_params:,}")
        print("\nDry run complete. Remove --dry-run to start training.")
    else:
        train_decision_cnn(
            config_path=args.config,
            jsonl_path=jsonl_path,
            output_dir=args.output_dir,
        )
