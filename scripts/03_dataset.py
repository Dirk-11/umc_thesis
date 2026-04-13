"""
03_dataset.py — PyTorch Dataset, transforms, and stone-level stratified splits.

Key design points:
- Splits are at STONE level (not image), so all 3 photos of a stone end up
  in the same fold. This prevents data leakage.
- Stratification uses the binary pure/mixed label by default (configurable).
- Augmentations are read from config.yaml so they can be tuned without
  touching code.
- The Dataset returns (image_tensor, label_int, stone_id) so we can aggregate
  per-stone at evaluation time.

Usage:
  from importlib import import_module
  ds_module = import_module('03_dataset')
  splits = ds_module.make_stratified_folds(...)
  train_loader = ds_module.make_dataloader(...)

Run as a script to verify splits look reasonable:
  python scripts/03_dataset.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, resolve_path, setup_logging

log = setup_logging("dataset")

LABEL_TO_INT = {"pure": 0, "mixed": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    """Build the torchvision transform pipeline from config.

    Training transforms include augmentation; eval transforms are deterministic
    (just resize + normalize).
    """
    img_cfg = cfg["image"]
    aug = cfg["augmentation"]

    base = [
        transforms.Resize((img_cfg["input_size"], img_cfg["input_size"])),
    ]

    if train:
        base += [
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
            transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]),
            transforms.RandomRotation(degrees=aug["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=aug["color_jitter_brightness"],
                contrast=aug["color_jitter_contrast"],
                saturation=aug["color_jitter_saturation"],
            ),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
    ]
    return transforms.Compose(base)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
@dataclass
class StoneSample:
    """One image from one stone."""
    image_path: str
    label: int            # 0=pure, 1=mixed
    stone_id: str         # for stone-level aggregation at eval time


class KidneyStoneDataset(Dataset):
    """Image-level dataset.

    Each item returns (image_tensor, label, stone_id). The stone_id lets us
    group the 3 photos of a stone back together at evaluation time without
    having to track indices separately.
    """

    def __init__(self, samples: list[StoneSample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        img = self.transform(img)
        return img, sample.label, sample.stone_id


# -----------------------------------------------------------------------------
# Sample construction from CSVs
# -----------------------------------------------------------------------------
def load_image_samples(images_csv: Path, require_files_exist: bool = True) -> list[StoneSample]:
    """Read the image-level CSV from 01_data_prep and convert to StoneSample list.

    Filters out:
      - Rows with no label (e.g. stones excluded for missing data)
      - Rows where the image file doesn't exist on disk (if require_files_exist)
    """
    df = pd.read_csv(images_csv)

    n_total = len(df)
    df = df[df["label"].notna()].copy()
    n_unlabeled = n_total - len(df)
    if n_unlabeled > 0:
        log.info(f"Skipped {n_unlabeled} rows with no label")

    if require_files_exist:
        n_before = len(df)
        df = df[df["image_exists"]].copy()
        n_missing = n_before - len(df)
        if n_missing > 0:
            log.warning(f"Skipped {n_missing} rows whose image files are missing")

    samples = [
        StoneSample(
            image_path=row["image_path"],
            label=LABEL_TO_INT[row["label"]],
            stone_id=row["fragment"],
        )
        for _, row in df.iterrows()
    ]
    log.info(f"Built {len(samples)} samples from {df['fragment'].nunique()} stones")
    return samples


# -----------------------------------------------------------------------------
# Stone-level stratified k-fold
# -----------------------------------------------------------------------------
def make_stratified_folds(
    samples: list[StoneSample],
    n_folds: int,
    seed: int,
    shuffle: bool = True,
) -> list[tuple[list[int], list[int]]]:
    """Return list of (train_idx, val_idx) pairs into the samples list.

    Splits are computed at the STONE level — all 3 photos of a stone end up
    in the same fold. Stratification is on the stone-level binary label.
    """
    # Build stone-level table: one row per stone
    stone_df = pd.DataFrame([
        {"stone_id": s.stone_id, "label": s.label} for s in samples
    ]).drop_duplicates(subset="stone_id").reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None)

    # Pre-build a lookup: stone_id → list of sample indices
    stone_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        stone_to_indices.setdefault(s.stone_id, []).append(i)

    folds: list[tuple[list[int], list[int]]] = []
    for fold_i, (train_stone_idx, val_stone_idx) in enumerate(
        skf.split(stone_df["stone_id"], stone_df["label"])
    ):
        train_stones = stone_df.iloc[train_stone_idx]["stone_id"].tolist()
        val_stones = stone_df.iloc[val_stone_idx]["stone_id"].tolist()

        train_sample_idx = [i for s in train_stones for i in stone_to_indices[s]]
        val_sample_idx = [i for s in val_stones for i in stone_to_indices[s]]

        # Sanity check: no stone should appear in both
        assert not (set(train_stones) & set(val_stones)), \
            f"Stone leakage in fold {fold_i}!"

        folds.append((train_sample_idx, val_sample_idx))

        # Log fold composition
        train_pure = sum(1 for s in train_stones
                         if stone_df.set_index("stone_id").at[s, "label"] == 0)
        val_pure = sum(1 for s in val_stones
                       if stone_df.set_index("stone_id").at[s, "label"] == 0)
        log.info(
            f"  Fold {fold_i}: "
            f"train={len(train_stones)} stones ({train_pure} pure / {len(train_stones)-train_pure} mixed, "
            f"{len(train_sample_idx)} images), "
            f"val={len(val_stones)} stones ({val_pure} pure / {len(val_stones)-val_pure} mixed, "
            f"{len(val_sample_idx)} images)"
        )

    return folds


# -----------------------------------------------------------------------------
# Held-out test split
# -----------------------------------------------------------------------------
def make_test_holdout(
    samples: list,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split off a held-out test set at stone level.

    Stratified on the binary pure/mixed label to preserve class balance.
    Must be called with the same seed as make_stratified_folds so the splits
    are reproducible across scripts.

    Returns:
        train_val_indices: indices into samples for CV training
        test_indices:      indices into samples for the held-out test set
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    stone_df = pd.DataFrame([
        {"stone_id": s.stone_id, "label": s.label} for s in samples
    ]).drop_duplicates(subset="stone_id").reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_stone_idx, test_stone_idx = next(
        sss.split(stone_df["stone_id"], stone_df["label"])
    )

    stone_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        stone_to_indices.setdefault(s.stone_id, []).append(i)

    train_val_stones = stone_df.iloc[train_val_stone_idx]["stone_id"].tolist()
    test_stones      = stone_df.iloc[test_stone_idx]["stone_id"].tolist()

    train_val_idx = [i for s in train_val_stones for i in stone_to_indices[s]]
    test_idx      = [i for s in test_stones      for i in stone_to_indices[s]]

    # Sanity check
    assert not (set(train_val_stones) & set(test_stones)), "Test leakage!"

    test_pure = sum(
        1 for s in test_stones
        if stone_df.set_index("stone_id").at[s, "label"] == 0
    )
    log.info(
        f"Test holdout: {len(test_stones)} stones "
        f"({test_pure} pure / {len(test_stones) - test_pure} mixed, "
        f"{len(test_idx)} images). "
        f"Train+val pool: {len(train_val_stones)} stones ({len(train_val_idx)} images)."
    )
    return train_val_idx, test_idx


# -----------------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------------
def make_dataloader(
    samples: list[StoneSample],
    indices: list[int],
    cfg: dict,
    train: bool,
) -> DataLoader:
    """Build a DataLoader from a subset of samples."""
    subset = [samples[i] for i in indices]
    transform = build_transforms(cfg, train=train)
    dataset = KidneyStoneDataset(subset, transform)
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=train,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["device"] != "cpu",
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Model B — Compositional dataset
# -----------------------------------------------------------------------------
@dataclass
class CompositionalSample:
    """One image from one stone, with full composition vector for regression."""
    image_path: str
    composition: list[float]   # length = n_classes, sums to 1.0
    stone_id: str
    label: int                 # binary pure/mixed — kept for fold stratification


class CompositionalKidneyStoneDataset(Dataset):
    """Image-level dataset for compositional regression (Model B).

    Each item returns (image_tensor, composition_tensor, stone_id, label) where:
      image_tensor:       (C, H, W) float32
      composition_tensor: (n_classes,) float32, sums to 1.0
      stone_id:           str, for stone-level aggregation at eval time
      label:              int, binary pure/mixed (0/1)
    """

    def __init__(self, samples: list[CompositionalSample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        img = self.transform(img)
        comp = torch.tensor(sample.composition, dtype=torch.float32)
        return img, comp, sample.stone_id, sample.label


def load_compositional_samples(
    images_csv: Path,
    final_classes: list[str],
    require_files_exist: bool = True,
) -> list[CompositionalSample]:
    """Read the image-level CSV and build CompositionalSample list.

    Expects comp_<class> columns to be present — re-run 01_data_prep.py if missing.
    Filters out rows with no composition data and, optionally, missing image files.
    """
    df = pd.read_csv(images_csv)

    comp_cols = [f"comp_{cls}" for cls in final_classes]
    missing_cols = [c for c in comp_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Composition columns missing from {images_csv}: {missing_cols}. "
            f"Re-run scripts/thesis_starter/scripts/01_data_prep.py to generate them."
        )

    n_total = len(df)
    df = df[df[comp_cols].sum(axis=1) > 0].copy()
    n_dropped = n_total - len(df)
    if n_dropped > 0:
        log.info(f"Skipped {n_dropped} rows with no composition data")

    df = df[df["label"].notna()].copy()

    if require_files_exist:
        n_before = len(df)
        df = df[df["image_exists"]].copy()
        n_missing = n_before - len(df)
        if n_missing > 0:
            log.warning(f"Skipped {n_missing} rows whose image files are missing")

    samples = [
        CompositionalSample(
            image_path=row["image_path"],
            composition=row[comp_cols].tolist(),
            stone_id=row["fragment"],
            label=LABEL_TO_INT[row["label"]],
        )
        for _, row in df.iterrows()
    ]
    log.info(
        f"Built {len(samples)} compositional samples "
        f"from {df['fragment'].nunique()} stones"
    )
    return samples


def make_compositional_dataloader(
    samples: list[CompositionalSample],
    indices: list[int],
    cfg: dict,
    train: bool,
) -> DataLoader:
    """Build a DataLoader for compositional regression from a subset of samples."""
    subset = [samples[i] for i in indices]
    transform = build_transforms(cfg, train=train)
    dataset = CompositionalKidneyStoneDataset(subset, transform)
    tcfg = cfg["training_moe"]
    return DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=train,
        num_workers=tcfg["num_workers"],
        pin_memory=tcfg["device"] != "cpu",
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def main() -> None:
    """Sanity check: build splits and report composition."""
    cfg = load_config()
    images_csv = resolve_path(cfg, "processed_images_csv")

    if not images_csv.exists():
        log.error(f"Run 01_data_prep.py first — {images_csv} not found")
        sys.exit(1)

    # Don't require image files for the smoke test (we may not have all of them)
    samples = load_image_samples(images_csv, require_files_exist=False)

    log.info(f"Building {cfg['cv']['n_folds']}-fold stratified splits at stone level:")
    folds = make_stratified_folds(
        samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=cfg["project"]["seed"],
        shuffle=cfg["cv"]["shuffle"],
    )

    log.info(f"Built {len(folds)} folds. No stone appears in both train and val "
             f"of any fold (verified).")

    # Smoke test the transforms
    log.info("Testing transforms...")
    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)
    log.info(f"  Train transform: {len(train_tf.transforms)} steps")
    log.info(f"  Eval transform:  {len(eval_tf.transforms)} steps")
    log.info("Done.")


if __name__ == "__main__":
    main()
