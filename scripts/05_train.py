"""
05_train.py — Train ResNet50 (or configured backbone) for pure/mixed classification.

Training scheme (staged fine-tuning):
  Phase 1: Freeze backbone, train head only, for N1 epochs (warm-up)
  Phase 2: Unfreeze backbone, fine-tune end-to-end at lower LR, for N2 epochs

Per-fold flow:
  - Build train/val DataLoaders for this fold
  - Run Phase 1 + Phase 2 with early stopping on val loss
  - Aggregate per-image predictions to per-stone predictions for "real" metrics
  - Save best checkpoint + per-epoch metrics CSV
  - Save fold-level summary

After all folds: write a cross-fold summary (mean ± std of val metrics).

Usage:
  python scripts/05_train.py              # runs all folds
  python scripts/05_train.py --fold 0     # single fold only (for dev/debug)
  python scripts/05_train.py --quick      # 2 epochs per phase, for smoke test

The image files must exist for this to run. Run 01_data_prep.py first to
verify everything is in place.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("train")


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    """Check the requested device is actually available and fall back gracefully."""
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("MPS requested but not available — falling back to CPU")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Training / eval loops
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": running_loss / total,
        "img_accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    aggregation: str = "mean_probability",
) -> dict:
    """Evaluate with BOTH image-level and stone-level metrics.

    For the stone-level metric, we aggregate the 3 photos of each stone into
    one prediction. This is the metric that actually matters clinically —
    the 3 photos represent different views of the same stone.
    """
    model.eval()
    running_loss = 0.0
    total = 0

    # Image-level buffers
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_stone_ids: list[str] = []

    for images, labels, stone_ids in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels_dev)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(mixed)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_stone_ids.extend(list(stone_ids))

    # --- Image-level metrics ---
    img_metrics = {
        "loss": running_loss / total,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    if len(set(all_labels)) > 1:
        img_metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)

    # --- Stone-level aggregation ---
    eval_df = pd.DataFrame({
        "stone_id": all_stone_ids,
        "label": all_labels,
        "pred": all_preds,
        "prob_mixed": all_probs,
    })
    if aggregation == "mean_probability":
        stone_df = eval_df.groupby("stone_id").agg(
            label=("label", "first"),          # all 3 photos share one label
            prob_mixed=("prob_mixed", "mean"),
        ).reset_index()
        stone_df["pred"] = (stone_df["prob_mixed"] >= 0.5).astype(int)
    elif aggregation == "majority_vote":
        stone_df = eval_df.groupby("stone_id").agg(
            label=("label", "first"),
            pred=("pred", lambda s: int(s.mode().iloc[0])),
            prob_mixed=("prob_mixed", "mean"),
        ).reset_index()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    stone_metrics = {
        "stone_accuracy": accuracy_score(stone_df["label"], stone_df["pred"]),
        "stone_precision": precision_score(stone_df["label"], stone_df["pred"], zero_division=0),
        "stone_recall": recall_score(stone_df["label"], stone_df["pred"], zero_division=0),
        "stone_f1": f1_score(stone_df["label"], stone_df["pred"], zero_division=0),
    }
    if stone_df["label"].nunique() > 1:
        stone_metrics["stone_roc_auc"] = roc_auc_score(stone_df["label"], stone_df["prob_mixed"])

    return {**img_metrics, **stone_metrics}


# -----------------------------------------------------------------------------
# Per-fold training
# -----------------------------------------------------------------------------
def train_one_fold(
    fold_i: int,
    samples: list,
    train_idx: list[int],
    val_idx: list[int],
    cfg: dict,
    device: torch.device,
    fold_dir: Path,
    quick: bool = False,
) -> dict:
    """Train a single fold end-to-end and return best-epoch metrics."""
    ensure_dir(fold_dir)

    # Lazy imports to avoid loading heavy modules at script import time
    ds = import_module("03_dataset")
    model_module = import_module("04_model")

    train_loader = ds.make_dataloader(samples, train_idx, cfg, train=True)
    val_loader = ds.make_dataloader(samples, val_idx, cfg, train=False)

    model = model_module.build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    tcfg = cfg["training"]
    epochs_frozen = 2 if quick else tcfg["epochs_frozen"]
    epochs_finetune = 2 if quick else tcfg["epochs_finetune"]
    patience = tcfg["early_stopping_patience"]

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    epochs_since_best = 0
    best_ckpt_path = fold_dir / "best.pt"

    def run_phase(phase_name: str, n_epochs: int, optimizer: torch.optim.Optimizer) -> bool:
        """Run a training phase. Returns True if early-stopped."""
        nonlocal best_val_loss, best_epoch, best_metrics, epochs_since_best

        for epoch in range(n_epochs):
            t0 = time.time()
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(
                model, val_loader, criterion, device,
                aggregation=cfg["evaluation"]["stone_level_aggregation"],
            )
            elapsed = time.time() - t0

            global_epoch = len(history)
            row = {
                "fold": fold_i,
                "phase": phase_name,
                "epoch": global_epoch,
                "phase_epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_img_acc": train_metrics["img_accuracy"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "seconds": round(elapsed, 1),
            }
            history.append(row)
            log.info(
                f"  [{phase_name}] epoch {epoch+1}/{n_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_stone_acc={val_metrics['stone_accuracy']:.3f} | "
                f"val_stone_f1={val_metrics['stone_f1']:.3f} | "
                f"{elapsed:.0f}s"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = global_epoch
                best_metrics = val_metrics.copy()
                epochs_since_best = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "fold": fold_i,
                    "epoch": global_epoch,
                    "val_metrics": val_metrics,
                }, best_ckpt_path)
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    log.info(f"  Early stopping (no improvement for {patience} epochs)")
                    return True
        return False

    # Phase 1: frozen backbone, train head only
    log.info(f"Fold {fold_i} — Phase 1: frozen backbone, {epochs_frozen} epochs")
    model.freeze_backbone()
    log.info(f"  Trainable params: {model.trainable_param_count():,}")
    optimizer = torch.optim.Adam(
        model.head_parameters(),
        lr=tcfg["learning_rate_head"],
        weight_decay=tcfg["weight_decay"],
    )
    stopped = run_phase("frozen", epochs_frozen, optimizer)

    # Phase 2: fine-tune whole model
    if not stopped:
        log.info(f"Fold {fold_i} — Phase 2: fine-tune, {epochs_finetune} epochs")
        model.unfreeze_backbone()
        log.info(f"  Trainable params: {model.trainable_param_count():,}")
        # Two param groups so backbone gets a smaller LR than head
        optimizer = torch.optim.Adam([
            {"params": model.backbone_parameters(), "lr": tcfg["learning_rate_backbone"]},
            {"params": model.head_parameters(),     "lr": tcfg["learning_rate_head"]},
        ], weight_decay=tcfg["weight_decay"])
        # Reset early-stopping counter for phase 2
        epochs_since_best = 0
        run_phase("finetune", epochs_finetune, optimizer)

    # Save per-fold outputs
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    fold_summary = {
        "fold": fold_i,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **{f"best_{k}": v for k, v in best_metrics.items()},
    }
    with open(fold_dir / "summary.json", "w") as f:
        json.dump(fold_summary, f, indent=2)
    log.info(f"Fold {fold_i} done. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")
    return fold_summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run only this fold")
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs per phase, for smoke testing")
    args = parser.parse_args()

    cfg = load_config()

    # Set seeds
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(cfg["training"]["device"])
    log.info(f"Device: {device}")

    # Build samples
    ds = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds.load_image_samples(images_csv, require_files_exist=True)

    # Hold out test set before CV
    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds.make_test_holdout(all_samples, test_fraction, seed)
    train_val_samples = [all_samples[i] for i in train_val_idx]

    folds = ds.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
    )

    ckpt_dir = ensure_dir(resolve_path(cfg, "checkpoints_dir"))
    logs_dir = ensure_dir(resolve_path(cfg, "logs_dir"))

    # Run folds (indices are into train_val_samples)
    all_summaries: list[dict] = []
    fold_range = [args.fold] if args.fold is not None else range(len(folds))
    for fold_i in fold_range:
        train_idx, val_idx = folds[fold_i]
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        summary = train_one_fold(
            fold_i, train_val_samples, train_idx, val_idx,
            cfg, device, fold_dir, quick=args.quick,
        )
        all_summaries.append(summary)

    # Cross-fold aggregate (only meaningful when running all folds)
    if args.fold is None and len(all_summaries) > 1:
        df = pd.DataFrame(all_summaries)
        metric_cols = [c for c in df.columns if c.startswith("best_") and c != "best_epoch"]
        agg = pd.concat([
            df[metric_cols].mean().rename("mean"),
            df[metric_cols].std().rename("std"),
        ], axis=1)
        agg.to_csv(logs_dir / "cv_summary.csv")
        df.to_csv(logs_dir / "fold_summaries.csv", index=False)
        log.info("Cross-fold summary:")
        for metric in ["best_val_stone_accuracy", "best_val_stone_f1"]:
            if metric in agg.index:
                log.info(f"  {metric}: {agg.at[metric, 'mean']:.3f} ± {agg.at[metric, 'std']:.3f}")

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # Evaluate each trained fold's best checkpoint on the test set and report
    # the mean ± std — this is the unbiased final performance estimate.
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    model_module = import_module("04_model")
    criterion = nn.CrossEntropyLoss()
    aggregation = cfg["evaluation"]["stone_level_aggregation"]
    test_loader = ds.make_dataloader(all_samples, test_idx, cfg, train=False)

    test_fold_metrics: list[dict] = []
    for fold_i in fold_range:
        ckpt_path = ckpt_dir / f"fold_{fold_i}" / "best.pt"
        if not ckpt_path.exists():
            log.warning(f"  Fold {fold_i}: checkpoint not found at {ckpt_path}, skipping")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = model_module.build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        test_m = evaluate(model, test_loader, criterion, device, aggregation)
        test_fold_metrics.append({"fold": fold_i, **test_m})
        log.info(
            f"  Fold {fold_i} test: "
            f"stone_acc={test_m['stone_accuracy']:.3f} | "
            f"stone_f1={test_m['stone_f1']:.3f} | "
            f"stone_roc_auc={test_m.get('stone_roc_auc', float('nan')):.3f}"
        )

    if test_fold_metrics:
        test_df = pd.DataFrame(test_fold_metrics)
        test_df.to_csv(logs_dir / "test_fold_metrics.csv", index=False)
        if len(test_fold_metrics) > 1:
            numeric_cols = [c for c in test_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_df[numeric_cols].mean().rename("mean"),
                test_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / "test_summary.csv")
            log.info("Test set summary (mean ± std across folds):")
            for metric in ["stone_accuracy", "stone_f1", "stone_roc_auc"]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.3f} ± "
                        f"{test_summary.at[metric, 'std']:.3f}"
                    )


if __name__ == "__main__":
    main()
