"""
05_train_multiclass.py — Train a 6-class classifier for primary component prediction (Model C).

The target label for each stone is the argmax of its composition vector — i.e.
whichever of the 6 final classes (CaOx, CHP, UA, MAP, CYS, OTH) makes up the
largest fraction. Loss is standard cross-entropy.

Training scheme (same staged fine-tuning as Models A and B):
  Phase 1: Freeze backbone, train head only for N epochs (warm-up)
  Phase 2: Unfreeze backbone, fine-tune end-to-end at lower LR for N epochs

Metrics:
  - Image-level and stone-level accuracy
  - Macro F1 (unweighted average across 6 classes)
  - Per-class F1

Usage:
  python scripts/05_train_multiclass.py              # all folds
  python scripts/05_train_multiclass.py --fold 0     # single fold
  python scripts/05_train_multiclass.py --quick      # 2 epochs/phase (smoke test)
"""
from __future__ import annotations

import argparse
import dataclasses
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
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("train_multiclass")


# -----------------------------------------------------------------------------
# Random-label baseline helper
# -----------------------------------------------------------------------------
def _shuffle_labels_stone_level(samples: list, rng: np.random.RandomState) -> list:
    """Return new samples with labels permuted at the stone level.

    All images of a stone receive the same (shuffled) label so there is no
    within-stone inconsistency.  Used for the random-label sanity-check
    baseline: a model trained this way should score near chance on the test set.
    """
    stone_ids_ordered: list[str] = []
    stone_to_label: dict[str, int] = {}
    for s in samples:
        if s.stone_id not in stone_to_label:
            stone_ids_ordered.append(s.stone_id)
            stone_to_label[s.stone_id] = s.label

    labels = [stone_to_label[sid] for sid in stone_ids_ordered]
    shuffled = rng.permutation(labels).tolist()
    shuffled_map = dict(zip(stone_ids_ordered, shuffled))
    return [dataclasses.replace(s, label=shuffled_map[s.stone_id]) for s in samples]


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    log.warning(f"{requested.upper()} not available — falling back to CPU")
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

    return {"loss": running_loss / total, "img_accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """Evaluate image-level and stone-level metrics."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    all_stone_ids: list[str] = []

    for images, labels, stone_ids in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels_dev)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        probs = torch.softmax(logits, dim=1).cpu().tolist()
        preds = logits.argmax(dim=1).cpu().tolist()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_stone_ids.extend(list(stone_ids))

    img_metrics = {
        "loss": running_loss / total,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    # Stone-level: average class probabilities across the 3 images, then argmax
    df = pd.DataFrame({
        "stone_id": all_stone_ids,
        "label": all_labels,
        **{f"prob_{c}": [p[i] for p in all_probs] for i, c in enumerate(class_names)},
    })
    prob_cols = [f"prob_{c}" for c in class_names]
    stone_df = df.groupby("stone_id").agg(
        label=("label", "first"),
        **{col: (col, "mean") for col in prob_cols},
    ).reset_index()
    stone_df["pred"] = stone_df[prob_cols].values.argmax(axis=1)

    stone_metrics = {
        "stone_accuracy": accuracy_score(stone_df["label"], stone_df["pred"]),
        "stone_macro_f1": f1_score(
            stone_df["label"], stone_df["pred"], average="macro", zero_division=0
        ),
    }
    # Per-class F1 at stone level
    per_class_f1 = f1_score(
        stone_df["label"], stone_df["pred"],
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    for cls, f1_val in zip(class_names, per_class_f1):
        stone_metrics[f"stone_f1_{cls}"] = float(f1_val)

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
    class_names: list[str],
    quick: bool = False,
) -> dict:
    ensure_dir(fold_dir)

    ds_module    = import_module("03_dataset")
    model_module = import_module("04_model_binary")  # StoneClassifier is generic

    train_loader = ds_module.make_multiclass_dataloader(samples, train_idx, cfg, train=True)
    val_loader   = ds_module.make_multiclass_dataloader(samples, val_idx,   cfg, train=False)

    # Build model from model_multiclass config section
    mc = cfg["model_multiclass"]
    model = model_module.StoneClassifier(
        backbone_name=mc["backbone"],
        weights=mc["pretrained_weights"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
    ).to(device)

    # Class-weighted loss to handle imbalance (CaOx=128, OTH=4)
    from collections import Counter
    train_labels = [samples[i].label for i in train_idx]
    counts = Counter(train_labels)
    n_train = len(train_labels)
    n_classes = len(class_names)
    class_weights = torch.tensor(
        [n_train / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    tcfg = cfg["training_multiclass"]
    epochs_frozen   = 2 if quick else tcfg["epochs_frozen"]
    epochs_finetune = 2 if quick else tcfg["epochs_finetune"]
    patience = tcfg["early_stopping_patience"]

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    epochs_since_best = 0
    best_ckpt_path = fold_dir / "best.pt"

    def run_phase(phase_name: str, n_epochs: int, optimizer: torch.optim.Optimizer) -> bool:
        nonlocal best_val_loss, best_epoch, best_metrics, epochs_since_best

        for epoch in range(n_epochs):
            t0 = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_m   = evaluate(model, val_loader, criterion, device, class_names)
            elapsed = time.time() - t0

            global_epoch = len(history)
            row = {
                "fold": fold_i, "phase": phase_name,
                "epoch": global_epoch, "phase_epoch": epoch,
                "train_loss": train_m["loss"],
                "train_img_acc": train_m["img_accuracy"],
                **{f"val_{k}": v for k, v in val_m.items()},
                "seconds": round(elapsed, 1),
            }
            history.append(row)
            log.info(
                f"  [{phase_name}] epoch {epoch+1}/{n_epochs} | "
                f"train_loss={train_m['loss']:.4f} | "
                f"val_loss={val_m['loss']:.4f} | "
                f"val_stone_acc={val_m['stone_accuracy']:.3f} | "
                f"val_stone_macro_f1={val_m['stone_macro_f1']:.3f} | "
                f"{elapsed:.0f}s"
            )

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_epoch = global_epoch
                best_metrics = val_m.copy()
                epochs_since_best = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "fold": fold_i,
                    "epoch": global_epoch,
                    "val_metrics": val_m,
                    "class_names": class_names,
                }, best_ckpt_path)
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    log.info(f"  Early stopping ({patience} epochs without improvement)")
                    return True
        return False

    # Phase 1: frozen backbone
    log.info(f"Fold {fold_i} — Phase 1: frozen backbone, {epochs_frozen} epochs")
    model.freeze_backbone()
    log.info(f"  Trainable params: {model.trainable_param_count():,}")
    optimizer = torch.optim.Adam(
        model.head_parameters(),
        lr=tcfg["learning_rate_head"],
        weight_decay=tcfg["weight_decay"],
    )
    stopped = run_phase("frozen", epochs_frozen, optimizer)

    # Phase 2: fine-tune full model
    if not stopped:
        log.info(f"Fold {fold_i} — Phase 2: fine-tune, {epochs_finetune} epochs")
        model.unfreeze_backbone()
        log.info(f"  Trainable params: {model.trainable_param_count():,}")
        optimizer = torch.optim.Adam([
            {"params": model.backbone_parameters(), "lr": tcfg["learning_rate_backbone"]},
            {"params": model.head_parameters(),     "lr": tcfg["learning_rate_head"]},
        ], weight_decay=tcfg["weight_decay"])
        epochs_since_best = 0
        best_val_loss = float("inf")
        run_phase("finetune", epochs_finetune, optimizer)

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    fold_summary = {
        "fold": fold_i,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **{f"best_{k}": v for k, v in best_metrics.items()},
    }
    with open(fold_dir / "summary.json", "w") as f:
        json.dump(fold_summary, f, indent=2)
    log.info(
        f"Fold {fold_i} done. best_val_loss={best_val_loss:.4f} | "
        f"stone_acc={best_metrics.get('stone_accuracy', float('nan')):.3f} | "
        f"stone_macro_f1={best_metrics.get('stone_macro_f1', float('nan')):.3f}"
    )
    return fold_summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",  type=int,  default=None)
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs per phase (smoke test)")
    parser.add_argument("--random-labels", action="store_true",
                        help="Permute training labels at the stone level before training "
                             "(random-label sanity-check baseline). Checkpoints and logs "
                             "are saved with a '_random' suffix so they never overwrite "
                             "the real-label run.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag appended to checkpoint directory and log files "
                             "(e.g. --tag resnet18_reg). Lets you run multiple configurations "
                             "without overwriting each other.")
    args = parser.parse_args()

    cfg  = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(cfg["training_multiclass"]["device"])
    log.info(f"Device: {device}")

    class_names = cfg["class_remapping"]["final_classes"]

    ds_module  = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds_module.load_multiclass_samples(
        images_csv, class_names, require_files_exist=True
    )

    # Drop excluded classes (e.g. OTH: too few stones, chemically heterogeneous)
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    if exclude:
        excl_idx = {i for i, c in enumerate(class_names) if c in exclude}
        all_samples = [s for s in all_samples if s.label not in excl_idx]
        class_names = [c for c in class_names if c not in exclude]
        cfg["model_multiclass"]["num_classes"] = len(class_names)
        log.info(
            f"Excluded classes {exclude} → {len(class_names)} classes remaining, "
            f"{len({s.stone_id for s in all_samples})} stones"
        )
    log.info(f"Classes ({len(class_names)}): {class_names}")

    # Stratify on dominant mineral class — more appropriate for a multiclass model
    # than binary pure/mixed stratification.
    dominant_labels = [s.label for s in all_samples]

    # Hold out test set — stratified on dominant mineral class
    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds_module.make_test_holdout(
        all_samples, test_fraction, seed, stratify_labels=dominant_labels
    )
    train_val_samples = [all_samples[i] for i in train_val_idx]
    train_val_dominant = [dominant_labels[i] for i in train_val_idx]

    if args.random_labels:
        rng = np.random.RandomState(seed)
        train_val_samples = _shuffle_labels_stone_level(train_val_samples, rng)
        log.info("--random-labels: stone-level labels have been shuffled (baseline mode)")

    # Folds — stratified on dominant mineral class
    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
        stratify_labels=train_val_dominant,
    )

    ckpt_dir = resolve_path(cfg, "checkpoints_multiclass_dir")
    if args.random_labels:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + "_random")
    if args.tag:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + f"_{args.tag}")
    ckpt_dir = ensure_dir(ckpt_dir)
    logs_dir = ensure_dir(resolve_path(cfg, "logs_dir"))
    suffix = "_random" if args.random_labels else ""
    if args.tag:
        suffix += f"_{args.tag}"

    fold_range = [args.fold] if args.fold is not None else range(len(folds))
    all_summaries: list[dict] = []

    for fold_i in fold_range:
        train_idx, val_idx = folds[fold_i]
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        log.info(f"=== Fold {fold_i} ===")
        summary = train_one_fold(
            fold_i, train_val_samples, train_idx, val_idx,
            cfg, device, fold_dir, class_names, quick=args.quick,
        )
        all_summaries.append(summary)

    # Cross-fold aggregate
    if args.fold is None and len(all_summaries) > 1:
        df = pd.DataFrame(all_summaries)
        metric_cols = [c for c in df.columns if c.startswith("best_") and c != "best_epoch"]
        agg = pd.concat([
            df[metric_cols].mean().rename("mean"),
            df[metric_cols].std().rename("std"),
        ], axis=1)
        agg.to_csv(logs_dir / f"cv_summary_multiclass{suffix}.csv")
        df.to_csv(logs_dir / f"fold_summaries_multiclass{suffix}.csv", index=False)
        log.info("Cross-fold results:")
        for metric in ["best_stone_accuracy", "best_stone_macro_f1"]:
            if metric in agg.index:
                log.info(f"  {metric}: {agg.at[metric, 'mean']:.3f} ± {agg.at[metric, 'std']:.3f}")

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    model_module = import_module("04_model_binary")
    criterion    = nn.CrossEntropyLoss()
    test_loader  = ds_module.make_multiclass_dataloader(all_samples, test_idx, cfg, train=False)

    test_fold_metrics: list[dict] = []
    for fold_i in fold_range:
        ckpt_path = ckpt_dir / f"fold_{fold_i}" / "best.pt"
        if not ckpt_path.exists():
            log.warning(f"  Fold {fold_i}: checkpoint not found, skipping")
            continue
        mc = cfg["model_multiclass"]
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = model_module.StoneClassifier(
            backbone_name=mc["backbone"],
            weights=mc["pretrained_weights"],
            num_classes=mc["num_classes"],
            dropout=mc["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        test_m = evaluate(model, test_loader, criterion, device, class_names)
        test_fold_metrics.append({"fold": fold_i, **test_m})
        log.info(
            f"  Fold {fold_i} test: "
            f"stone_acc={test_m['stone_accuracy']:.3f} | "
            f"stone_macro_f1={test_m['stone_macro_f1']:.3f}"
        )

    if test_fold_metrics:
        test_df = pd.DataFrame(test_fold_metrics)
        test_df.to_csv(logs_dir / f"test_fold_metrics_multiclass{suffix}.csv", index=False)
        if len(test_fold_metrics) > 1:
            numeric_cols = [c for c in test_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_df[numeric_cols].mean().rename("mean"),
                test_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / f"test_summary_multiclass{suffix}.csv")
            log.info("Test set summary (mean ± std across folds):")
            for metric in ["stone_accuracy", "stone_macro_f1"]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.3f} ± "
                        f"{test_summary.at[metric, 'std']:.3f}"
                    )


if __name__ == "__main__":
    main()
