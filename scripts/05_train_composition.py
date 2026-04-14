"""
05_train_moe.py — Train the MoE compositional regression model (Model B).

Training scheme (staged fine-tuning, same as Model A):
  Phase 1: Freeze backbone, train expert heads only for N epochs (warm-up)
  Phase 2: Unfreeze backbone, fine-tune end-to-end at lower LR for N epochs

Per-fold flow:
  - Build train/val compositional DataLoaders
  - Run Phase 1 + Phase 2 with early stopping on val loss
  - Aggregate per-image predictions to per-stone predictions (mean composition)
  - Save best checkpoint + per-epoch metrics CSV + fold summary JSON

After all folds: write cross-fold summary (mean ± std of val metrics).

Evaluation metrics (image-level AND stone-level):
  - MAE per class: mean absolute error for each of the 6 composition classes
  - MAE overall:   mean of the per-class MAEs
  - Dominant acc:  fraction of stones where argmax(pred) == argmax(true)
  - Aitchison:     mean Euclidean distance in centered log-ratio (CLR) space

Usage:
  python scripts/05_train_moe.py              # all folds
  python scripts/05_train_moe.py --fold 0     # single fold
  python scripts/05_train_moe.py --quick      # 2 epochs/phase (smoke test)
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
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("train_moe")


# -----------------------------------------------------------------------------
# Device helper
# -----------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in ("mps", "cuda"):
        return torch.device(requested)
    log.warning(f"{requested.upper()} requested but not available — falling back to CPU")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------------
def make_criterion(loss_type: str):
    """Return a loss function (logits, targets) → scalar.

    logits:  (batch, n_classes) raw model output
    targets: (batch, n_classes) ground-truth composition, sums to 1.0
    """
    if loss_type == "kl":
        # KLDivLoss expects log-probabilities as input, probabilities as target
        def fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.kl_div(
                F.log_softmax(logits, dim=1), targets, reduction="batchmean"
            )
    elif loss_type == "l1":
        def fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(F.softmax(logits, dim=1), targets)
    elif loss_type == "mse":
        def fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(F.softmax(logits, dim=1), targets)
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Choose: kl, l1, mse")
    return fn


# -----------------------------------------------------------------------------
# Compositional metrics
# -----------------------------------------------------------------------------
def aitchison_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Aitchison distance between rows of p and q.

    Aitchison distance = Euclidean distance in centered log-ratio (CLR) space.
    CLR(x)_i = log(x_i) - mean(log(x))

    Args:
        p, q: (N, D) arrays of compositions (will be clipped to avoid log(0))
    Returns:
        Scalar mean distance over N samples.
    """
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    clr_p = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
    clr_q = np.log(q) - np.log(q).mean(axis=1, keepdims=True)
    return float(np.sqrt(((clr_p - clr_q) ** 2).sum(axis=1)).mean())


def compute_compositional_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    class_names: list[str],
    prefix: str = "",
) -> dict:
    """Compute MAE per class, overall MAE, dominant-component accuracy, Aitchison distance.

    Args:
        pred, true: (N, n_classes) float arrays of composition vectors
        class_names: list of class names matching the column order
        prefix:      string prefix for dict keys (e.g. "stone_")
    """
    mae_per_class = np.abs(pred - true).mean(axis=0)  # (n_classes,)
    metrics = {f"{prefix}mae_overall": float(mae_per_class.mean())}
    for cls, mae in zip(class_names, mae_per_class):
        metrics[f"{prefix}mae_{cls}"] = float(mae)

    dominant_pred = pred.argmax(axis=1)
    dominant_true = true.argmax(axis=1)
    metrics[f"{prefix}dominant_acc"] = float((dominant_pred == dominant_true).mean())
    metrics[f"{prefix}aitchison"] = aitchison_distance(pred, true)

    return metrics


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_one_epoch_moe(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> dict:
    model.train()
    running_loss = 0.0
    total = 0

    for images, comp_targets, _stone_ids, _labels in loader:
        images = images.to(device, non_blocking=True)
        comp_targets = comp_targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, comp_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return {"loss": running_loss / total}


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_moe(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    class_names: list[str],
    aggregation: str = "mean_composition",
) -> dict:
    """Evaluate with image-level and stone-level compositional metrics."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_stone_ids: list[str] = []
    all_pred_comp: list[np.ndarray] = []
    all_true_comp: list[np.ndarray] = []
    all_labels: list[int] = []

    for images, comp_targets, stone_ids, labels in loader:
        images = images.to(device, non_blocking=True)
        comp_dev = comp_targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, comp_dev)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_pred_comp.append(probs)
        all_true_comp.append(comp_targets.numpy())
        all_stone_ids.extend(list(stone_ids))
        all_labels.extend(labels.tolist() if hasattr(labels, "tolist") else list(labels))

    pred_arr = np.concatenate(all_pred_comp, axis=0)   # (N_images, n_classes)
    true_arr = np.concatenate(all_true_comp, axis=0)

    # --- Image-level metrics ---
    img_metrics = {
        "loss": running_loss / total,
        **compute_compositional_metrics(pred_arr, true_arr, class_names, prefix="img_"),
    }

    # --- Stone-level aggregation ---
    df = pd.DataFrame({
        "stone_id": all_stone_ids,
        "label":    all_labels,
        **{f"pred_{c}": pred_arr[:, i] for i, c in enumerate(class_names)},
        **{f"true_{c}": true_arr[:, i] for i, c in enumerate(class_names)},
    })

    if aggregation == "mean_composition":
        pred_cols = [f"pred_{c}" for c in class_names]
        true_cols = [f"true_{c}" for c in class_names]
        stone_df = df.groupby("stone_id")[pred_cols + true_cols].mean().reset_index()

        # Re-normalise predictions to enforce simplex after averaging
        pred_sum = stone_df[pred_cols].sum(axis=1)
        stone_df[pred_cols] = stone_df[pred_cols].div(pred_sum, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    stone_pred = stone_df[[f"pred_{c}" for c in class_names]].values
    stone_true = stone_df[[f"true_{c}" for c in class_names]].values

    stone_metrics = compute_compositional_metrics(
        stone_pred, stone_true, class_names, prefix="stone_"
    )

    return {**img_metrics, **stone_metrics}


# -----------------------------------------------------------------------------
# Per-fold training
# -----------------------------------------------------------------------------
def train_one_fold_moe(
    fold_i: int,
    samples: list,
    train_idx: list[int],
    val_idx: list[int],
    cfg: dict,
    device: torch.device,
    fold_dir: Path,
    quick: bool = False,
) -> dict:
    ensure_dir(fold_dir)

    ds_module   = import_module("03_dataset")
    model_module = import_module("04_model_composition")

    train_loader = ds_module.make_compositional_dataloader(samples, train_idx, cfg, train=True)
    val_loader   = ds_module.make_compositional_dataloader(samples, val_idx,   cfg, train=False)

    model = model_module.build_moe_model(cfg).to(device)
    criterion = make_criterion(cfg["training_moe"]["loss"])
    class_names = cfg["class_remapping"]["final_classes"]
    aggregation = cfg["evaluation_moe"]["stone_level_aggregation"]

    tcfg = cfg["training_moe"]
    epochs_frozen   = 2 if quick else tcfg["epochs_frozen"]
    epochs_finetune = 2 if quick else tcfg["epochs_finetune"]
    patience = tcfg["early_stopping_patience"]

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    epochs_since_best = 0
    best_ckpt_path = fold_dir / "best_moe.pt"

    def run_phase(phase_name: str, n_epochs: int, optimizer: torch.optim.Optimizer) -> bool:
        nonlocal best_val_loss, best_epoch, best_metrics, epochs_since_best

        for epoch in range(n_epochs):
            t0 = time.time()
            train_m = train_one_epoch_moe(model, train_loader, optimizer, criterion, device)
            val_m   = evaluate_moe(model, val_loader, criterion, device,
                                   class_names, aggregation)
            elapsed = time.time() - t0

            global_epoch = len(history)
            row = {
                "fold": fold_i, "phase": phase_name,
                "epoch": global_epoch, "phase_epoch": epoch,
                "train_loss": train_m["loss"],
                **{f"val_{k}": v for k, v in val_m.items()},
                "seconds": round(elapsed, 1),
            }
            history.append(row)
            log.info(
                f"  [{phase_name}] epoch {epoch+1}/{n_epochs} | "
                f"train_loss={train_m['loss']:.4f} | "
                f"val_loss={val_m['loss']:.4f} | "
                f"val_stone_mae={val_m['stone_mae_overall']:.4f} | "
                f"val_stone_dom_acc={val_m['stone_dominant_acc']:.3f} | "
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
                    log.info(f"  Early stopping (no improvement for {patience} epochs)")
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
        run_phase("finetune", epochs_finetune, optimizer)

    pd.DataFrame(history).to_csv(fold_dir / "history_moe.csv", index=False)
    fold_summary = {
        "fold": fold_i,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **{f"best_{k}": v for k, v in best_metrics.items()},
    }
    with open(fold_dir / "summary_moe.json", "w") as f:
        json.dump(fold_summary, f, indent=2)

    log.info(
        f"Fold {fold_i} done. best_val_loss={best_val_loss:.4f} | "
        f"stone_mae={best_metrics.get('stone_mae_overall', float('nan')):.4f} | "
        f"stone_dom_acc={best_metrics.get('stone_dominant_acc', float('nan')):.3f} | "
        f"stone_aitchison={best_metrics.get('stone_aitchison', float('nan')):.4f}"
    )
    return fold_summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",  type=int,  default=None,  help="Run only this fold")
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs per phase (smoke test)")
    args = parser.parse_args()

    cfg  = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(cfg["training_moe"]["device"])
    log.info(f"Device: {device}")

    # Load compositional samples
    ds_module    = import_module("03_dataset")
    images_csv   = resolve_path(cfg, "processed_images_csv")
    final_classes = cfg["class_remapping"]["final_classes"]

    all_samples = ds_module.load_compositional_samples(
        images_csv, final_classes, require_files_exist=True
    )

    # Hold out test set before CV
    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds_module.make_test_holdout(all_samples, test_fraction, seed)
    train_val_samples = [all_samples[i] for i in train_val_idx]

    # Folds — stratified on binary label (same split as Model A for fair comparison)
    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
    )

    ckpt_moe_dir = ensure_dir(resolve_path(cfg, "checkpoints_moe_dir"))
    logs_dir     = ensure_dir(resolve_path(cfg, "logs_dir"))

    fold_range = [args.fold] if args.fold is not None else range(len(folds))
    all_summaries: list[dict] = []

    for fold_i in fold_range:
        train_idx, val_idx = folds[fold_i]
        fold_dir = ckpt_moe_dir / f"fold_{fold_i}"
        log.info(f"=== Fold {fold_i} ===")
        summary = train_one_fold_moe(
            fold_i, train_val_samples, train_idx, val_idx,
            cfg, device, fold_dir, quick=args.quick,
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
        agg.to_csv(logs_dir / "cv_summary_moe.csv")
        df.to_csv(logs_dir / "fold_summaries_moe.csv", index=False)

        log.info("Cross-fold results:")
        for metric in [
            "best_stone_mae_overall",
            "best_stone_dominant_acc",
            "best_stone_aitchison",
        ]:
            if metric in agg.index:
                log.info(
                    f"  {metric}: "
                    f"{agg.at[metric, 'mean']:.4f} ± {agg.at[metric, 'std']:.4f}"
                )

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    model_module = import_module("04_model_composition")
    criterion    = make_criterion(cfg["training_moe"]["loss"])
    aggregation  = cfg["evaluation_moe"]["stone_level_aggregation"]
    test_loader  = ds_module.make_compositional_dataloader(
        all_samples, test_idx, cfg, train=False
    )

    test_fold_metrics: list[dict] = []
    for fold_i in fold_range:
        ckpt_path = ckpt_moe_dir / f"fold_{fold_i}" / "best_moe.pt"
        if not ckpt_path.exists():
            log.warning(f"  Fold {fold_i}: checkpoint not found at {ckpt_path}, skipping")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = model_module.build_moe_model(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        test_m = evaluate_moe(model, test_loader, criterion, device,
                              final_classes, aggregation)
        test_fold_metrics.append({"fold": fold_i, **test_m})
        log.info(
            f"  Fold {fold_i} test: "
            f"stone_mae={test_m['stone_mae_overall']:.4f} | "
            f"stone_dom_acc={test_m['stone_dominant_acc']:.3f} | "
            f"stone_aitchison={test_m['stone_aitchison']:.4f}"
        )

    if test_fold_metrics:
        test_df = pd.DataFrame(test_fold_metrics)
        test_df.to_csv(logs_dir / "test_fold_metrics_moe.csv", index=False)
        if len(test_fold_metrics) > 1:
            numeric_cols = [c for c in test_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_df[numeric_cols].mean().rename("mean"),
                test_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / "test_summary_moe.csv")
            log.info("Test set summary (mean ± std across folds):")
            for metric in ["stone_mae_overall", "stone_dominant_acc", "stone_aitchison"]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.4f} ± "
                        f"{test_summary.at[metric, 'std']:.4f}"
                    )


if __name__ == "__main__":
    main()
