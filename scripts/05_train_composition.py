"""
05_train_composition.py — Train the compositional regression model (Model B).

Training scheme (staged fine-tuning, same as Model A):
  Phase 1: Freeze backbone, train heads only for N epochs (warm-up)
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

Multi-task training:
  - Primary loss: composition regression (KL divergence by default)
  - Auxiliary loss: presence/absence detection head (BCE per class, independent)
  - Total loss = comp_loss + aux_loss_weight × aux_loss
  - Presence targets derived from composition: class present if fraction > presence_threshold

Usage:
  python scripts/05_train_composition.py              # all folds
  python scripts/05_train_composition.py --fold 0     # single fold
  python scripts/05_train_composition.py --quick      # 2 epochs/phase (smoke test)
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
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("train_composition")


# -----------------------------------------------------------------------------
# Random-label baseline helper
# -----------------------------------------------------------------------------
def _shuffle_compositions_stone_level(samples: list, rng: np.random.RandomState) -> list:
    """Return new samples with composition vectors permuted at the stone level.

    All images of a stone receive the same (shuffled) composition so there is no
    within-stone inconsistency.  Used for the random-label sanity-check baseline.
    """
    stone_ids_ordered: list[str] = []
    stone_to_comp: dict[str, list] = {}
    for s in samples:
        if s.stone_id not in stone_to_comp:
            stone_ids_ordered.append(s.stone_id)
            stone_to_comp[s.stone_id] = s.composition

    comps = [stone_to_comp[sid] for sid in stone_ids_ordered]
    perm = rng.permutation(len(comps)).tolist()
    shuffled_map = {sid: comps[perm[i]] for i, sid in enumerate(stone_ids_ordered)}
    return [dataclasses.replace(s, composition=shuffled_map[s.stone_id]) for s in samples]


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
def compute_kl_weights(
    samples: list, class_names: list[str], device: torch.device, eps: float = 1e-6
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training compositions.

    w_c = 1 / (mean true fraction of class c + eps), normalised so weights
    sum to n_classes.  Rare classes (small mean fraction) get higher weight.
    """
    comps = np.array([s.composition for s in samples])   # (N, n_classes)
    mean_fracs = comps.mean(axis=0)                       # (n_classes,)
    weights = 1.0 / (mean_fracs + eps)
    weights = weights / weights.sum() * len(class_names)  # normalise
    log.info("Weighted KL class weights (inv-freq, normalised):")
    for cls, w, mf in zip(class_names, weights, mean_fracs):
        log.info(f"  {cls}: mean_frac={mf:.4f}  weight={w:.3f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_criterion(loss_type: str, kl_weights: torch.Tensor | None = None):
    """Return a loss function (logits, targets) → scalar.

    logits:  (batch, n_classes) raw model output
    targets: (batch, n_classes) ground-truth composition, sums to 1.0

    When kl_weights is provided (and loss_type == "kl"), applies per-class
    inverse-frequency weighting before summing the KL terms.
    """
    if loss_type == "kl":
        if kl_weights is not None:
            def fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                log_pred = F.log_softmax(logits, dim=1)         # (B, C)
                kl_per_class = targets * (targets.clamp(min=1e-8).log() - log_pred)
                w = kl_weights.to(logits.device)
                return (kl_per_class * w).sum(dim=1).mean()
        else:
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
    metrics[f"{prefix}dominant_macro_f1"] = float(
        f1_score(dominant_true, dominant_pred, average="macro", zero_division=0)
    )
    per_class_f1 = f1_score(
        dominant_true, dominant_pred,
        labels=list(range(len(class_names))),
        average=None, zero_division=0,
    )
    for cls, val in zip(class_names, per_class_f1):
        metrics[f"{prefix}dominant_f1_{cls}"] = float(val)
    metrics[f"{prefix}aitchison"] = aitchison_distance(pred, true)

    return metrics


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    aux_loss_weight: float,
    presence_threshold: float,
) -> dict:
    model.train()
    running_comp_loss  = 0.0
    running_aux_loss   = 0.0
    running_total_loss = 0.0
    total = 0

    for images, comp_targets, _stone_ids, _labels in loader:
        images = images.to(device, non_blocking=True)
        comp_targets = comp_targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        comp_logits, pres_logits = model(images)

        comp_loss = criterion(comp_logits, comp_targets)

        # Aux: binary presence/absence per class (1 if fraction > threshold, else 0)
        pres_targets = (comp_targets > presence_threshold).float()
        aux_loss = F.binary_cross_entropy_with_logits(pres_logits, pres_targets)

        loss = comp_loss + aux_loss_weight * aux_loss
        loss.backward()
        optimizer.step()

        running_comp_loss  += comp_loss.item() * images.size(0)
        running_aux_loss   += aux_loss.item() * images.size(0)
        running_total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return {
        "comp_loss":  running_comp_loss  / total,
        "aux_loss":   running_aux_loss   / total,
        "total_loss": running_total_loss / total,
    }


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
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

        comp_logits, _pres_logits = model(images)
        loss = criterion(comp_logits, comp_dev)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        probs = F.softmax(comp_logits, dim=1).cpu().numpy()
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
def train_one_fold(
    fold_i: int,
    samples: list,
    train_idx: list[int],
    val_idx: list[int],
    cfg: dict,
    device: torch.device,
    fold_dir: Path,
    quick: bool = False,
    model_type: str = "parallel",
    kl_weights: torch.Tensor | None = None,
) -> dict:
    ensure_dir(fold_dir)

    ds_module    = import_module("03_dataset")
    model_module = import_module("04_model_composition")

    train_loader = ds_module.make_compositional_dataloader(samples, train_idx, cfg, train=True)
    val_loader   = ds_module.make_compositional_dataloader(samples, val_idx,   cfg, train=False)

    if model_type.split("_")[0] == "simple":
        model = model_module.build_simple_composition_model(cfg).to(device)
    else:
        model = model_module.build_composition_model(cfg).to(device)
    criterion = make_criterion(cfg["training_composition"]["loss"], kl_weights=kl_weights)
    class_names = cfg["class_remapping"]["final_classes"]
    aggregation = cfg["evaluation_composition"]["stone_level_aggregation"]

    tcfg = cfg["training_composition"]
    epochs_frozen   = 2 if quick else tcfg["epochs_frozen"]
    epochs_finetune = 2 if quick else tcfg["epochs_finetune"]
    patience           = tcfg["early_stopping_patience"]
    aux_loss_weight    = tcfg.get("aux_loss_weight", 0.3)
    presence_threshold = tcfg.get("presence_threshold", 0.05)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    epochs_since_best = 0
    best_ckpt_path = fold_dir / f"best_{model_type}.pt"

    def run_phase(phase_name: str, n_epochs: int, optimizer: torch.optim.Optimizer) -> bool:
        nonlocal best_val_loss, best_epoch, best_metrics, epochs_since_best

        for epoch in range(n_epochs):
            t0 = time.time()
            train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                aux_loss_weight, presence_threshold,
            )
            val_m = evaluate(model, val_loader, criterion, device,
                             class_names, aggregation)
            elapsed = time.time() - t0

            global_epoch = len(history)
            row = {
                "fold": fold_i, "phase": phase_name,
                "epoch": global_epoch, "phase_epoch": epoch,
                "train_comp_loss":  train_m["comp_loss"],
                "train_aux_loss":   train_m["aux_loss"],
                "train_total_loss": train_m["total_loss"],
                **{f"val_{k}": v for k, v in val_m.items()},
                "seconds": round(elapsed, 1),
            }
            history.append(row)
            log.info(
                f"  [{phase_name}] epoch {epoch+1}/{n_epochs} | "
                f"train_total={train_m['total_loss']:.4f} "
                f"(comp={train_m['comp_loss']:.4f} aux={train_m['aux_loss']:.4f}) | "
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
            {"params": model.backbone_parameters(),  "lr": tcfg["learning_rate_backbone"]},
            {"params": model.head_parameters(),      "lr": tcfg["learning_rate_head"]},
        ], weight_decay=tcfg["weight_decay"])
        epochs_since_best = 0
        best_val_loss = float("inf")
        run_phase("finetune", epochs_finetune, optimizer)

    pd.DataFrame(history).to_csv(fold_dir / f"history_{model_type}.csv", index=False)
    fold_summary = {
        "fold": fold_i,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **{f"best_{k}": v for k, v in best_metrics.items()},
    }
    with open(fold_dir / f"summary_{model_type}.json", "w") as f:
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
    parser.add_argument("--model", choices=["parallel", "simple"], default="simple",
                        help="simple = single shared head (default), parallel = expert heads per class")
    parser.add_argument("--random-labels", action="store_true",
                        help="Permute composition vectors at the stone level before training "
                             "(random-label sanity-check baseline). Checkpoints and logs "
                             "are saved with a '_random' suffix so they never overwrite "
                             "the real-label run.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag appended to checkpoint directory and log files "
                             "(e.g. --tag resnet18_reg). Lets you run multiple configurations "
                             "without overwriting each other.")
    args = parser.parse_args()
    model_type = args.model
    # model_type_tag is used for all file naming so random runs never overwrite real runs
    model_type_tag = f"{model_type}_random" if args.random_labels else model_type
    if args.tag:
        model_type_tag = f"{model_type_tag}_{args.tag}"

    cfg  = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(cfg["training_composition"]["device"])
    log.info(f"Device: {device}")
    if torch.cuda.is_available() and cfg["training_composition"].get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        log.info("cuDNN benchmark enabled")

    # Load compositional samples
    ds_module    = import_module("03_dataset")
    images_csv   = resolve_path(cfg, "processed_images_csv")
    final_classes = cfg["class_remapping"]["final_classes"]

    all_samples = ds_module.load_compositional_samples(
        images_csv, final_classes, require_files_exist=True
    )

    # Drop excluded classes (e.g. OTH) from the composition model:
    #   1. Remove stones where an excluded class is the dominant component
    #      (their composition vector is unreliable as a regression target).
    #   2. Drop the excluded column(s) from every remaining stone's composition
    #      and renormalise so the vector still sums to 1.
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    if exclude:
        excl_idx = [i for i, c in enumerate(final_classes) if c in exclude]
        keep_idx = [i for i, c in enumerate(final_classes) if c not in exclude]
        # Step 1: remove dominant-excluded stones
        all_samples = [
            s for s in all_samples
            if int(np.argmax(s.composition)) not in excl_idx
        ]
        # Step 2: drop excluded columns and renormalise
        trimmed = []
        for s in all_samples:
            comp = [s.composition[i] for i in keep_idx]
            total = sum(comp)
            comp = [x / total for x in comp] if total > 0 else comp
            trimmed.append(dataclasses.replace(s, composition=comp))
        all_samples = trimmed
        final_classes = [c for c in final_classes if c not in exclude]
        # Update cfg so build_composition_model uses the trimmed class list
        cfg["class_remapping"]["final_classes"] = final_classes
        log.info(
            f"Excluded classes {exclude} → {len(final_classes)} classes remaining, "
            f"{len({s.stone_id for s in all_samples})} stones"
        )

    # Hold out test set — stratified on dominant mineral class
    test_fraction = cfg["cv"]["test_fraction"]
    all_dominant = [int(np.argmax(s.composition)) for s in all_samples]
    train_val_idx, test_idx = ds_module.make_test_holdout(
        all_samples, test_fraction, seed, stratify_labels=all_dominant
    )
    train_val_samples = [all_samples[i] for i in train_val_idx]

    if args.random_labels:
        rng = np.random.RandomState(seed)
        train_val_samples = _shuffle_compositions_stone_level(train_val_samples, rng)
        log.info("--random-labels: stone-level composition vectors have been shuffled (baseline mode)")

    # Folds — stratified on dominant mineral class (argmax of composition vector)
    # This ensures each fold has a representative mix of all mineral types,
    # which is more appropriate for a composition regression model than binary stratification.
    dominant_labels = [int(np.argmax(s.composition)) for s in train_val_samples]
    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
        stratify_labels=dominant_labels,
    )

    ckpt_key  = "checkpoints_composition_dir"
    ckpt_dir  = resolve_path(cfg, ckpt_key)
    if args.tag:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + f"_{args.tag}")
    ckpt_dir  = ensure_dir(ckpt_dir)
    logs_dir  = ensure_dir(resolve_path(cfg, "logs_dir"))
    log.info(f"Model type: {model_type_tag}  |  checkpoints → {ckpt_dir}")

    # Weighted KL: compute inverse-frequency weights from the full train+val pool.
    # Per-fold train split weights would be slightly more correct, but the overall
    # pool is stable across folds and avoids recomputing weights each fold.
    tcfg = cfg["training_composition"]
    kl_weights: torch.Tensor | None = None
    if tcfg.get("use_weighted_kl", False) and tcfg["loss"] == "kl":
        log.info("use_weighted_kl=true — computing inverse-frequency weights from train+val pool")
        kl_weights = compute_kl_weights(train_val_samples, final_classes, device)
    else:
        log.info("use_weighted_kl=false — using standard (unweighted) KL divergence")

    fold_range = [args.fold] if args.fold is not None else range(len(folds))
    all_summaries: list[dict] = []

    for fold_i in fold_range:
        train_idx, val_idx = folds[fold_i]
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        log.info(f"=== Fold {fold_i} ===")
        summary = train_one_fold(
            fold_i, train_val_samples, train_idx, val_idx,
            cfg, device, fold_dir, quick=args.quick, model_type=model_type_tag,
            kl_weights=kl_weights,
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
        agg.to_csv(logs_dir / f"cv_summary_{model_type_tag}.csv")
        df.to_csv(logs_dir / f"fold_summaries_{model_type_tag}.csv", index=False)

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
    criterion    = make_criterion(cfg["training_composition"]["loss"])
    aggregation  = cfg["evaluation_composition"]["stone_level_aggregation"]
    test_loader  = ds_module.make_compositional_dataloader(
        all_samples, test_idx, cfg, train=False
    )

    build_fn = (model_module.build_simple_composition_model
                if model_type == "simple"
                else model_module.build_composition_model)

    test_fold_metrics: list[dict] = []
    for fold_i in fold_range:
        ckpt_path = ckpt_dir / f"fold_{fold_i}" / f"best_{model_type_tag}.pt"
        if not ckpt_path.exists():
            log.warning(f"  Fold {fold_i}: checkpoint not found at {ckpt_path}, skipping")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_fn(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        test_m = evaluate(model, test_loader, criterion, device,
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
        test_df.to_csv(logs_dir / f"test_fold_metrics_{model_type_tag}.csv", index=False)
        if len(test_fold_metrics) > 1:
            numeric_cols = [c for c in test_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_df[numeric_cols].mean().rename("mean"),
                test_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / f"test_summary_{model_type_tag}.csv")
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
