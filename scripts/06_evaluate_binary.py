"""
06_evaluate.py — Load best checkpoints and produce final evaluation outputs.

For each fold:
  - Loads the best checkpoint saved by 05_train.py
  - Runs inference on that fold's validation set (no gradient, eval mode)
  - Aggregates per-image → per-stone predictions
  - Records image-level and stone-level metrics

Across all folds:
  - Cross-fold mean ± std table (CSV + console)
  - Aggregated confusion matrix (figure)
  - Mean ROC curve with ± std band (figure)
  - Per-stone predictions CSV (for manual review / error analysis)

Outputs (all under outputs/):
  figures/confusion_matrix.png
  figures/roc_curve.png
  logs/eval_fold_summaries.csv    — per-fold metrics
  logs/eval_cv_summary.csv        — mean ± std across folds
  logs/eval_stone_predictions.csv — per-stone prob + pred + true label

Usage:
  python scripts/06_evaluate.py                        # evaluate all folds (fine-tuned)
  python scripts/06_evaluate.py --fold 0               # single fold only
  python scripts/06_evaluate.py --frozen-baseline      # evaluate the frozen backbone baseline

Run 05_train.py first to produce the checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from utils import PALETTE, ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("evaluate")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    """Run the model over a DataLoader and return a per-image DataFrame.

    Columns: stone_id, label, prob_mixed, pred
    """
    model.eval()
    rows: list[dict] = []

    for images, labels, stone_ids in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()  # P(mixed)
        preds = logits.argmax(dim=1).cpu()

        for stone_id, label, prob, pred in zip(
            stone_ids, labels.tolist(), probs.tolist(), preds.tolist()
        ):
            rows.append({
                "stone_id": stone_id,
                "label": label,
                "prob_mixed": prob,
                "pred": pred,
            })

    return pd.DataFrame(rows)


def aggregate_to_stone_level(img_df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """Collapse per-image rows to one row per stone."""
    if aggregation == "mean_probability":
        stone_df = img_df.groupby("stone_id").agg(
            label=("label", "first"),
            prob_mixed=("prob_mixed", "mean"),
        ).reset_index()
        stone_df["pred"] = (stone_df["prob_mixed"] >= 0.5).astype(int)
    elif aggregation == "majority_vote":
        stone_df = img_df.groupby("stone_id").agg(
            label=("label", "first"),
            pred=("pred", lambda s: int(s.mode().iloc[0])),
            prob_mixed=("prob_mixed", "mean"),
        ).reset_index()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    return stone_df


def compute_metrics(df: pd.DataFrame, prefix: str = "") -> dict:
    """Compute classification metrics from a DataFrame with label/pred/prob_mixed."""
    metrics: dict = {}
    metrics[f"{prefix}accuracy"] = accuracy_score(df["label"], df["pred"])
    metrics[f"{prefix}precision"] = precision_score(df["label"], df["pred"], zero_division=0)
    metrics[f"{prefix}recall"] = recall_score(df["label"], df["pred"], zero_division=0)
    metrics[f"{prefix}f1"] = f1_score(df["label"], df["pred"], zero_division=0)
    if df["label"].nunique() > 1:
        metrics[f"{prefix}roc_auc"] = roc_auc_score(df["label"], df["prob_mixed"])
    return metrics


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_training_curves(
    history_dfs: list[pd.DataFrame],
    output_path: Path,
    title: str = "Training and validation loss",
    train_col: str = "train_loss",
    val_col: str = "val_loss",
) -> None:
    """Plot mean ± std train and val loss across folds.

    Draws a vertical dashed line where the backbone is unfrozen (frozen→finetune
    phase boundary) — this is informative for staged fine-tuning methods.
    Shaded bands show ± 1 std across folds.
    """
    # Find the epoch at which the finetune phase begins (first fold that has it)
    phase_boundary: int | None = None
    for df in history_dfs:
        ft = df[df["phase"] == "finetune"]
        if not ft.empty:
            b = int(ft["epoch"].min())
            if phase_boundary is None or b < phase_boundary:
                phase_boundary = b

    # Collect per-epoch values across folds
    max_epoch = max(int(df["epoch"].max()) for df in history_dfs)
    train_by_epoch: dict[int, list[float]] = {e: [] for e in range(max_epoch + 1)}
    val_by_epoch:   dict[int, list[float]] = {e: [] for e in range(max_epoch + 1)}

    for df in history_dfs:
        for _, row in df.iterrows():
            e = int(row["epoch"])
            if train_col in df.columns and pd.notna(row[train_col]):
                train_by_epoch[e].append(float(row[train_col]))
            if val_col in df.columns and pd.notna(row[val_col]):
                val_by_epoch[e].append(float(row[val_col]))

    epochs     = [e for e in range(max_epoch + 1) if train_by_epoch[e]]
    train_mean = np.array([np.mean(train_by_epoch[e]) for e in epochs])
    train_std  = np.array([np.std(train_by_epoch[e])  for e in epochs])
    val_mean   = np.array([np.mean(val_by_epoch[e])   for e in epochs])
    val_std    = np.array([np.std(val_by_epoch[e])    for e in epochs])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_mean, color=PALETTE["train"], lw=2, label="Train loss")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color=PALETTE["train"])

    ax.plot(epochs, val_mean, color=PALETTE["val"], lw=2, label="Val loss")
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color=PALETTE["val"])

    if phase_boundary is not None:
        ax.axvline(phase_boundary - 0.5, color="gray", linestyle="--", lw=1.5,
                   label="Backbone unfrozen")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion matrix (aggregated across folds)",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_total = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = 100 * cm[i, j] / row_total if row_total > 0 else 0
            ax.text(
                j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_roc_curves(
    fold_fprs: list[np.ndarray],
    fold_tprs: list[np.ndarray],
    fold_aucs: list[float],
    output_path: Path,
    title: str = "ROC curve — stone-level (cross-validation)",
) -> None:
    """Plot per-fold ROC curves plus mean ± std band."""
    fig, ax = plt.subplots(figsize=(6, 5))

    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs: list[np.ndarray] = []

    for i, (fpr, tpr, auc_val) in enumerate(zip(fold_fprs, fold_tprs, fold_aucs)):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        ax.plot(fpr, tpr, alpha=0.3, lw=1, label=f"Fold {i} (AUC={auc_val:.3f})")

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tprs, axis=0)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="navy", lw=2,
        label=f"Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})",
    )
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.15, color="navy", label="± 1 std",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Per-fold evaluation
# -----------------------------------------------------------------------------
def evaluate_fold(
    fold_i: int,
    samples: list,
    val_idx: list[int],
    cfg: dict,
    device: torch.device,
    fold_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    """Load best checkpoint and evaluate val set. Returns (metrics_dict, stone_df)."""
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for fold {fold_i} at {ckpt_path}. Run 05_train.py first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_module = import_module("04_model_binary")
    model = model_module.build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    log.info(
        f"Fold {fold_i}: loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_metrics'].get('loss', '?'):.4f})"
    )

    ds_module = import_module("03_dataset")
    val_loader = ds_module.make_dataloader(samples, val_idx, cfg, train=False)

    img_df = run_inference(model, val_loader, device)
    img_df["fold"] = fold_i

    aggregation = cfg["evaluation"]["stone_level_aggregation"]
    stone_df = aggregate_to_stone_level(img_df, aggregation)
    stone_df["fold"] = fold_i

    img_metrics = compute_metrics(img_df, prefix="img_")
    stone_metrics = compute_metrics(stone_df, prefix="stone_")

    metrics = {"fold": fold_i, **img_metrics, **stone_metrics}

    # Log key numbers
    log.info(
        f"  Stone-level — acc={stone_metrics.get('stone_accuracy', 0):.3f}  "
        f"f1={stone_metrics.get('stone_f1', 0):.3f}  "
        f"roc_auc={stone_metrics.get('stone_roc_auc', float('nan')):.3f}"
    )

    return metrics, stone_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Evaluate only this fold")
    parser.add_argument("--frozen-baseline", action="store_true",
                        help="Load checkpoints from checkpoints_frozen/ (the --no-finetune run). "
                             "Outputs are written with a _frozen suffix.")
    parser.add_argument("--random-labels", action="store_true",
                        help="Load checkpoints from checkpoints_random/ (the --random-labels "
                             "training run). Outputs are written with a _random suffix.")
    parser.add_argument("--purity-threshold", type=float, default=None,
                        help="Override the purity threshold (e.g. 90) used to define "
                             "pure vs mixed when building samples for fold reconstruction "
                             "and test evaluation.  Must match the value used during "
                             "training.  Adds a '_t<value>' suffix to all outputs.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag — must match the --tag used during training "
                             "(e.g. --tag wd001_lrlow). Points evaluation at the tagged "
                             "checkpoint directory and adds the tag to all output filenames.")
    args = parser.parse_args()

    cfg = load_config()

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Resolve device (same helper pattern as 05_train.py)
    requested = cfg["training"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Resolve purity threshold: CLI flag overrides config
    purity_threshold = args.purity_threshold
    if purity_threshold is None:
        purity_threshold = float(cfg["data"]["purity_threshold"])
    final_classes = cfg["class_remapping"]["final_classes"]
    threshold_tag = f"_t{int(purity_threshold)}" if args.purity_threshold is not None else ""

    # Build samples + folds (must use same seed/threshold as training to get same splits)
    ds_module = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds_module.load_image_samples(
        images_csv,
        require_files_exist=True,
        purity_threshold=purity_threshold,
        final_classes=final_classes,
    )

    # Reconstruct the same test holdout used during training
    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds_module.make_test_holdout(all_samples, test_fraction, seed)
    train_val_samples = [all_samples[i] for i in train_val_idx]

    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
    )

    ckpt_key = "checkpoints_frozen_dir" if args.frozen_baseline else "checkpoints_dir"
    ckpt_dir = resolve_path(cfg, ckpt_key)
    suffix = threshold_tag
    if args.frozen_baseline:
        suffix += "_frozen"
    if threshold_tag:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + threshold_tag)
    if args.random_labels:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + "_random")
        suffix += "_random"
    if args.tag:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + f"_{args.tag}")
        suffix += f"_{args.tag}"
    title_tag = ""
    if threshold_tag:
        title_tag += f" — {int(purity_threshold)}% purity threshold"
    if args.random_labels:
        title_tag += " — random-label baseline"
    if args.tag:
        title_tag += f" — {args.tag}"
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))
    logs_dir = ensure_dir(resolve_path(cfg, "logs_dir"))

    fold_range = [args.fold] if args.fold is not None else range(len(folds))

    # -------------------------------------------------------------------------
    # Training curves (reads history.csv written by 05_train_binary.py)
    # -------------------------------------------------------------------------
    history_dfs = []
    for fold_i in fold_range:
        hist_path = ckpt_dir / f"fold_{fold_i}" / "history.csv"
        if hist_path.exists():
            history_dfs.append(pd.read_csv(hist_path))
    if history_dfs:
        plot_training_curves(
            history_dfs,
            output_path=figures_dir / f"training_curves{suffix}.png",
            title=f"Training and validation loss — Model A (binary){title_tag}",
        )
    else:
        log.warning("No history.csv files found — skipping training curves plot")

    all_metrics: list[dict] = []
    all_stone_dfs: list[pd.DataFrame] = []
    fold_roc: list[tuple[np.ndarray, np.ndarray, float]] = []  # (fpr, tpr, auc)

    for fold_i in fold_range:
        _, val_idx = folds[fold_i]
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        metrics, stone_df = evaluate_fold(
            fold_i, train_val_samples, val_idx, cfg, device, fold_dir
        )
        all_metrics.append(metrics)
        all_stone_dfs.append(stone_df)

        # Collect ROC data for the plot (stone-level)
        if stone_df["label"].nunique() > 1:
            fpr, tpr, _ = roc_curve(stone_df["label"], stone_df["prob_mixed"])
            auc_val = auc(fpr, tpr)
            fold_roc.append((fpr, tpr, auc_val))

    # -------------------------------------------------------------------------
    # Save per-stone predictions
    # -------------------------------------------------------------------------
    all_stones = pd.concat(all_stone_dfs, ignore_index=True)
    all_stones.to_csv(logs_dir / f"eval_stone_predictions{suffix}.csv", index=False)
    log.info(f"Saved: {logs_dir / f'eval_stone_predictions{suffix}.csv'}")

    # -------------------------------------------------------------------------
    # Per-fold metrics CSV
    # -------------------------------------------------------------------------
    fold_df = pd.DataFrame(all_metrics)
    fold_df.to_csv(logs_dir / f"eval_fold_summaries{suffix}.csv", index=False)
    log.info(f"Saved: {logs_dir / f'eval_fold_summaries{suffix}.csv'}")

    # -------------------------------------------------------------------------
    # Cross-fold summary (only when > 1 fold evaluated)
    # -------------------------------------------------------------------------
    if len(all_metrics) > 1:
        numeric_cols = [c for c in fold_df.columns if c != "fold"]
        summary = pd.concat([
            fold_df[numeric_cols].mean().rename("mean"),
            fold_df[numeric_cols].std().rename("std"),
        ], axis=1)
        summary.to_csv(logs_dir / f"eval_cv_summary{suffix}.csv")
        log.info(f"Saved: {logs_dir / f'eval_cv_summary{suffix}.csv'}")

        log.info("Cross-fold results:")
        for metric in ["stone_accuracy", "stone_f1", "stone_roc_auc"]:
            if metric in summary.index:
                log.info(
                    f"  {metric}: "
                    f"{summary.at[metric, 'mean']:.3f} ± {summary.at[metric, 'std']:.3f}"
                )

    # -------------------------------------------------------------------------
    # Aggregated confusion matrix (stone-level, all folds combined)
    # -------------------------------------------------------------------------
    cm = confusion_matrix(all_stones["label"], all_stones["pred"])
    plot_confusion_matrix(
        cm,
        class_names=["pure", "mixed"],
        output_path=figures_dir / f"confusion_matrix{suffix}.png",
        title=f"Confusion matrix — Model A, binary (cross-validation){title_tag}",
    )

    # -------------------------------------------------------------------------
    # ROC curve
    # -------------------------------------------------------------------------
    if fold_roc:
        fprs, tprs, aucs = zip(*fold_roc)
        plot_roc_curves(
            list(fprs), list(tprs), list(aucs),
            output_path=figures_dir / f"roc_curve{suffix}.png",
            title=f"ROC curve — stone-level (cross-validation){title_tag}",
        )
    else:
        log.warning("Skipping ROC plot — not enough class diversity in val sets")

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # Each fold's best checkpoint is evaluated on the same test set; results
    # are averaged to give the final unbiased performance estimate.
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    model_module = import_module("04_model_binary")
    test_metrics_all: list[dict] = []
    test_stone_dfs:   list[pd.DataFrame] = []
    test_roc: list[tuple[np.ndarray, np.ndarray, float]] = []

    for fold_i in fold_range:
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        # evaluate_fold works with any (samples, indices) pair
        t_metrics, t_stone_df = evaluate_fold(
            fold_i, all_samples, test_idx, cfg, device, fold_dir
        )
        test_metrics_all.append(t_metrics)
        test_stone_dfs.append(t_stone_df)
        if t_stone_df["label"].nunique() > 1:
            fpr, tpr, _ = roc_curve(t_stone_df["label"], t_stone_df["prob_mixed"])
            test_roc.append((fpr, tpr, auc(fpr, tpr)))

    if test_metrics_all:
        test_fold_df = pd.DataFrame(test_metrics_all)
        test_fold_df.to_csv(logs_dir / f"eval_test_fold_summaries{suffix}.csv", index=False)

        # Aggregate stone predictions across folds (all folds see the same test stones)
        all_test_stones = pd.concat(test_stone_dfs, ignore_index=True)
        # Average per stone across folds
        test_stone_agg = all_test_stones.groupby("stone_id").agg(
            label=("label", "first"),
            prob_mixed=("prob_mixed", "mean"),
        ).reset_index()
        test_stone_agg["pred"] = (test_stone_agg["prob_mixed"] >= 0.5).astype(int)
        test_stone_agg.to_csv(logs_dir / f"eval_test_stone_predictions{suffix}.csv", index=False)
        log.info(f"Saved: {logs_dir / f'eval_test_stone_predictions{suffix}.csv'}")

        if len(test_metrics_all) > 1:
            numeric_cols = [c for c in test_fold_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_fold_df[numeric_cols].mean().rename("mean"),
                test_fold_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / f"eval_test_summary{suffix}.csv")
            log.info(f"Saved: {logs_dir / f'eval_test_summary{suffix}.csv'}")
            log.info("Test set results (mean ± std across folds):")
            for metric in ["stone_accuracy", "stone_f1", "stone_roc_auc"]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.3f} ± "
                        f"{test_summary.at[metric, 'std']:.3f}"
                    )

        # Confusion matrix on aggregated test predictions
        cm_test = confusion_matrix(test_stone_agg["label"], test_stone_agg["pred"])
        plot_confusion_matrix(
            cm_test,
            class_names=["pure", "mixed"],
            output_path=figures_dir / f"confusion_matrix_test{suffix}.png",
            title=f"Confusion matrix — held-out test set{title_tag}",
        )

        # ROC curve for test set
        if test_roc:
            fprs_t, tprs_t, aucs_t = zip(*test_roc)
            plot_roc_curves(
                list(fprs_t), list(tprs_t), list(aucs_t),
                output_path=figures_dir / f"roc_curve_test{suffix}.png",
                title=f"ROC curve — stone-level (held-out test set){title_tag}",
            )

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
