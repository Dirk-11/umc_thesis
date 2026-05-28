"""
06_evaluate_multiclass.py — Evaluate Model C (6-class primary component classifier).

For each fold:
  - Loads the best checkpoint saved by 05_train_multiclass.py
  - Runs inference on that fold's validation set
  - Aggregates per-image → per-stone predictions (mean class probabilities, then argmax)
  - Records image-level and stone-level metrics

Across all folds:
  - Cross-fold mean ± std table (CSV + console)
  - Aggregated 6x6 confusion matrix (figure)
  - Per-class F1 summary

Held-out test set:
  - Each fold's checkpoint is evaluated on the same test stones
  - Probabilities are averaged across folds (ensemble) → final prediction per stone

Outputs (all under outputs/):
  figures/confusion_matrix_multiclass.png
  figures/confusion_matrix_test_multiclass.png
  logs/eval_fold_summaries_multiclass.csv
  logs/eval_cv_summary_multiclass.csv
  logs/eval_stone_predictions_multiclass.csv
  logs/eval_test_summary_multiclass.csv
  logs/eval_test_stone_predictions_multiclass.csv

Usage:
  python scripts/06_evaluate_multiclass.py
  python scripts/06_evaluate_multiclass.py --fold 0
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from utils import PALETTE, ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("evaluate_multiclass")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
) -> pd.DataFrame:
    """Run model over a DataLoader and return a per-image DataFrame."""
    model.eval()
    rows: list[dict] = []

    for images, labels, stone_ids in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu()
        preds  = logits.argmax(dim=1).cpu()

        for stone_id, label, prob_row, pred in zip(
            stone_ids, labels.tolist(), probs.tolist(), preds.tolist()
        ):
            row = {"stone_id": stone_id, "label": label, "pred": pred}
            for cls, p in zip(class_names, prob_row):
                row[f"prob_{cls}"] = p
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_to_stone_level(
    img_df: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Average class probabilities across images of the same stone, then argmax."""
    prob_cols = [f"prob_{c}" for c in class_names]
    stone_df = img_df.groupby("stone_id").agg(
        label=("label", "first"),
        **{col: (col, "mean") for col in prob_cols},
    ).reset_index()
    stone_df["pred"] = stone_df[prob_cols].values.argmax(axis=1)
    return stone_df


def compute_metrics(df: pd.DataFrame, class_names: list[str], prefix: str = "") -> dict:
    metrics: dict = {}
    metrics[f"{prefix}accuracy"] = accuracy_score(df["label"], df["pred"])
    metrics[f"{prefix}macro_f1"] = f1_score(
        df["label"], df["pred"], average="macro", zero_division=0
    )
    per_class = f1_score(
        df["label"], df["pred"],
        labels=list(range(len(class_names))),
        average=None, zero_division=0,
    )
    for cls, val in zip(class_names, per_class):
        metrics[f"{prefix}f1_{cls}"] = float(val)
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
    """Plot mean ± std train and val loss across folds with phase boundary marker."""
    phase_boundary: int | None = None
    for df in history_dfs:
        ft = df[df["phase"] == "finetune"]
        if not ft.empty:
            b = int(ft["epoch"].min())
            if phase_boundary is None or b < phase_boundary:
                phase_boundary = b

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
    title: str = "Confusion matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=10)
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
    class_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for fold {fold_i} at {ckpt_path}. "
            f"Run 05_train_multiclass.py first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_module = import_module("04_model_binary")
    mc = cfg["model_multiclass"]
    model = model_module.StoneClassifier(
        backbone_name=mc["backbone"],
        weights=mc["pretrained_weights"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    log.info(
        f"Fold {fold_i}: loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_metrics'].get('loss', '?'):.4f})"
    )

    ds_module  = import_module("03_dataset")
    val_loader = ds_module.make_multiclass_dataloader(samples, val_idx, cfg, train=False)

    img_df = run_inference(model, val_loader, device, class_names)
    img_df["fold"] = fold_i

    stone_df = aggregate_to_stone_level(img_df, class_names)
    stone_df["fold"] = fold_i

    img_metrics   = compute_metrics(img_df,   class_names, prefix="img_")
    stone_metrics = compute_metrics(stone_df, class_names, prefix="stone_")
    metrics = {"fold": fold_i, **img_metrics, **stone_metrics}

    log.info(
        f"  Stone-level — acc={stone_metrics['stone_accuracy']:.3f}  "
        f"macro_f1={stone_metrics['stone_macro_f1']:.3f}"
    )
    return metrics, stone_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--random-labels", action="store_true",
                        help="Load checkpoints from checkpoints_multiclass_random/ "
                             "(the --random-labels training run). Outputs are written "
                             "with a _random suffix.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag — must match the --tag used during training. "
                             "Points evaluation at the tagged checkpoint directory and adds "
                             "the tag to all output filenames.")
    args = parser.parse_args()

    cfg  = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = cfg["training_multiclass"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    class_names = cfg["class_remapping"]["final_classes"]

    ds_module  = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds_module.load_multiclass_samples(
        images_csv, class_names, require_files_exist=True
    )

    # Drop excluded classes — must match what 05_train_multiclass.py did so
    # the held-out splits reconstruct identically
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

    # Stratify on dominant mineral class — matches 05_train_multiclass.py
    dominant_labels = [s.label for s in all_samples]

    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds_module.make_test_holdout(
        all_samples, test_fraction, seed, stratify_labels=dominant_labels
    )
    train_val_samples = [all_samples[i] for i in train_val_idx]
    train_val_dominant = [dominant_labels[i] for i in train_val_idx]

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
    suffix    = "_random" if args.random_labels else ""
    if args.tag:
        suffix += f"_{args.tag}"
    title_tag = " — random-label baseline" if args.random_labels else ""
    if args.tag:
        title_tag += f" — {args.tag}"
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))
    logs_dir    = ensure_dir(resolve_path(cfg, "logs_dir"))

    fold_range = [args.fold] if args.fold is not None else range(len(folds))

    # -------------------------------------------------------------------------
    # Training curves (reads history.csv written by 05_train_multiclass.py)
    # -------------------------------------------------------------------------
    history_dfs = []
    for fold_i in fold_range:
        hist_path = ckpt_dir / f"fold_{fold_i}" / "history.csv"
        if hist_path.exists():
            history_dfs.append(pd.read_csv(hist_path))
    if history_dfs:
        plot_training_curves(
            history_dfs,
            output_path=figures_dir / f"training_curves_multiclass{suffix}.png",
            title=f"Training and validation loss — Model C (6-class){title_tag}",
        )
    else:
        log.warning("No history.csv files found — skipping training curves plot")

    all_metrics:    list[dict]         = []
    all_stone_dfs:  list[pd.DataFrame] = []

    for fold_i in fold_range:
        _, val_idx = folds[fold_i]
        fold_dir   = ckpt_dir / f"fold_{fold_i}"
        metrics, stone_df = evaluate_fold(
            fold_i, train_val_samples, val_idx, cfg, device, fold_dir, class_names
        )
        all_metrics.append(metrics)
        all_stone_dfs.append(stone_df)

    # -------------------------------------------------------------------------
    # Save per-stone predictions (CV validation sets)
    # -------------------------------------------------------------------------
    all_stones = pd.concat(all_stone_dfs, ignore_index=True)
    all_stones.to_csv(logs_dir / f"eval_stone_predictions_multiclass{suffix}.csv", index=False)

    fold_df = pd.DataFrame(all_metrics)
    fold_df.to_csv(logs_dir / f"eval_fold_summaries_multiclass{suffix}.csv", index=False)

    if len(all_metrics) > 1:
        numeric_cols = [c for c in fold_df.columns if c != "fold"]
        summary = pd.concat([
            fold_df[numeric_cols].mean().rename("mean"),
            fold_df[numeric_cols].std().rename("std"),
        ], axis=1)
        summary.to_csv(logs_dir / f"eval_cv_summary_multiclass{suffix}.csv")
        log.info(f"Saved: {logs_dir / f'eval_cv_summary_multiclass{suffix}.csv'}")
        log.info("Cross-fold results:")
        for metric in ["stone_accuracy", "stone_macro_f1"]:
            if metric in summary.index:
                log.info(
                    f"  {metric}: "
                    f"{summary.at[metric, 'mean']:.3f} ± {summary.at[metric, 'std']:.3f}"
                )
        log.info("  Per-class F1 (stone-level, mean ± std):")
        log.info(f"  {'Class':<10} {'F1':>6}  {'Std':>6}")
        log.info(f"  {'-'*26}")
        for c in class_names:
            m = f"stone_f1_{c}"
            if m in summary.index:
                log.info(f"  {c:<10} {summary.at[m, 'mean']:>6.3f}  {summary.at[m, 'std']:>6.3f}")

    # -------------------------------------------------------------------------
    # Aggregated confusion matrix (CV, all folds combined)
    # -------------------------------------------------------------------------
    cm = confusion_matrix(
        all_stones["label"], all_stones["pred"],
        labels=list(range(len(class_names))),
    )
    plot_confusion_matrix(
        cm, class_names,
        output_path=figures_dir / f"confusion_matrix_multiclass{suffix}.png",
        title=f"Confusion matrix — primary component (cross-validation){title_tag}",
    )

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    model_module = import_module("04_model_binary")
    test_metrics_all: list[dict]         = []
    test_stone_dfs:   list[pd.DataFrame] = []

    prob_cols = [f"prob_{c}" for c in class_names]

    for fold_i in fold_range:
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        t_metrics, t_stone_df = evaluate_fold(
            fold_i, all_samples, test_idx, cfg, device, fold_dir, class_names
        )
        test_metrics_all.append(t_metrics)
        test_stone_dfs.append(t_stone_df)

    if test_metrics_all:
        test_fold_df = pd.DataFrame(test_metrics_all)
        test_fold_df.to_csv(logs_dir / f"eval_test_fold_summaries_multiclass{suffix}.csv", index=False)

        # Average probabilities across folds per stone
        all_test_stones = pd.concat(test_stone_dfs, ignore_index=True)
        test_stone_agg = all_test_stones.groupby("stone_id").agg(
            label=("label", "first"),
            **{col: (col, "mean") for col in prob_cols},
        ).reset_index()
        test_stone_agg["pred"] = test_stone_agg[prob_cols].values.argmax(axis=1)
        test_stone_agg.to_csv(
            logs_dir / f"eval_test_stone_predictions_multiclass{suffix}.csv", index=False
        )
        log.info(f"Saved: {logs_dir / f'eval_test_stone_predictions_multiclass{suffix}.csv'}")

        if len(test_metrics_all) > 1:
            numeric_cols = [c for c in test_fold_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_fold_df[numeric_cols].mean().rename("mean"),
                test_fold_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / f"eval_test_summary_multiclass{suffix}.csv")
            log.info(f"Saved: {logs_dir / f'eval_test_summary_multiclass{suffix}.csv'}")
            log.info("Test set results (mean ± std across folds):")
            for metric in ["stone_accuracy", "stone_macro_f1"]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.3f} ± "
                        f"{test_summary.at[metric, 'std']:.3f}"
                    )
            log.info("  Per-class F1 (stone-level, mean ± std):")
            log.info(f"  {'Class':<10} {'F1':>6}  {'Std':>6}")
            log.info(f"  {'-'*26}")
            for c in class_names:
                m = f"stone_f1_{c}"
                if m in test_summary.index:
                    log.info(f"  {c:<10} {test_summary.at[m, 'mean']:>6.3f}  {test_summary.at[m, 'std']:>6.3f}")

        # Confusion matrix on ensemble test predictions
        cm_test = confusion_matrix(
            test_stone_agg["label"], test_stone_agg["pred"],
            labels=list(range(len(class_names))),
        )
        plot_confusion_matrix(
            cm_test, class_names,
            output_path=figures_dir / f"confusion_matrix_test_multiclass{suffix}.png",
            title=f"Confusion matrix — primary component (held-out test set){title_tag}",
        )

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
