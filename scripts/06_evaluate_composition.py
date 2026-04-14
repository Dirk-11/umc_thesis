"""
06_evaluate_moe.py — Load best MoE checkpoints and produce final evaluation outputs.

For each fold:
  - Loads best_moe.pt saved by 05_train_moe.py
  - Runs inference on that fold's validation set
  - Aggregates per-image → per-stone predictions (mean composition)
  - Records image-level and stone-level metrics

Across all folds:
  - Cross-fold mean ± std table (CSV + console)
  - Per-class MAE bar chart
  - Predicted vs true composition scatter plots (stone-level)
  - Primary-component confusion matrix (argmax of predicted vs true)
  - Per-stone predictions CSV

Also evaluates the held-out test set (same split as training).

Outputs (all under outputs/):
  figures/moe_mae_bars.png
  figures/moe_composition_scatter.png
  figures/moe_primary_confusion.png
  figures/moe_mae_bars_test.png
  figures/moe_primary_confusion_test.png
  logs/eval_moe_fold_summaries.csv
  logs/eval_moe_cv_summary.csv
  logs/eval_moe_stone_predictions.csv
  logs/eval_moe_test_fold_summaries.csv
  logs/eval_moe_test_summary.csv
  logs/eval_moe_test_stone_predictions.csv

Usage:
  python scripts/06_evaluate_moe.py              # all folds
  python scripts/06_evaluate_moe.py --fold 0     # single fold only

Run 05_train_moe.py first to produce checkpoints.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("evaluate_moe")


# -----------------------------------------------------------------------------
# Metrics (mirrors 05_train_moe.py)
# -----------------------------------------------------------------------------
def aitchison_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    clr_p = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
    clr_q = np.log(q) - np.log(q).mean(axis=1, keepdims=True)
    return float(np.sqrt(((clr_p - clr_q) ** 2).sum(axis=1)).mean())


def compute_moe_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    class_names: list[str],
    prefix: str = "",
) -> dict:
    mae_per_class = np.abs(pred - true).mean(axis=0)
    metrics = {f"{prefix}mae_overall": float(mae_per_class.mean())}
    for cls, mae in zip(class_names, mae_per_class):
        metrics[f"{prefix}mae_{cls}"] = float(mae)
    metrics[f"{prefix}dominant_acc"] = float(
        (pred.argmax(axis=1) == true.argmax(axis=1)).mean()
    )
    metrics[f"{prefix}aitchison"] = aitchison_distance(pred, true)
    return metrics


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference_moe(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
) -> pd.DataFrame:
    """Run MoE model over a DataLoader; return per-image DataFrame.

    Columns: stone_id, label, pred_{cls}, true_{cls} for each class.
    """
    model.eval()
    rows: list[dict] = []

    for images, comp_targets, stone_ids, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        true_comp = comp_targets.numpy()

        for i, stone_id in enumerate(stone_ids):
            row: dict = {
                "stone_id": stone_id,
                "label": labels[i].item() if hasattr(labels[i], "item") else int(labels[i]),
            }
            for j, cls in enumerate(class_names):
                row[f"pred_{cls}"] = float(probs[i, j])
                row[f"true_{cls}"] = float(true_comp[i, j])
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_stone_moe(img_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Mean-aggregate per-image predictions to stone level, then re-normalise."""
    pred_cols = [f"pred_{c}" for c in class_names]
    true_cols = [f"true_{c}" for c in class_names]
    stone_df = img_df.groupby("stone_id")[["label"] + pred_cols + true_cols].agg(
        {**{"label": "first"}, **{c: "mean" for c in pred_cols + true_cols}}
    ).reset_index()
    # Re-normalise predictions to enforce simplex
    pred_sum = stone_df[pred_cols].sum(axis=1)
    stone_df[pred_cols] = stone_df[pred_cols].div(pred_sum, axis=0)
    return stone_df


# -----------------------------------------------------------------------------
# Per-fold evaluation
# -----------------------------------------------------------------------------
def evaluate_fold_moe(
    fold_i: int,
    samples: list,
    idx: list[int],
    cfg: dict,
    device: torch.device,
    fold_dir: Path,
    class_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Load best_moe.pt and evaluate on the given sample indices."""
    ckpt_path = fold_dir / "best_moe.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No MoE checkpoint for fold {fold_i} at {ckpt_path}. "
            f"Run 05_train_moe.py first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_module = import_module("04_model_composition")
    model = model_module.build_moe_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    log.info(
        f"Fold {fold_i}: loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_metrics'].get('loss', float('nan')):.4f})"
    )

    ds_module = import_module("03_dataset")
    loader = ds_module.make_compositional_dataloader(samples, idx, cfg, train=False)

    img_df = run_inference_moe(model, loader, device, class_names)
    img_df["fold"] = fold_i

    stone_df = aggregate_stone_moe(img_df, class_names)
    stone_df["fold"] = fold_i

    pred_cols = [f"pred_{c}" for c in class_names]
    true_cols = [f"true_{c}" for c in class_names]

    img_metrics = compute_moe_metrics(
        img_df[pred_cols].values, img_df[true_cols].values, class_names, prefix="img_"
    )
    stone_metrics = compute_moe_metrics(
        stone_df[pred_cols].values, stone_df[true_cols].values, class_names, prefix="stone_"
    )
    metrics = {"fold": fold_i, **img_metrics, **stone_metrics}

    log.info(
        f"  Stone-level — mae={stone_metrics['stone_mae_overall']:.4f}  "
        f"dom_acc={stone_metrics['stone_dominant_acc']:.3f}  "
        f"aitchison={stone_metrics['stone_aitchison']:.4f}"
    )
    return metrics, stone_df


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_mae_bars(
    summary_df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    title: str = "Per-class MAE — stone level (cross-validation)",
) -> None:
    """Bar chart of per-class MAE with error bars (mean ± std across folds)."""
    means = [summary_df.at[f"stone_mae_{c}", "mean"] for c in class_names]
    stds  = [summary_df.at[f"stone_mae_{c}", "std"]  for c in class_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(class_names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.axhline(
        summary_df.at["stone_mae_overall", "mean"],
        color="firebrick", linestyle="--", lw=1.5,
        label=f"Overall MAE = {summary_df.at['stone_mae_overall', 'mean']:.3f}"
              f" ± {summary_df.at['stone_mae_overall', 'std']:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_composition_scatter(
    stone_df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    title: str = "Predicted vs true composition — stone level",
) -> None:
    """One scatter subplot per class: true (x) vs predicted (y) composition fraction."""
    n = len(class_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, cls in enumerate(class_names):
        ax = axes[i]
        true_vals = stone_df[f"true_{cls}"].values
        pred_vals = stone_df[f"pred_{cls}"].values
        mae = float(np.abs(pred_vals - true_vals).mean())

        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20, color="steelblue")
        lim = max(true_vals.max(), pred_vals.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("True", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{cls}  (MAE={mae:.3f})", fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_primary_confusion(
    stone_df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    title: str = "Primary component — predicted vs true (stone level)",
) -> None:
    """Confusion matrix where each cell is argmax(predicted) vs argmax(true).

    Rows = true primary component, columns = predicted primary component.
    Both raw counts and row-normalised accuracy are shown.
    """
    from sklearn.metrics import confusion_matrix

    pred_cols = [f"pred_{c}" for c in class_names]
    true_cols = [f"true_{c}" for c in class_names]

    pred_primary = np.array(class_names)[stone_df[pred_cols].values.argmax(axis=1)]
    true_primary = np.array(class_names)[stone_df[true_cols].values.argmax(axis=1)]

    cm = confusion_matrix(true_primary, pred_primary, labels=class_names)
    # Row-normalise (true counts may be zero for some classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True), 0.0)

    overall_acc = float((pred_primary == true_primary).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw counts", "Row-normalised (recall per class)"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel("Predicted primary", fontsize=11)
        ax.set_ylabel("True primary", fontsize=11)
        ax.set_title(subtitle, fontsize=11)
        for r in range(len(class_names)):
            for c in range(len(class_names)):
                val = data[r, c]
                text = format(int(val), fmt) if fmt == "d" else format(val, fmt)
                ax.text(c, r, text, ha="center", va="center",
                        color="white" if val > data.max() * 0.6 else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}\nOverall primary-component accuracy = {overall_acc:.3f}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")
    log.info(f"  Primary-component accuracy: {overall_acc:.3f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Evaluate only this fold")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = cfg["training_moe"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    ds_module     = import_module("03_dataset")
    images_csv    = resolve_path(cfg, "processed_images_csv")
    final_classes = cfg["class_remapping"]["final_classes"]

    all_samples = ds_module.load_compositional_samples(
        images_csv, final_classes, require_files_exist=True
    )

    # Reconstruct the same holdout used during training (same seed = same stones)
    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds_module.make_test_holdout(all_samples, test_fraction, seed)
    train_val_samples = [all_samples[i] for i in train_val_idx]

    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
    )

    ckpt_moe_dir = resolve_path(cfg, "checkpoints_moe_dir")
    figures_dir  = ensure_dir(resolve_path(cfg, "figures_dir"))
    logs_dir     = ensure_dir(resolve_path(cfg, "logs_dir"))

    fold_range = [args.fold] if args.fold is not None else range(len(folds))

    # -------------------------------------------------------------------------
    # Validation fold evaluation
    # -------------------------------------------------------------------------
    all_metrics:    list[dict]          = []
    all_stone_dfs:  list[pd.DataFrame]  = []

    for fold_i in fold_range:
        _, val_idx = folds[fold_i]
        fold_dir = ckpt_moe_dir / f"fold_{fold_i}"
        metrics, stone_df = evaluate_fold_moe(
            fold_i, train_val_samples, val_idx, cfg, device, fold_dir, final_classes
        )
        all_metrics.append(metrics)
        all_stone_dfs.append(stone_df)

    all_stones = pd.concat(all_stone_dfs, ignore_index=True)
    all_stones.to_csv(logs_dir / "eval_moe_stone_predictions.csv", index=False)
    log.info(f"Saved: {logs_dir / 'eval_moe_stone_predictions.csv'}")

    fold_df = pd.DataFrame(all_metrics)
    fold_df.to_csv(logs_dir / "eval_moe_fold_summaries.csv", index=False)
    log.info(f"Saved: {logs_dir / 'eval_moe_fold_summaries.csv'}")

    cv_summary = None
    if len(all_metrics) > 1:
        numeric_cols = [c for c in fold_df.columns if c != "fold"]
        cv_summary = pd.concat([
            fold_df[numeric_cols].mean().rename("mean"),
            fold_df[numeric_cols].std().rename("std"),
        ], axis=1)
        cv_summary.to_csv(logs_dir / "eval_moe_cv_summary.csv")
        log.info(f"Saved: {logs_dir / 'eval_moe_cv_summary.csv'}")
        log.info("Cross-fold validation results:")
        for metric in ["stone_mae_overall", "stone_dominant_acc", "stone_aitchison"]:
            if metric in cv_summary.index:
                log.info(
                    f"  {metric}: "
                    f"{cv_summary.at[metric, 'mean']:.4f} ± "
                    f"{cv_summary.at[metric, 'std']:.4f}"
                )

    # Figures — val
    if cv_summary is not None:
        plot_mae_bars(cv_summary, final_classes,
                      figures_dir / "moe_mae_bars.png")
    plot_composition_scatter(all_stones, final_classes,
                             figures_dir / "moe_composition_scatter.png")
    plot_primary_confusion(all_stones, final_classes,
                           figures_dir / "moe_primary_confusion.png")

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    test_metrics_all:  list[dict]         = []
    test_stone_dfs:    list[pd.DataFrame] = []

    for fold_i in fold_range:
        fold_dir = ckpt_moe_dir / f"fold_{fold_i}"
        t_metrics, t_stone_df = evaluate_fold_moe(
            fold_i, all_samples, test_idx, cfg, device, fold_dir, final_classes
        )
        test_metrics_all.append(t_metrics)
        test_stone_dfs.append(t_stone_df)

    if test_metrics_all:
        test_fold_df = pd.DataFrame(test_metrics_all)
        test_fold_df.to_csv(logs_dir / "eval_moe_test_fold_summaries.csv", index=False)

        # Average per-stone predictions across folds (same stones seen by all)
        pred_cols = [f"pred_{c}" for c in final_classes]
        true_cols = [f"true_{c}" for c in final_classes]
        all_test = pd.concat(test_stone_dfs, ignore_index=True)
        test_agg = all_test.groupby("stone_id")[["label"] + pred_cols + true_cols].agg(
            {**{"label": "first"}, **{c: "mean" for c in pred_cols + true_cols}}
        ).reset_index()
        pred_sum = test_agg[pred_cols].sum(axis=1)
        test_agg[pred_cols] = test_agg[pred_cols].div(pred_sum, axis=0)
        test_agg.to_csv(logs_dir / "eval_moe_test_stone_predictions.csv", index=False)
        log.info(f"Saved: {logs_dir / 'eval_moe_test_stone_predictions.csv'}")

        if len(test_metrics_all) > 1:
            numeric_cols = [c for c in test_fold_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_fold_df[numeric_cols].mean().rename("mean"),
                test_fold_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / "eval_moe_test_summary.csv")
            log.info(f"Saved: {logs_dir / 'eval_moe_test_summary.csv'}")
            log.info("Test set results (mean ± std across folds):")
            for metric in ["stone_mae_overall", "stone_dominant_acc", "stone_aitchison"]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.4f} ± "
                        f"{test_summary.at[metric, 'std']:.4f}"
                    )
            plot_mae_bars(test_summary, final_classes,
                          figures_dir / "moe_mae_bars_test.png",
                          title="Per-class MAE — stone level (held-out test set)")

        plot_composition_scatter(test_agg, final_classes,
                                 figures_dir / "moe_composition_scatter_test.png",
                                 title="Predicted vs true composition — held-out test set")
        plot_primary_confusion(test_agg, final_classes,
                               figures_dir / "moe_primary_confusion_test.png",
                               title="Primary component — predicted vs true (held-out test set)")

    log.info("MoE evaluation complete.")


if __name__ == "__main__":
    main()
