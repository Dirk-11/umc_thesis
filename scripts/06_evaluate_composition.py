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
import dataclasses
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import PALETTE, ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("evaluate_composition")


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
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference_moe(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
) -> pd.DataFrame:
    """Run model over a DataLoader; return per-image DataFrame.

    Columns: stone_id, label, pred_{cls}, true_{cls}, pres_{cls} for each class.
    pred_{cls}  — softmax composition fraction from composition head
    true_{cls}  — ground-truth composition fraction
    pres_{cls}  — sigmoid presence probability from presence head
    """
    model.eval()
    rows: list[dict] = []

    for images, comp_targets, stone_ids, labels in loader:
        images = images.to(device, non_blocking=True)
        comp_logits, pres_logits = model(images)
        probs      = F.softmax(comp_logits, dim=1).cpu().numpy()
        pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
        true_comp  = comp_targets.numpy()

        for i, stone_id in enumerate(stone_ids):
            row: dict = {
                "stone_id": stone_id,
                "label": labels[i].item() if hasattr(labels[i], "item") else int(labels[i]),
            }
            for j, cls in enumerate(class_names):
                row[f"pred_{cls}"] = float(probs[i, j])
                row[f"true_{cls}"] = float(true_comp[i, j])
                row[f"pres_{cls}"] = float(pres_probs[i, j])
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_stone_moe(img_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Mean-aggregate per-image predictions to stone level, then re-normalise."""
    pred_cols = [f"pred_{c}" for c in class_names]
    true_cols = [f"true_{c}" for c in class_names]
    pres_cols = [f"pres_{c}" for c in class_names]
    stone_df = img_df.groupby("stone_id")[["label"] + pred_cols + true_cols + pres_cols].agg(
        {**{"label": "first"}, **{c: "mean" for c in pred_cols + true_cols + pres_cols}}
    ).reset_index()
    # Re-normalise composition predictions to enforce simplex
    pred_sum = stone_df[pred_cols].sum(axis=1)
    stone_df[pred_cols] = stone_df[pred_cols].div(pred_sum, axis=0)
    return stone_df


# -----------------------------------------------------------------------------
# Detection and quantification metrics
# -----------------------------------------------------------------------------
def compute_detection_metrics(
    stone_df: pd.DataFrame,
    class_names: list[str],
    presence_threshold: float = 0.05,
    detection_threshold: float = 0.5,
) -> dict:
    """Precision, recall, F1 per class from the presence head.

    Ground-truth presence: true_fraction > presence_threshold
    Predicted presence:    pres_prob > detection_threshold

    Returns per-class and macro-averaged metrics prefixed with 'det_'.
    """
    from sklearn.metrics import precision_recall_fscore_support

    true_mat = (
        np.stack([stone_df[f"true_{c}"].values for c in class_names], axis=1)
        > presence_threshold
    ).astype(int)  # (N, n_classes)
    pred_mat = (
        np.stack([stone_df[f"pres_{c}"].values for c in class_names], axis=1)
        > detection_threshold
    ).astype(int)

    metrics: dict = {}
    for i, cls in enumerate(class_names):
        p, r, f, _ = precision_recall_fscore_support(
            true_mat[:, i], pred_mat[:, i], average="binary", zero_division=0
        )
        metrics[f"det_precision_{cls}"] = float(p)
        metrics[f"det_recall_{cls}"]    = float(r)
        metrics[f"det_f1_{cls}"]        = float(f)

    p, r, f, _ = precision_recall_fscore_support(
        true_mat, pred_mat, average="macro", zero_division=0
    )
    metrics["det_precision_macro"] = float(p)
    metrics["det_recall_macro"]    = float(r)
    metrics["det_f1_macro"]        = float(f)
    return metrics


def compute_quantification_mae(
    stone_df: pd.DataFrame,
    class_names: list[str],
    presence_threshold: float = 0.05,
) -> dict:
    """MAE computed only over stones where the class is actually present.

    This removes the zero-inflation bias: classes absent from a stone
    (true_fraction = 0) are excluded from the average, so the metric
    reflects how accurately the model estimates fractions for minerals
    that genuinely exist in the stone.

    Returns per-class MAE (with support count) and overall mean,
    prefixed with 'quant_'.
    """
    metrics: dict = {}
    all_errors: list[float] = []

    for cls in class_names:
        true_vals = stone_df[f"true_{cls}"].values
        pred_vals = stone_df[f"pred_{cls}"].values
        mask = true_vals > presence_threshold
        n = int(mask.sum())
        if n > 0:
            errors = np.abs(pred_vals[mask] - true_vals[mask])
            mae = float(errors.mean())
            all_errors.extend(errors.tolist())
        else:
            mae = float("nan")
        metrics[f"quant_mae_{cls}"] = mae
        metrics[f"quant_n_{cls}"]   = n

    metrics["quant_mae_overall"] = float(np.mean(all_errors)) if all_errors else float("nan")
    return metrics


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
    model_type: str = "simple",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load best checkpoint and evaluate on the given sample indices."""
    ckpt_path = fold_dir / f"best_{model_type}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for fold {fold_i} at {ckpt_path}. "
            f"Run 05_train_composition.py --model {model_type} first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_module = import_module("04_model_composition")
    if model_type.split("_")[0] == "simple":
        model = model_module.build_simple_composition_model(cfg).to(device)
    else:
        model = model_module.build_composition_model(cfg).to(device)
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

    presence_threshold = cfg["training_composition"]["presence_threshold"]

    img_metrics = compute_moe_metrics(
        img_df[pred_cols].values, img_df[true_cols].values, class_names, prefix="img_"
    )
    stone_metrics = compute_moe_metrics(
        stone_df[pred_cols].values, stone_df[true_cols].values, class_names, prefix="stone_"
    )
    det_metrics   = compute_detection_metrics(stone_df, class_names, presence_threshold)
    quant_metrics = compute_quantification_mae(stone_df, class_names, presence_threshold)

    metrics = {"fold": fold_i, **img_metrics, **stone_metrics, **det_metrics, **quant_metrics}

    log.info(
        f"  Stone-level — mae={stone_metrics['stone_mae_overall']:.4f}  "
        f"dom_acc={stone_metrics['stone_dominant_acc']:.3f}  "
        f"aitchison={stone_metrics['stone_aitchison']:.4f}"
    )
    log.info(
        f"  Detection (macro) — precision={det_metrics['det_precision_macro']:.3f}  "
        f"recall={det_metrics['det_recall_macro']:.3f}  "
        f"f1={det_metrics['det_f1_macro']:.3f}"
    )
    log.info(
        f"  Quantification MAE (present only) — "
        f"overall={quant_metrics['quant_mae_overall']:.4f}  "
        + "  ".join(
            f"{c}={quant_metrics[f'quant_mae_{c}']:.4f}(n={quant_metrics[f'quant_n_{c}']})"
            for c in class_names
        )
    )
    return metrics, stone_df, img_df


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_training_curves(
    history_dfs: list[pd.DataFrame],
    output_path: Path,
    title: str = "Training and validation loss — Model B (composition)",
) -> None:
    """Two-panel training curves for the composition model.

    Top panel:    total training loss vs val loss (KL divergence).
    Bottom panel: val stone-level MAE and val dominant-component accuracy.

    The KL loss alone is hard to interpret, so showing MAE alongside it
    gives the reader a metric they can intuitively understand.
    Shaded bands = ± 1 std across folds.  Dashed line = backbone unfrozen.
    """
    phase_boundary: int | None = None
    for df in history_dfs:
        ft = df[df["phase"] == "finetune"]
        if not ft.empty:
            b = int(ft["epoch"].min())
            if phase_boundary is None or b < phase_boundary:
                phase_boundary = b

    max_epoch = max(int(df["epoch"].max()) for df in history_dfs)

    # Collect per-epoch values across folds for all four series
    cols = ["train_total_loss", "val_loss", "val_stone_mae_overall", "val_stone_dominant_acc"]
    by_epoch: dict[str, dict[int, list[float]]] = {
        c: {e: [] for e in range(max_epoch + 1)} for c in cols
    }
    for df in history_dfs:
        for _, row in df.iterrows():
            e = int(row["epoch"])
            for c in cols:
                if c in df.columns and pd.notna(row.get(c)):
                    by_epoch[c][e].append(float(row[c]))

    def _agg(col: str) -> tuple[list[int], np.ndarray, np.ndarray]:
        epochs = [e for e in range(max_epoch + 1) if by_epoch[col][e]]
        mean   = np.array([np.mean(by_epoch[col][e]) for e in epochs])
        std    = np.array([np.std(by_epoch[col][e])  for e in epochs])
        return epochs, mean, std

    fig, (ax_loss, ax_perf) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_loss.tick_params(labelbottom=True)

    # --- Top: loss ---
    for col, label, color in [
        ("train_total_loss", "Train loss (total)", PALETTE["train"]),
        ("val_loss",         "Val loss (KL)",       PALETTE["val"]),
    ]:
        epochs, mean, std = _agg(col)
        ax_loss.plot(epochs, mean, color=color, lw=2, label=label)
        ax_loss.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
    if phase_boundary is not None:
        ax_loss.axvline(phase_boundary - 0.5, color="gray", linestyle="--",
                        lw=1.5, label="Backbone unfrozen")
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.legend(fontsize=10)
    ax_loss.set_title(title, fontsize=13)

    # --- Bottom: interpretable metrics ---
    ax_mae = ax_perf
    ax_acc = ax_mae.twinx()

    ep_mae, mean_mae, std_mae = _agg("val_stone_mae_overall")
    ax_mae.plot(ep_mae, mean_mae, color="darkorange", lw=2, label="Val stone MAE")
    ax_mae.fill_between(ep_mae, mean_mae - std_mae, mean_mae + std_mae,
                         alpha=0.2, color="darkorange")
    ax_mae.set_ylabel("Stone MAE (↓ better)", fontsize=12, color="darkorange")
    ax_mae.tick_params(axis="y", labelcolor="darkorange")

    ep_acc, mean_acc, std_acc = _agg("val_stone_dominant_acc")
    ax_acc.plot(ep_acc, mean_acc, color="seagreen", lw=2, linestyle="--",
                label="Val dominant acc")
    ax_acc.fill_between(ep_acc, mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.15, color="seagreen")
    ax_acc.set_ylabel("Dominant-class accuracy (↑ better)", fontsize=12, color="seagreen")
    ax_acc.tick_params(axis="y", labelcolor="seagreen")

    if phase_boundary is not None:
        ax_mae.axvline(phase_boundary - 0.5, color="gray", linestyle="--", lw=1.5)

    ax_mae.set_xlabel("Epoch", fontsize=12)

    # Combined legend for bottom panel
    lines_mae, labels_mae = ax_mae.get_legend_handles_labels()
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_mae.legend(lines_mae + lines_acc, labels_mae + labels_acc, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


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
    from scipy.stats import spearmanr

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
        rho = spearmanr(true_vals, pred_vals).statistic if len(true_vals) > 2 else float("nan")

        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20, color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("True", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{cls}  (MAE={mae:.3f}  ρ={rho:.2f})", fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_composition_error_violin(
    stone_df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    title: str = "Prediction error per class — stone level",
) -> None:
    """Violin plot of (predicted − true) per class, centered at 0."""
    from scipy.stats import spearmanr

    colors = ["#2166AC", "#D6604D", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628"]

    errors = []
    stats = []
    for cls in class_names:
        true_vals = stone_df[f"true_{cls}"].values
        pred_vals = stone_df[f"pred_{cls}"].values
        err = pred_vals - true_vals
        errors.append(err)
        mae = float(np.abs(err).mean())
        rho = spearmanr(true_vals, pred_vals).statistic if len(true_vals) > 2 else float("nan")
        stats.append((mae, rho))

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(errors, positions=range(len(class_names)),
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color("black")
        parts[part].set_linewidth(1)

    for i, (mae, rho) in enumerate(stats):
        ax.text(i, 0.88, f"MAE={mae:.3f}\nρ={rho:.2f}",
                ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel("Predicted − True", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-1, 1)
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
    parser.add_argument("--model", choices=["parallel", "simple"], default="simple",
                        help="simple = single shared head (default), parallel = expert heads")
    parser.add_argument("--random-labels", action="store_true",
                        help="Load checkpoints trained with --random-labels. Checkpoint names "
                             "and all output files use the '_random' suffix automatically.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag — must match the --tag used during training. "
                             "Points evaluation at the tagged checkpoint directory and adds "
                             "the tag to all output filenames.")
    args = parser.parse_args()
    model_type = args.model
    # model_type_tag drives all file naming; random runs get a distinct tag
    model_type_tag = f"{model_type}_random" if args.random_labels else model_type
    if args.tag:
        model_type_tag = f"{model_type_tag}_{args.tag}"
    title_tag = " — random-label baseline" if args.random_labels else ""
    if args.tag:
        title_tag += f" — {args.tag}"

    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = cfg["training_composition"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}  |  model: {model_type_tag}")

    ds_module     = import_module("03_dataset")
    images_csv    = resolve_path(cfg, "processed_images_csv")
    final_classes = cfg["class_remapping"]["final_classes"]

    all_samples = ds_module.load_compositional_samples(
        images_csv, final_classes, require_files_exist=True
    )

    # Drop excluded classes — must mirror 05_train_composition.py exactly so
    # the held-out splits reconstruct identically
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    if exclude:
        excl_idx = [i for i, c in enumerate(final_classes) if c in exclude]
        keep_idx = [i for i, c in enumerate(final_classes) if c not in exclude]
        all_samples = [
            s for s in all_samples
            if int(np.argmax(s.composition)) not in excl_idx
        ]
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

    test_fraction = cfg["cv"]["test_fraction"]
    all_dominant = [int(np.argmax(s.composition)) for s in all_samples]
    train_val_idx, test_idx = ds_module.make_test_holdout(
        all_samples, test_fraction, seed, stratify_labels=all_dominant
    )
    train_val_samples = [all_samples[i] for i in train_val_idx]
    train_val_dominant = [all_dominant[i] for i in train_val_idx]

    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
        stratify_labels=train_val_dominant,
    )

    ckpt_key  = "checkpoints_composition_dir"
    ckpt_dir  = resolve_path(cfg, ckpt_key)
    if args.tag:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + f"_{args.tag}")
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))
    logs_dir    = ensure_dir(resolve_path(cfg, "logs_dir"))
    p = model_type_tag  # short alias for filenames (includes _random when applicable)

    fold_range = [args.fold] if args.fold is not None else range(len(folds))

    # -------------------------------------------------------------------------
    # Training curves (reads history_{model_type_tag}.csv from training)
    # -------------------------------------------------------------------------
    history_dfs = []
    for fold_i in fold_range:
        hist_path = ckpt_dir / f"fold_{fold_i}" / f"history_{model_type_tag}.csv"
        if hist_path.exists():
            history_dfs.append(pd.read_csv(hist_path))
    if history_dfs:
        plot_training_curves(
            history_dfs,
            output_path=figures_dir / f"{p}_training_curves.png",
            title=f"Training and validation loss — Model B ({model_type_tag}){title_tag}",
        )
    else:
        log.warning(f"No history_{model_type_tag}.csv files found — skipping training curves plot")

    # -------------------------------------------------------------------------
    # Validation fold evaluation
    # -------------------------------------------------------------------------
    all_metrics:   list[dict]         = []
    all_stone_dfs: list[pd.DataFrame] = []
    all_img_dfs:   list[pd.DataFrame] = []

    for fold_i in fold_range:
        _, val_idx = folds[fold_i]
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        metrics, stone_df, img_df = evaluate_fold_moe(
            fold_i, train_val_samples, val_idx, cfg, device, fold_dir,
            final_classes, model_type=model_type_tag,
        )
        all_metrics.append(metrics)
        all_stone_dfs.append(stone_df)
        all_img_dfs.append(img_df)

    all_stones = pd.concat(all_stone_dfs, ignore_index=True)
    all_stones.to_csv(logs_dir / f"eval_{p}_stone_predictions.csv", index=False)
    log.info(f"Saved: {logs_dir / f'eval_{p}_stone_predictions.csv'}")

    fold_df = pd.DataFrame(all_metrics)
    fold_df.to_csv(logs_dir / f"eval_{p}_fold_summaries.csv", index=False)
    log.info(f"Saved: {logs_dir / f'eval_{p}_fold_summaries.csv'}")

    cv_summary = None
    if len(all_metrics) > 1:
        numeric_cols = [c for c in fold_df.columns if c != "fold"]
        cv_summary = pd.concat([
            fold_df[numeric_cols].mean().rename("mean"),
            fold_df[numeric_cols].std().rename("std"),
        ], axis=1)
        cv_summary.to_csv(logs_dir / f"eval_{p}_cv_summary.csv")
        log.info(f"Saved: {logs_dir / f'eval_{p}_cv_summary.csv'}")
        log.info("Cross-fold validation results:")
        for metric in [
            "stone_mae_overall", "stone_dominant_acc", "stone_dominant_macro_f1",
            "stone_aitchison", "det_f1_macro", "det_precision_macro",
            "det_recall_macro", "quant_mae_overall",
        ]:
            if metric in cv_summary.index:
                log.info(
                    f"  {metric}: "
                    f"{cv_summary.at[metric, 'mean']:.4f} ± "
                    f"{cv_summary.at[metric, 'std']:.4f}"
                )
        log.info("  Per-class F1 — dominant component (stone-level, mean ± std):")
        log.info(f"  {'Class':<10} {'F1':>6}  {'Std':>6}")
        log.info(f"  {'-'*26}")
        for c in final_classes:
            m = f"stone_dominant_f1_{c}"
            if m in cv_summary.index:
                log.info(f"  {c:<10} {cv_summary.at[m, 'mean']:>6.3f}  {cv_summary.at[m, 'std']:>6.3f}")

    # Figures — val
    if cv_summary is not None:
        plot_mae_bars(cv_summary, final_classes,
                      figures_dir / f"{p}_mae_bars.png",
                      title=f"Per-class MAE — stone level (cross-validation){title_tag}")
    plot_composition_scatter(all_stones, final_classes,
                             figures_dir / f"{p}_composition_scatter.png",
                             title=f"Predicted vs true composition — stone level{title_tag}")
    plot_composition_error_violin(all_stones, final_classes,
                             figures_dir / f"{p}_composition_error_violin.png",
                             title=f"Prediction error per class — stone level{title_tag}")
    plot_primary_confusion(all_stones, final_classes,
                           figures_dir / f"{p}_primary_confusion.png",
                           title=f"Primary component — predicted vs true (stone level){title_tag}")

    # -------------------------------------------------------------------------
    # Held-out test set evaluation
    # -------------------------------------------------------------------------
    log.info("=== Held-out test set evaluation ===")
    test_metrics_all: list[dict]         = []
    test_stone_dfs:   list[pd.DataFrame] = []

    for fold_i in fold_range:
        fold_dir = ckpt_dir / f"fold_{fold_i}"
        t_metrics, t_stone_df, _ = evaluate_fold_moe(
            fold_i, all_samples, test_idx, cfg, device, fold_dir,
            final_classes, model_type=model_type_tag,
        )
        test_metrics_all.append(t_metrics)
        test_stone_dfs.append(t_stone_df)

    if test_metrics_all:
        test_fold_df = pd.DataFrame(test_metrics_all)
        test_fold_df.to_csv(logs_dir / f"eval_{p}_test_fold_summaries.csv", index=False)

        pred_cols = [f"pred_{c}" for c in final_classes]
        true_cols = [f"true_{c}" for c in final_classes]
        pres_cols = [f"pres_{c}" for c in final_classes]
        all_test = pd.concat(test_stone_dfs, ignore_index=True)
        agg_cols = ["label"] + pred_cols + true_cols + pres_cols
        test_agg = all_test.groupby("stone_id")[agg_cols].agg(
            {**{"label": "first"}, **{c: "mean" for c in pred_cols + true_cols + pres_cols}}
        ).reset_index()
        pred_sum = test_agg[pred_cols].sum(axis=1)
        test_agg[pred_cols] = test_agg[pred_cols].div(pred_sum, axis=0)
        test_agg.to_csv(logs_dir / f"eval_{p}_test_stone_predictions.csv", index=False)
        log.info(f"Saved: {logs_dir / f'eval_{p}_test_stone_predictions.csv'}")

        if len(test_metrics_all) > 1:
            numeric_cols = [c for c in test_fold_df.columns if c != "fold"]
            test_summary = pd.concat([
                test_fold_df[numeric_cols].mean().rename("mean"),
                test_fold_df[numeric_cols].std().rename("std"),
            ], axis=1)
            test_summary.to_csv(logs_dir / f"eval_{p}_test_summary.csv")
            log.info(f"Saved: {logs_dir / f'eval_{p}_test_summary.csv'}")
            log.info("Test set results (mean ± std across folds):")
            for metric in [
                "stone_mae_overall", "stone_dominant_acc", "stone_dominant_macro_f1",
                "stone_aitchison", "det_f1_macro", "det_precision_macro",
                "det_recall_macro", "quant_mae_overall",
            ]:
                if metric in test_summary.index:
                    log.info(
                        f"  {metric}: "
                        f"{test_summary.at[metric, 'mean']:.4f} ± "
                        f"{test_summary.at[metric, 'std']:.4f}"
                    )
            log.info("  Per-class F1 — dominant component (stone-level, mean ± std):")
            log.info(f"  {'Class':<10} {'F1':>6}  {'Std':>6}")
            log.info(f"  {'-'*26}")
            for c in final_classes:
                m = f"stone_dominant_f1_{c}"
                if m in test_summary.index:
                    log.info(f"  {c:<10} {test_summary.at[m, 'mean']:>6.3f}  {test_summary.at[m, 'std']:>6.3f}")
            plot_mae_bars(test_summary, final_classes,
                          figures_dir / f"{p}_mae_bars_test.png",
                          title=f"Per-class MAE — stone level (held-out test set) [{p}]{title_tag}")

        # Presence-head masked MAE: zero out components where pres_prob < 0.5, renormalize
        pred_arr  = test_agg[pred_cols].values.copy()
        pres_arr  = test_agg[pres_cols].values
        true_arr2 = test_agg[true_cols].values
        mask = pres_arr >= 0.5
        pred_masked = pred_arr * mask
        row_sums = pred_masked.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0          # avoid div-by-zero for all-zero rows
        pred_masked = pred_masked / row_sums
        mae_raw    = float(np.abs(pred_arr  - true_arr2).mean())
        mae_masked = float(np.abs(pred_masked - true_arr2).mean())
        log.info(f"Post-processing MAE comparison (test set):")
        log.info(f"  Raw softmax output:          MAE = {mae_raw:.4f}")
        log.info(f"  Presence-head masked (≥0.5): MAE = {mae_masked:.4f}  (Δ={mae_masked-mae_raw:+.4f})")

        plot_composition_scatter(test_agg, final_classes,
                                 figures_dir / f"{p}_composition_scatter_test.png",
                                 title=f"Predicted vs true composition — held-out test set [{p}]{title_tag}")
        plot_composition_error_violin(test_agg, final_classes,
                                 figures_dir / f"{p}_composition_error_violin_test.png",
                                 title=f"Prediction error per class — held-out test set")
        plot_primary_confusion(test_agg, final_classes,
                               figures_dir / f"{p}_primary_confusion_test.png",
                               title=f"Primary component — predicted vs true (held-out test set) [{p}]{title_tag}")

    # -------------------------------------------------------------------------
    # Within-stone prediction variance (3.1)
    # -------------------------------------------------------------------------
    if all_img_dfs:
        all_imgs = pd.concat(all_img_dfs, ignore_index=True)
        pred_cols = [f"pred_{c}" for c in final_classes]
        within_std = all_imgs.groupby("stone_id")[pred_cols].std()
        within_variance = pd.DataFrame({
            "mean_std_per_class": within_std.mean(axis=1),
            **{f"std_{c}": within_std[f"pred_{c}"] for c in final_classes},
        })
        within_variance.to_csv(logs_dir / f"{p}_within_stone_variance.csv")
        log.info(
            f"Within-stone prediction std (mean across classes): "
            f"{within_variance['mean_std_per_class'].mean():.4f}"
        )
        log.info(f"Saved: {logs_dir / f'{p}_within_stone_variance.csv'}")

    # -------------------------------------------------------------------------
    # Baselines (2.1) — evaluated on test set
    # -------------------------------------------------------------------------
    if test_metrics_all:
        true_cols = [f"true_{c}" for c in final_classes]
        pred_cols = [f"pred_{c}" for c in final_classes]

        # Training composition vectors (all train_val stones)
        train_comps = np.array([s.composition for s in train_val_samples])
        mean_comp   = train_comps.mean(axis=0)
        mean_comp  /= mean_comp.sum()  # ensure sums to 1

        true_arr = test_agg[true_cols].values

        # Baseline 1: always predict the training-set mean composition
        mean_pred = np.tile(mean_comp, (len(true_arr), 1))
        b1 = compute_moe_metrics(mean_pred, true_arr, final_classes, prefix="b1_")

        # Baseline 2: argmax one-hot (perfect dominant class, no minor components)
        onehot_pred = np.zeros_like(true_arr)
        onehot_pred[np.arange(len(true_arr)), true_arr.argmax(axis=1)] = 1.0
        b2 = compute_moe_metrics(onehot_pred, true_arr, final_classes, prefix="b2_")

        baselines_df = pd.DataFrame([
            {"baseline": "dataset_mean",  **{k[3:]: v for k, v in b1.items()}},
            {"baseline": "primary_onehot", **{k[3:]: v for k, v in b2.items()}},
        ])
        baselines_df.to_csv(logs_dir / "eval_baselines.csv", index=False)
        log.info("Baselines (test set):")
        log.info(f"  Dataset mean  — MAE={b1['b1_mae_overall']:.4f}  "
                 f"dom_acc={b1['b1_dominant_acc']:.3f}  "
                 f"aitchison={b1['b1_aitchison']:.4f}")
        log.info(f"  Primary onehot — MAE={b2['b2_mae_overall']:.4f}  "
                 f"dom_acc={b2['b2_dominant_acc']:.3f}  "
                 f"aitchison={b2['b2_aitchison']:.4f}")
        log.info(f"Saved: {logs_dir / 'eval_baselines.csv'}")

    # -------------------------------------------------------------------------
    # Derived predictions + cross-model comparison (2.2, 5.2)
    # -------------------------------------------------------------------------
    if test_metrics_all:
        purity_threshold = cfg["data"]["purity_threshold"] / 100.0

        # Derive binary pure/mixed from Model B composition output
        pred_arr = test_agg[pred_cols].values
        max_frac = pred_arr.max(axis=1)
        derived_binary = (max_frac < purity_threshold).astype(int)  # 1=mixed, 0=pure

        comparison_rows = []

        # Compare with Model A binary predictions (if available)
        model_a_path = logs_dir / "eval_test_stone_predictions.csv"
        if model_a_path.exists():
            from sklearn.metrics import f1_score as _f1, accuracy_score as _acc
            model_a = pd.read_csv(model_a_path).set_index("stone_id")
            common = test_agg["stone_id"].values
            a_true  = model_a.loc[common, "label"].values
            a_pred  = model_a.loc[common, "pred"].values
            b_binary_true = test_agg.set_index("stone_id").loc[common, "label"].values

            model_a_f1   = _f1(a_true, a_pred, zero_division=0)
            derived_b_f1 = _f1(b_binary_true, derived_binary, zero_division=0)
            comparison_rows.append({
                "comparison": "binary_pure_mixed",
                "model_a_f1":         model_a_f1,
                "model_b_derived_f1": derived_b_f1,
            })
            log.info(f"Model A binary F1:          {model_a_f1:.3f}")
            log.info(f"Model B derived binary F1:  {derived_b_f1:.3f}")

        # Compare with Model C dominant-class predictions (if available)
        model_c_path = logs_dir / "eval_test_stone_predictions_multiclass.csv"
        if model_c_path.exists():
            model_c   = pd.read_csv(model_c_path).set_index("stone_id")
            common_mc = [s for s in test_agg["stone_id"].values if s in model_c.index]
            c_true     = model_c.loc[common_mc, "label"].values
            c_pred     = model_c.loc[common_mc, "pred"].values
            b_dom_true = test_agg.set_index("stone_id").loc[common_mc, true_cols].values.argmax(axis=1)
            b_dom_pred = test_agg.set_index("stone_id").loc[common_mc, pred_cols].values.argmax(axis=1)

            from sklearn.metrics import f1_score as _f1, accuracy_score as _acc
            model_c_acc   = _acc(c_true, c_pred)
            derived_b_acc = _acc(b_dom_true, b_dom_pred)
            comparison_rows.append({
                "comparison":           "dominant_component",
                "model_c_accuracy":     model_c_acc,
                "model_b_derived_acc":  derived_b_acc,
            })
            log.info(f"Model C dominant accuracy:          {model_c_acc:.3f}")
            log.info(f"Model B derived dominant accuracy:  {derived_b_acc:.3f}")

        if comparison_rows:
            pd.DataFrame(comparison_rows).to_csv(logs_dir / "model_comparison.csv", index=False)
            log.info(f"Saved: {logs_dir / 'model_comparison.csv'}")

    log.info(f"Evaluation complete ({model_type}).")


if __name__ == "__main__":
    main()
