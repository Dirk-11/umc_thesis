"""
07_compare_roc.py — Overlay mean ROC curves from multiple binary-model runs.

Reads the per-stone prediction CSVs produced by 06_evaluate_binary.py and plots
the mean ROC curve (± 1 std band across folds) for each run in one figure.

Each "run" is specified as "Label:suffix" where suffix matches the filename
suffix written by 06_evaluate_binary.py:
  ""           → eval_stone_predictions.csv         (100% threshold, no tag)
  "_t90"       → eval_stone_predictions_t90.csv     (90% threshold)
  "_frozen"    → eval_stone_predictions_frozen.csv  (frozen-backbone baseline)
  "_resnet18"  → eval_stone_predictions_resnet18.csv (tagged experiment)

Usage:
  python scripts/07_compare_roc.py \\
      --runs "100% purity:" "90% purity:_t90" "Frozen baseline:_frozen"

  python scripts/07_compare_roc.py \\
      --runs "ResNet18:_resnet18" "100% purity:" \\
      --test                      # use held-out test set instead of val set
      --output roc_comparison_custom.png

Outputs (figures/):
  roc_comparison.png  (default) — or whatever --output specifies
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("compare_roc")

# One distinct color per run (up to 6)
RUN_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]


def compute_mean_roc(
    stone_df: pd.DataFrame,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute mean ± std ROC curve from a per-stone prediction DataFrame.

    If a 'fold' column is present, computes per-fold ROC curves and returns
    mean ± std across folds (used for cross-validation val set predictions).

    If no 'fold' column, computes a single ROC curve with std=0 (used for
    the aggregated test set predictions where folds are already averaged).

    Returns:
        mean_fpr:  (n_points,) common FPR grid
        mean_tpr:  (n_points,) mean TPR
        std_tpr:   (n_points,) std of TPR (zeros when no fold column)
        mean_auc:  scalar mean AUC
        std_auc:   scalar std AUC (0.0 when no fold column)
    """
    mean_fpr = np.linspace(0, 1, n_points)

    # No fold column — single aggregated curve (e.g. test set)
    if "fold" not in stone_df.columns:
        if stone_df["label"].nunique() < 2:
            raise ValueError("Only one class present — cannot compute ROC curve.")
        fpr, tpr, _ = roc_curve(stone_df["label"], stone_df["prob_mixed"])
        auc_val = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        return mean_fpr, interp_tpr, np.zeros_like(interp_tpr), auc_val, 0.0

    interp_tprs: list[np.ndarray] = []
    fold_aucs: list[float] = []

    for fold_id, fold_df in stone_df.groupby("fold"):
        if fold_df["label"].nunique() < 2:
            log.warning(f"  Fold {fold_id}: only one class present — skipping for ROC")
            continue
        fpr, tpr, _ = roc_curve(fold_df["label"], fold_df["prob_mixed"])
        fold_aucs.append(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    if not interp_tprs:
        raise ValueError("No valid folds found — cannot compute ROC curve.")

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tprs, axis=0)

    return mean_fpr, mean_tpr, std_tpr, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def plot_comparison(
    runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float]],
    output_path: Path,
    title: str,
) -> None:
    """Plot mean ROC curves with ± 1 std bands for multiple runs.

    runs: list of (label, mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for (label, mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc), color in zip(
        runs, RUN_COLORS
    ):
        ax.plot(
            mean_fpr, mean_tpr,
            color=color, lw=2,
            label=f"{label}  (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )
        ax.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.12, color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay mean ROC curves from multiple binary-model runs."
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        metavar="LABEL:SUFFIX",
        help=(
            'Each run as "Label:suffix", e.g. "100%% purity:" "90%% purity:_t90" '
            '"Frozen baseline:_frozen". Suffix must match the suffix written by '
            "06_evaluate_binary.py (empty string for the default run)."
        ),
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Use held-out test set predictions instead of cross-validation val set.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (relative to figures/). Default: roc_comparison.png or roc_comparison_test.png",
    )
    args = parser.parse_args()

    cfg = load_config()
    logs_dir = resolve_path(cfg, "logs_dir")
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))

    csv_prefix = "eval_test_stone_predictions" if args.test else "eval_stone_predictions"
    set_label = "held-out test set" if args.test else "cross-validation"

    default_output = f"roc_comparison{'_test' if args.test else ''}.png"
    output_path = figures_dir / (args.output if args.output else default_output)

    # -------------------------------------------------------------------------
    # Parse and load each run
    # -------------------------------------------------------------------------
    run_results = []
    for run_spec in args.runs:
        if ":" not in run_spec:
            log.error(
                f"Invalid run spec '{run_spec}' — expected 'Label:suffix' "
                "(use empty suffix for default run, e.g. 'My run:')"
            )
            sys.exit(1)

        label, suffix = run_spec.split(":", 1)
        csv_path = logs_dir / f"{csv_prefix}{suffix}.csv"

        if not csv_path.exists():
            log.error(
                f"Prediction CSV not found for run '{label}': {csv_path}\n"
                f"  Run 06_evaluate_binary.py first with the matching flags."
            )
            sys.exit(1)

        stone_df = pd.read_csv(csv_path)
        required_cols = {"label", "prob_mixed"}
        if not required_cols.issubset(stone_df.columns):
            log.error(
                f"CSV {csv_path} is missing columns {required_cols - set(stone_df.columns)}"
            )
            sys.exit(1)

        n_stones = stone_df["stone_id"].nunique() if "stone_id" in stone_df.columns else len(stone_df)
        n_folds = stone_df["fold"].nunique()
        log.info(f"Run '{label}' (suffix='{suffix}'): {n_stones} stones, {n_folds} folds")

        try:
            mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc = compute_mean_roc(stone_df)
        except ValueError as e:
            log.error(f"Could not compute ROC for run '{label}': {e}")
            sys.exit(1)

        log.info(f"  AUC = {mean_auc:.3f} ± {std_auc:.3f}")
        run_results.append((label, mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc))

    if not run_results:
        log.error("No valid runs loaded — nothing to plot.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Build title
    # -------------------------------------------------------------------------
    run_labels = ", ".join(r[0] for r in run_results)
    title = f"ROC curve comparison — {set_label}\n({run_labels})"

    plot_comparison(run_results, output_path, title)
    log.info("Done.")


if __name__ == "__main__":
    main()
