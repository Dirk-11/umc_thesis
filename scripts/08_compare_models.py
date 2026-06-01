"""
08_compare_models.py — Side-by-side F1 comparison of Model B vs Model C.

Reads the cross-validation (or held-out test) summary CSVs produced by
06_evaluate_composition.py and 06_evaluate_multiclass.py, then produces:

  1. A CSV table with per-class and macro F1 for both models plus the diff.
  2. A grouped bar chart with error bars (± 1 std across folds).

Outputs (outputs/logs/ and outputs/figures/):
  model_bc_f1_comparison_cv.csv
  model_bc_f1_comparison_cv.png
  model_bc_f1_comparison_test.csv      (only if test summaries exist)
  model_bc_f1_comparison_test.png

Usage:
  python scripts/08_compare_models.py          # CV results (default)
  python scripts/08_compare_models.py --test   # held-out test results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import PALETTE, ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("compare_models")


def load_summary(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.error(f"{label} summary not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, index_col=0)
    log.info(f"Loaded {label}: {path}")
    return df


def build_comparison_table(
    b_summary: pd.DataFrame,
    c_summary: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Assemble a per-class + macro comparison table.

    Model B column names: stone_dominant_f1_{cls}, stone_dominant_macro_f1
    Model C column names: stone_f1_{cls}, stone_macro_f1
    """
    rows = []
    for cls in class_names:
        b_key = f"stone_dominant_f1_{cls}"
        c_key = f"stone_f1_{cls}"
        b_f1  = b_summary.at[b_key, "mean"] if b_key in b_summary.index else float("nan")
        b_std = b_summary.at[b_key, "std"]  if b_key in b_summary.index else float("nan")
        c_f1  = c_summary.at[c_key, "mean"] if c_key in c_summary.index else float("nan")
        c_std = c_summary.at[c_key, "std"]  if c_key in c_summary.index else float("nan")
        rows.append({
            "class":     cls,
            "model_b_f1":  b_f1,
            "model_b_std": b_std,
            "model_c_f1":  c_f1,
            "model_c_std": c_std,
            "diff":        b_f1 - c_f1,
        })

    # Macro row
    b_mac = b_summary.at["stone_dominant_macro_f1", "mean"] if "stone_dominant_macro_f1" in b_summary.index else float("nan")
    b_mac_std = b_summary.at["stone_dominant_macro_f1", "std"] if "stone_dominant_macro_f1" in b_summary.index else float("nan")
    c_mac = c_summary.at["stone_macro_f1", "mean"] if "stone_macro_f1" in c_summary.index else float("nan")
    c_mac_std = c_summary.at["stone_macro_f1", "std"] if "stone_macro_f1" in c_summary.index else float("nan")
    rows.append({
        "class":     "macro",
        "model_b_f1":  b_mac,
        "model_b_std": b_mac_std,
        "model_c_f1":  c_mac,
        "model_c_std": c_mac_std,
        "diff":        b_mac - c_mac,
    })

    return pd.DataFrame(rows)


def plot_comparison(
    table: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Grouped bar chart — one group per class, two bars (Model B / Model C)."""
    labels = table["class"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    b_vals = table["model_b_f1"].values
    b_errs = table["model_b_std"].values
    c_vals = table["model_c_f1"].values
    c_errs = table["model_c_std"].values

    b_mac = table.loc[table["class"] == "macro", "model_b_f1"].values
    c_mac = table.loc[table["class"] == "macro", "model_c_f1"].values

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))

    ax.bar(x - width / 2, b_vals, width, yerr=b_errs, capsize=4,
           color=PALETTE.get("val", "#377eb8"), alpha=0.85, label="Compositional regression")
    ax.bar(x + width / 2, c_vals, width, yerr=c_errs, capsize=4,
           color=PALETTE.get("train", "#e41a1c"), alpha=0.85, label="Multi-class classification")

    if len(b_mac) > 0 and not np.isnan(b_mac[0]):
        ax.axhline(b_mac[0], color=PALETTE.get("val", "#377eb8"),
                   linestyle="--", lw=1.5, alpha=0.7, label=f"Compositional regression macro = {b_mac[0]:.3f}")
    if len(c_mac) > 0 and not np.isnan(c_mac[0]):
        ax.axhline(c_mac[0], color=PALETTE.get("train", "#e41a1c"),
                   linestyle="--", lw=1.5, alpha=0.7, label=f"Multi-class classification macro = {c_mac[0]:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def run(split: str, logs_dir: Path, figures_dir: Path, class_names: list[str]) -> None:
    """Build comparison for one split (cv or test)."""
    if split == "cv":
        b_path = logs_dir / "eval_simple_cv_summary.csv"
        c_path = logs_dir / "eval_cv_summary_multiclass.csv"
        tag = "cv"
        title_suffix = "cross-validation"
    else:
        b_path = logs_dir / "eval_simple_test_summary.csv"
        c_path = logs_dir / "eval_test_summary_multiclass.csv"
        tag = "test"
        title_suffix = "held-out test set"

    if not b_path.exists() or not c_path.exists():
        missing = [p for p in [b_path, c_path] if not p.exists()]
        log.error(f"Missing summary file(s): {[str(p) for p in missing]}")
        log.error("Run 06_evaluate_composition.py and 06_evaluate_multiclass.py first.")
        sys.exit(1)

    b_summary = load_summary(b_path, "Model B")
    c_summary = load_summary(c_path, "Model C")

    table = build_comparison_table(b_summary, c_summary, class_names)

    csv_path = logs_dir / f"model_bc_f1_comparison_{tag}.csv"
    table.to_csv(csv_path, index=False)
    log.info(f"Saved: {csv_path}")

    log.info(f"F1 comparison ({title_suffix}):")
    log.info(f"  {'Class':<10} {'Model B':>8} {'±':>4} {'Model C':>8} {'±':>4} {'Diff':>7}")
    log.info(f"  {'-'*47}")
    for _, row in table.iterrows():
        log.info(
            f"  {row['class']:<10} {row['model_b_f1']:>8.3f} {row['model_b_std']:>4.3f} "
            f"{row['model_c_f1']:>8.3f} {row['model_c_std']:>4.3f} {row['diff']:>+7.3f}"
        )

    plot_comparison(
        table,
        output_path=figures_dir / f"model_bc_f1_comparison_{tag}.png",
        title=(
            f"Dominant-component F1 — Compositional regression vs "
            f"Multi-class classification\n({title_suffix})"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Use held-out test summaries instead of CV summaries")
    args = parser.parse_args()

    cfg = load_config()
    final_classes = cfg["class_remapping"]["final_classes"]
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    class_names = [c for c in final_classes if c not in exclude]

    logs_dir    = ensure_dir(resolve_path(cfg, "logs_dir"))
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))

    split = "test" if args.test else "cv"
    run(split, logs_dir, figures_dir, class_names)


if __name__ == "__main__":
    main()
