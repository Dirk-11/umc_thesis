"""
statistical_tests.py — Paired significance tests across CV folds.

Compares model pairs on key metrics using paired tests across the 5 folds.
With only 5 fold measurements, normality of differences is assessed with
Shapiro-Wilk; if p > 0.05 a paired t-test is used, otherwise Wilcoxon
signed-rank.

Reads fold-level metric CSVs produced by the evaluate scripts.

Usage:
  python scripts/statistical_tests.py

Outputs:
  logs/statistical_tests.csv  — one row per comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, resolve_path, setup_logging

log = setup_logging("statistical_tests")


def paired_test(
    metric_a: list[float],
    metric_b: list[float],
    metric_name: str,
    label_a: str,
    label_b: str,
) -> dict:
    """Paired comparison of a metric across k folds.

    Returns a dict with test name, statistic, p-value, and mean difference.
    """
    diffs = [a - b for a, b in zip(metric_a, metric_b)]
    mean_diff = float(np.mean(diffs))

    if len(diffs) >= 5:
        _, p_norm = shapiro(diffs)
        if p_norm > 0.05:
            stat, p_val = ttest_rel(metric_a, metric_b)
            test_name = "paired t-test"
        else:
            try:
                stat, p_val = wilcoxon(diffs)
                test_name = "Wilcoxon signed-rank"
            except ValueError:
                # All differences are zero
                stat, p_val = 0.0, 1.0
                test_name = "Wilcoxon signed-rank (zero diff)"
    else:
        try:
            stat, p_val = wilcoxon(diffs)
            test_name = "Wilcoxon signed-rank"
        except ValueError:
            stat, p_val = 0.0, 1.0
            test_name = "Wilcoxon signed-rank (zero diff)"

    result = {
        "metric":      metric_name,
        "model_a":     label_a,
        "model_b":     label_b,
        "mean_a":      float(np.mean(metric_a)),
        "mean_b":      float(np.mean(metric_b)),
        "mean_diff":   mean_diff,
        "test":        test_name,
        "statistic":   float(stat),
        "p_value":     float(p_val),
        "significant": p_val < 0.05,
    }
    log.info(
        f"  {metric_name} | {label_a} ({result['mean_a']:.4f}) vs "
        f"{label_b} ({result['mean_b']:.4f}) | "
        f"diff={mean_diff:+.4f} | {test_name}: p={p_val:.4f}"
        + (" *" if p_val < 0.05 else "")
    )
    return result


def load_fold_metric(csv_path: Path, metric_col: str) -> list[float] | None:
    """Load per-fold metric values from a fold-summaries CSV."""
    if not csv_path.exists():
        log.warning(f"  Not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if metric_col not in df.columns:
        log.warning(f"  Column '{metric_col}' not in {csv_path.name}")
        return None
    return df[metric_col].tolist()


def main() -> None:
    cfg      = load_config()
    logs_dir = resolve_path(cfg, "logs_dir")

    results: list[dict] = []

    # -------------------------------------------------------------------------
    # Model B (composition) dominant accuracy vs Model C (multiclass) accuracy
    # -------------------------------------------------------------------------
    log.info("=== Model B dominant accuracy vs Model C stone accuracy ===")
    b_dom = load_fold_metric(
        logs_dir / "eval_simple_fold_summaries.csv", "stone_dominant_acc"
    )
    c_acc = load_fold_metric(
        logs_dir / "eval_fold_summaries_multiclass.csv", "stone_accuracy"
    )
    if b_dom and c_acc and len(b_dom) == len(c_acc):
        results.append(paired_test(b_dom, c_acc,
                                   "dominant_accuracy",
                                   "Model B (composition)",
                                   "Model C (multiclass)"))

    # -------------------------------------------------------------------------
    # Model B MAE overall: CV folds
    # -------------------------------------------------------------------------
    log.info("=== Model B stone MAE (CV folds) ===")
    b_mae = load_fold_metric(
        logs_dir / "eval_simple_fold_summaries.csv", "stone_mae_overall"
    )
    if b_mae:
        log.info(f"  Model B stone MAE: {np.mean(b_mae):.4f} ± {np.std(b_mae):.4f}")

    # -------------------------------------------------------------------------
    # Model A binary F1 (CV folds)
    # -------------------------------------------------------------------------
    log.info("=== Model A binary F1 (CV folds) ===")
    a_f1 = load_fold_metric(
        logs_dir / "eval_fold_summaries.csv", "stone_f1"
    )
    if a_f1:
        log.info(f"  Model A stone F1: {np.mean(a_f1):.4f} ± {np.std(a_f1):.4f}")

    # -------------------------------------------------------------------------
    # Model B detection macro F1 vs Model C macro F1
    # -------------------------------------------------------------------------
    log.info("=== Model B detection macro F1 vs Model C macro F1 ===")
    b_det = load_fold_metric(
        logs_dir / "eval_simple_fold_summaries.csv", "det_f1_macro"
    )
    c_f1 = load_fold_metric(
        logs_dir / "eval_fold_summaries_multiclass.csv", "stone_macro_f1"
    )
    if b_det and c_f1 and len(b_det) == len(c_f1):
        results.append(paired_test(b_det, c_f1,
                                   "macro_f1",
                                   "Model B detection head",
                                   "Model C (multiclass)"))

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    if results:
        out_path = logs_dir / "statistical_tests.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        log.info(f"Saved: {out_path}")
    else:
        log.warning("No comparisons could be run — check that eval CSVs exist.")
        log.info("Run evaluate scripts first: 06_evaluate_binary.py, "
                 "06_evaluate_composition.py, 06_evaluate_multiclass.py")


if __name__ == "__main__":
    main()
