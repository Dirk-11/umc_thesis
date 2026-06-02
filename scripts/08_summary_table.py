"""
08_summary_table.py — Per-class F1 summary figure for Models B and C.

Reads test-set summary CSVs produced by the evaluation scripts and outputs
a grouped bar chart showing per-class F1 (mean ± std) for:
  - Model C (6-class primary component)
  - Model C random-label baseline
  - Model B (dominant component from composition head)

Outputs (figures/):
  f1_summary.png

Usage:
  python scripts/08_summary_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import PALETTE, ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("summary_table")


def load_f1(csv_path: Path, class_names: list[str], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Read per-class F1 mean and std from a summary CSV.

    Returns (means, stds) arrays aligned to class_names.
    Missing entries (class not in CSV) are NaN.
    """
    if not csv_path.exists():
        log.warning(f"Not found: {csv_path}")
        return np.full(len(class_names), np.nan), np.full(len(class_names), np.nan)

    df = pd.read_csv(csv_path, index_col=0)
    means, stds = [], []
    for cls in class_names:
        key = f"{prefix}{cls}"
        means.append(df.at[key, "mean"] if key in df.index else np.nan)
        stds.append(df.at[key, "std"]  if key in df.index else np.nan)
    return np.array(means), np.array(stds)


def load_macro_f1(csv_path: Path, key: str) -> tuple[float, float]:
    if not csv_path.exists():
        return np.nan, np.nan
    df = pd.read_csv(csv_path, index_col=0)
    if key not in df.index:
        return np.nan, np.nan
    return float(df.at[key, "mean"]), float(df.at[key, "std"])


def main() -> None:
    cfg = load_config()
    logs_dir    = resolve_path(cfg, "logs_dir")
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))

    exclude      = cfg["class_remapping"].get("exclude_classes", [])
    class_names  = [c for c in cfg["class_remapping"]["final_classes"] if c not in exclude]

    # ── Load per-class F1 for each model ────────────────────────────────────
    mc_means,   mc_stds   = load_f1(
        logs_dir / "eval_test_summary_multiclass.csv",
        class_names, prefix="stone_f1_",
    )
    rand_means, rand_stds = load_f1(
        logs_dir / "eval_test_summary_multiclass_random.csv",
        class_names, prefix="stone_f1_",
    )
    b_means,    b_stds    = load_f1(
        logs_dir / "eval_simple_test_summary.csv",
        class_names, prefix="stone_dominant_f1_",
    )

    mc_macro_mean,   mc_macro_std   = load_macro_f1(
        logs_dir / "eval_test_summary_multiclass.csv",        "stone_macro_f1")
    rand_macro_mean, rand_macro_std = load_macro_f1(
        logs_dir / "eval_test_summary_multiclass_random.csv", "stone_macro_f1")
    b_macro_mean,    b_macro_std    = load_macro_f1(
        logs_dir / "eval_simple_test_summary.csv",            "stone_dominant_macro_f1")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_cls    = len(class_names)
    n_groups = n_cls + 1          # per-class + macro average
    x        = np.arange(n_groups)
    width    = 0.26
    labels   = class_names + ["Macro avg"]

    # Append macro to arrays
    mc_m   = np.append(mc_means,   mc_macro_mean)
    mc_s   = np.append(mc_stds,    mc_macro_std)
    rand_m = np.append(rand_means, rand_macro_mean)
    rand_s = np.append(rand_stds,  rand_macro_std)
    b_m    = np.append(b_means,    b_macro_mean)
    b_s    = np.append(b_stds,     b_macro_std)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_mc = ax.bar(x - width, mc_m,   width, yerr=mc_s,   capsize=4,
                     color=PALETTE["pure"],    alpha=0.85, label="Multiclass model")
    bars_rand = ax.bar(x,       rand_m, width, yerr=rand_s, capsize=4,
                     color=PALETTE["neutral"], alpha=0.85, label="Multiclass model — random baseline")
    bars_b  = ax.bar(x + width, b_m,    width, yerr=b_s,    capsize=4,
                     color=PALETTE["mixed"],   alpha=0.85, label="Composition model (dominant component)")

    # Separator between per-class and macro
    ax.axvline(n_cls - 0.5, color="gray", linestyle="--", lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 — held-out test set", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = figures_dir / "f1_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
