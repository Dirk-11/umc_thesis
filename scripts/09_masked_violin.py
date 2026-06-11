"""
09_masked_violin.py — Regenerate the composition error violin using presence-head masking.

Loads the saved test stone predictions (eval_simple_test_stone_predictions.csv),
applies the presence-head masking step (zero out predicted fractions where
sigmoid < 0.5, then renormalise), and saves a new violin figure.

No GPU or model checkpoint needed — operates on the saved CSV only.

Usage (login node or interactive session is fine):
  python scripts/09_masked_violin.py

Output:
  outputs/figures/simple_composition_error_violin_test_masked.png
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("masked_violin")

CLASSES = ["CaOx", "CaP", "UA", "MAP", "CYS"]
PRESENCE_THRESHOLD = 0.5
COLORS = ["#2166AC", "#D6604D", "#4DAF4A", "#FF7F00", "#984EA3"]


def apply_presence_masking(
    df: pd.DataFrame,
    classes: list[str],
    threshold: float = 0.5,
) -> np.ndarray:
    pred_cols = [f"pred_{c}" for c in classes]
    pres_cols = [f"pres_{c}" for c in classes]
    pred_arr = df[pred_cols].values.copy()
    pres_arr = df[pres_cols].values
    mask = pres_arr >= threshold
    pred_masked = pred_arr * mask
    row_sums = pred_masked.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pred_masked / row_sums


def plot_masked_violin(
    pred_masked: np.ndarray,
    true_arr: np.ndarray,
    classes: list[str],
    output_path: Path,
) -> None:
    errors = []
    stats = []
    for i, cls in enumerate(classes):
        err = pred_masked[:, i] - true_arr[:, i]
        errors.append(err)
        mae = float(np.abs(err).mean())
        rho_result = spearmanr(true_arr[:, i], pred_masked[:, i])
        rho = rho_result.statistic if not np.isnan(rho_result.statistic) else float("nan")
        stats.append((mae, rho))
        log.info(f"  {cls}: MAE={mae:.3f}  rho={rho:.2f}")

    overall_mae = float(np.abs(pred_masked - true_arr).mean())
    log.info(f"  Overall MAE: {overall_mae:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(errors, positions=range(len(classes)),
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color("black")
        parts[part].set_linewidth(1)

    for i, (mae, rho) in enumerate(stats):
        rho_str = f"{rho:.2f}" if not np.isnan(rho) else "n/a"
        ax.text(i, 0.88, f"MAE={mae:.3f}\nρ={rho_str}",
                ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Predicted − True", fontsize=11)
    ax.set_title("Prediction error per class — held-out test set", fontsize=12)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def main() -> None:
    cfg = load_config()
    logs_dir    = resolve_path(cfg, "logs_dir")
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))

    csv_path = logs_dir / "eval_simple_test_stone_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}\n"
            "Run 06_evaluate_composition.py --model simple first."
        )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} stones from {csv_path}")

    true_arr    = df[[f"true_{c}" for c in CLASSES]].values
    pred_masked = apply_presence_masking(df, CLASSES, PRESENCE_THRESHOLD)

    log.info("Per-class metrics (presence-head masked predictions):")
    output_path = figures_dir / "simple_composition_error_violin_test_masked.png"
    plot_masked_violin(pred_masked, true_arr, CLASSES, output_path)


if __name__ == "__main__":
    main()
