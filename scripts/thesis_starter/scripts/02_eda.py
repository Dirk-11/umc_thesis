"""
02_eda.py — Exploratory data analysis figures.

Reads the processed CSVs from 01_data_prep.py and regenerates the figures
we produced during the data understanding phase:
  - Component distribution with purity breakdown
  - Threshold sweep (pure vs mixed balance)
  - Total presence across all positions

All figures saved to outputs/figures/.

Run:
  python scripts/02_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("eda")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


def fig_component_distribution(stones: pd.DataFrame, threshold: int, out_path: Path) -> None:
    """Component counts stacked by purity level."""
    comps = stones["primary"].value_counts().index.tolist()
    pure_100, near_pure, mixed = [], [], []
    for c in comps:
        sub = stones[stones["primary"] == c]
        p100 = (sub["primary_pct"] == 100).sum()
        pT = (sub["primary_pct"] >= threshold).sum()
        pure_100.append(p100)
        near_pure.append(pT - p100)
        mixed.append(len(sub) - pT)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(comps))
    w = 0.6
    ax.bar(x, pure_100, w, label="Pure (100%)", color="#2eab6f")
    ax.bar(x, near_pure, w, bottom=pure_100, label=f"Near-pure ({threshold}-99%)", color="#90d4b0")
    ax.bar(x, mixed, w,
           bottom=[a + b for a, b in zip(pure_100, near_pure)],
           label=f"Mixed (<{threshold}%)", color="#e87474")

    for i, c in enumerate(comps):
        total = pure_100[i] + near_pure[i] + mixed[i]
        ax.text(i, total + 1, str(total), ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("Number of stones")
    ax.set_title(f"Stone count per primary component (threshold={threshold}%)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path)
    plt.close()
    log.info(f"  Saved {out_path.name}")


def fig_threshold_sweep(stones: pd.DataFrame, current_threshold: int, out_path: Path) -> None:
    """How the pure/mixed balance changes across threshold values."""
    valid = stones[stones["primary_pct"].notna()]
    thresholds = list(range(50, 101, 5))
    pure = [(valid["primary_pct"] >= t).sum() for t in thresholds]
    mixed = [(valid["primary_pct"] < t).sum() for t in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(thresholds)), pure, color="#2eab6f", label="Pure")
    ax1.bar(range(len(thresholds)), mixed, bottom=pure, color="#e87474", label="Mixed")
    ax1.set_xticks(range(len(thresholds)))
    ax1.set_xticklabels([f">={t}%" for t in thresholds], rotation=45, ha="right")
    ax1.set_ylabel("Number of stones")
    ax1.set_title("Pure vs mixed at different thresholds")
    ax1.legend(framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ratios = [p / (p + m) if (p + m) > 0 else 0 for p, m in zip(pure, mixed)]
    ax2.plot(thresholds, ratios, "o-", color="#3266ad", linewidth=2, markersize=6)
    ax2.axhline(y=0.5, color="#999", linestyle="--", linewidth=1, label="Balanced (50:50)")
    ax2.axvline(x=current_threshold, color="#e87474", linestyle=":",
                linewidth=1.5, label=f"Current threshold ({current_threshold}%)")
    ax2.set_xlabel("Purity threshold (%)")
    ax2.set_ylabel("Fraction labeled 'pure'")
    ax2.set_title("Class balance across thresholds")
    ax2.set_ylim(0, 1.05)
    ax2.legend(framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(f"Purity threshold analysis (N={len(valid)} stones)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()
    log.info(f"  Saved {out_path.name}")


def fig_total_presence(stones: pd.DataFrame, out_path: Path) -> None:
    """How often each component appears across all positions (not just primary)."""
    comps = sorted(set(stones["primary"].dropna()) |
                   set(stones["secondary"].dropna()) |
                   set(stones["tertiary"].dropna()))
    data = []
    for c in comps:
        p = (stones["primary"] == c).sum()
        s = (stones["secondary"] == c).sum()
        t = (stones["tertiary"] == c).sum()
        total = 0
        for _, row in stones.iterrows():
            if row["primary"] == c or row["secondary"] == c or row["tertiary"] == c:
                total += 1
        data.append({"component": c, "primary": p, "secondary": s, "tertiary": t, "total": total})

    dfp = pd.DataFrame(data).sort_values("total", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(dfp))
    w = 0.6
    ax.bar(x, dfp["primary"], w, label="As primary", color="#3266ad")
    ax.bar(x, dfp["secondary"], w, bottom=dfp["primary"], label="As secondary", color="#85B7EB")
    ax.bar(x, dfp["tertiary"], w,
           bottom=dfp["primary"] + dfp["secondary"], label="As tertiary", color="#B5D4F4")

    for i in range(len(dfp)):
        pct = dfp.loc[i, "total"] / len(stones) * 100
        ax.text(i, dfp.loc[i, "total"] + 2,
                f"{dfp.loc[i, 'total']}\n({pct:.0f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dfp["component"])
    ax.set_ylabel("Number of stones containing component")
    ax.set_title("Component presence across ALL positions")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, dfp["total"].max() + 25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path)
    plt.close()
    log.info(f"  Saved {out_path.name}")


def main() -> None:
    cfg = load_config()
    processed_csv = resolve_path(cfg, "processed_csv")
    figures_dir = ensure_dir(resolve_path(cfg, "figures_dir"))
    threshold = cfg["data"]["purity_threshold"]

    if not processed_csv.exists():
        log.error(f"Run 01_data_prep.py first — {processed_csv} not found")
        sys.exit(1)

    stones = pd.read_csv(processed_csv)
    log.info(f"Loaded {len(stones)} stones from {processed_csv.name}")

    # Dataset summary to console
    log.info("=" * 50)
    log.info("DATASET SUMMARY")
    log.info(f"Total stones: {len(stones)}")
    log.info(f"Pure:  {(stones['label'] == 'pure').sum()}")
    log.info(f"Mixed: {(stones['label'] == 'mixed').sum()}")
    log.info(f"Unlabeled: {stones['label'].isna().sum()}")
    log.info("=" * 50)

    fig_component_distribution(stones, threshold, figures_dir / "01_component_distribution.png")
    fig_threshold_sweep(stones, threshold, figures_dir / "02_threshold_sweep.png")
    fig_total_presence(stones, figures_dir / "03_total_presence.png")

    log.info("Done.")


if __name__ == "__main__":
    main()
