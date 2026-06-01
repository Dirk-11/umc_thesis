"""
visualize_architecture.py — Draw architecture diagrams for Model A and Model B.

Outputs:
  outputs/figures/arch_model_a.png   — StoneClassifier (binary)
  outputs/figures/arch_model_b.png   — CompositionModel (parallel) + SimpleCompositionModel
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, resolve_path, ensure_dir


# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT    = "#E8F4F8"
C_BACKBONE = "#2C5F8A"
C_FEAT     = "#4A9B8E"
C_HEAD     = "#E07B39"
C_HEAD2    = "#C0392B"
C_OUTPUT   = "#8E44AD"
C_ARROW    = "#555555"
C_TEXT_LT  = "white"
C_TEXT_DK  = "#222222"
C_BORDER   = "#333333"

FONT = "DejaVu Sans"


def box(ax, x, y, w, h, label, sublabel=None, color=C_HEAD, text_color=C_TEXT_LT,
        fontsize=10, radius=0.04):
    """Draw a rounded rectangle with label (and optional sublabel)."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2, edgecolor=C_BORDER, facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + h * 0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                fontfamily=FONT, zorder=4)
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color,
                fontfamily=FONT, zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                fontfamily=FONT, zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.4, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.02, my, label, fontsize=8, color=C_ARROW,
                ha="left", va="center", fontfamily=FONT, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Model A — StoneClassifier
# ══════════════════════════════════════════════════════════════════════════════
def draw_model_a(save_path: Path):
    fig, ax = plt.subplots(figsize=(5, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    title = "Model A — Binary Classifier (StoneClassifier)"
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=11,
            fontweight="bold", fontfamily=FONT, color=C_TEXT_DK)

    cx = 0.5

    # Input
    box(ax, cx, 0.87, 0.60, 0.07, "Input image",
        sublabel="(3 × 224 × 224)", color=C_INPUT, text_color=C_TEXT_DK, fontsize=9)
    arrow(ax, cx, 0.835, cx, 0.775)

    # Backbone
    box(ax, cx, 0.74, 0.62, 0.07, "ResNet50 backbone",
        sublabel="ImageNet pretrained  •  ~23M params",
        color=C_BACKBONE, fontsize=9)
    arrow(ax, cx, 0.705, cx, 0.645)

    # Feature vector
    box(ax, cx, 0.61, 0.52, 0.06, "Feature vector",
        sublabel="2048-dim", color=C_FEAT, fontsize=9)
    arrow(ax, cx, 0.58, cx, 0.52)

    # Head
    box(ax, cx, 0.49, 0.52, 0.055,
        "Head: Dropout(0.3) → Linear(2048 → 2)",
        color=C_HEAD, fontsize=8.5)
    arrow(ax, cx, 0.462, cx, 0.40)

    # Softmax
    box(ax, cx, 0.375, 0.38, 0.045, "Softmax", color=C_OUTPUT, fontsize=9)
    arrow(ax, cx, 0.352, cx, 0.29)

    # Output
    box(ax, cx - 0.18, 0.255, 0.28, 0.06,
        "P(pure)", sublabel="class 0", color=C_OUTPUT, fontsize=9)
    box(ax, cx + 0.18, 0.255, 0.28, 0.06,
        "P(mixed)", sublabel="class 1", color=C_OUTPUT, fontsize=9)

    # fan-out arrows
    ax.annotate("", xy=(cx - 0.18, 0.285), xytext=(cx, 0.285),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2,
                                mutation_scale=12), zorder=2)
    ax.annotate("", xy=(cx + 0.18, 0.285), xytext=(cx, 0.285),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2,
                                mutation_scale=12), zorder=2)
    ax.plot([cx, cx], [0.352, 0.285], color=C_ARROW, lw=1.4, zorder=1)

    # Training notes
    note = (
        "Phase 1 (5 ep):  freeze backbone, train head\n"
        "                       lr_head = 1e-3\n"
        "Phase 2 (20 ep): unfreeze all\n"
        "                       lr_backbone = 1e-4,  lr_head = 1e-3\n"
        "Loss: CrossEntropyLoss\n"
        "Early stopping: patience = 5"
    )
    ax.text(0.5, 0.17, note, ha="center", va="top", fontsize=7.8,
            fontfamily="monospace", color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", lw=0.8))

    plt.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Model B — CompositionModel (parallel) vs SimpleCompositionModel
# ══════════════════════════════════════════════════════════════════════════════
def draw_model_b(save_path: Path):
    CLASSES = ["CaOx", "CaP", "UA", "MAP", "CYS", "OTH"]
    N = len(CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle("Model B — Compositional Regression", fontsize=13,
                 fontweight="bold", fontfamily=FONT, y=0.99)

    for ax_i, (ax, title, is_parallel) in enumerate(zip(
        axes,
        ["CompositionModel\n(parallel expert heads)", "SimpleCompositionModel\n(single shared head)"],
        [True, False],
    )):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", fontfamily=FONT,
                     color=C_TEXT_DK, pad=6)

        cx = 0.5

        # Input
        box(ax, cx, 0.945, 0.60, 0.065, "Input image",
            sublabel="(3 × 224 × 224)", color=C_INPUT, text_color=C_TEXT_DK, fontsize=8.5)
        arrow(ax, cx, 0.912, cx, 0.852)

        # Backbone
        box(ax, cx, 0.820, 0.65, 0.062, "ResNet50 backbone",
            sublabel="ImageNet pretrained  •  ~23M params",
            color=C_BACKBONE, fontsize=8.5)
        arrow(ax, cx, 0.789, cx, 0.729)

        # Feature vector
        box(ax, cx, 0.700, 0.50, 0.055, "Feature vector",
            sublabel="2048-dim", color=C_FEAT, fontsize=8.5)

        if is_parallel:
            # ── Parallel expert heads ──────────────────────────────────────
            # Fan out from feature vector to N expert heads
            head_y    = 0.565
            head_h    = 0.055
            head_w    = 0.105
            gap       = 0.005
            total_w   = N * head_w + (N - 1) * gap
            xs = [0.5 - total_w / 2 + i * (head_w + gap) + head_w / 2 for i in range(N)]

            # Horizontal line + fan
            ax.plot([xs[0], xs[-1]], [head_y + head_h / 2 + 0.02,
                                       head_y + head_h / 2 + 0.02],
                    color=C_ARROW, lw=1.2, zorder=1)
            ax.plot([cx, cx], [0.672, head_y + head_h / 2 + 0.02],
                    color=C_ARROW, lw=1.4, zorder=1)
            for x in xs:
                ax.annotate("", xy=(x, head_y + head_h / 2),
                            xytext=(x, head_y + head_h / 2 + 0.02),
                            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                            lw=1.0, mutation_scale=10), zorder=2)

            for i, (cls, x) in enumerate(zip(CLASSES, xs)):
                box(ax, x, head_y, head_w, head_h,
                    cls,
                    sublabel="Lin→ReLU\n→Drop→Lin",
                    color=C_HEAD, fontsize=6.5)

            # Concat
            concat_y = 0.465
            ax.text(cx, concat_y, "cat → (B, 6)", ha="center", va="center",
                    fontsize=7.5, color=C_TEXT_DK, fontfamily=FONT,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0",
                              edgecolor=C_HEAD, lw=1.0), zorder=4)
            for x in xs:
                arrow(ax, x, head_y - head_h / 2, x, concat_y + 0.013)

        else:
            # ── Single shared head ─────────────────────────────────────────
            head_y = 0.565
            arrow(ax, cx, 0.672, cx, head_y + 0.030)
            box(ax, cx, head_y, 0.62, 0.055,
                "Composition head: Lin(2048→128) → ReLU → Drop → Lin(128→6)",
                color=C_HEAD, fontsize=7.5)
            concat_y = 0.465
            arrow(ax, cx, head_y - 0.028, cx, concat_y + 0.013)
            ax.text(cx, concat_y, "(B, 6) logits", ha="center", va="center",
                    fontsize=7.5, color=C_TEXT_DK, fontfamily=FONT,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0",
                              edgecolor=C_HEAD, lw=1.0), zorder=4)

        # ── Softmax → composition output ──────────────────────────────────
        softmax_y = 0.395
        arrow(ax, cx, concat_y - 0.013, cx, softmax_y + 0.022)
        box(ax, cx, softmax_y, 0.36, 0.042, "Softmax (dim=1)", color=C_OUTPUT, fontsize=8)

        out_y = 0.325
        arrow(ax, cx, softmax_y - 0.021, cx, out_y + 0.022)
        box(ax, cx, out_y, 0.60, 0.042,
            "Composition probabilities  (B, 6)  Σ = 1",
            color=C_OUTPUT, fontsize=7.5)

        # ── Presence head (shared between both variants) ───────────────────
        ph_x  = cx
        ph_y0 = 0.672
        ph_y  = 0.178
        ph_w  = 0.62

        # Branch line from feature vector down along the right side
        side_x = 0.96
        ax.plot([cx, side_x], [ph_y0, ph_y0], color=C_ARROW, lw=1.2,
                ls="dashed", zorder=1)
        ax.plot([side_x, side_x], [ph_y0, ph_y + 0.025], color=C_ARROW,
                lw=1.2, ls="dashed", zorder=1)
        ax.annotate("", xy=(ph_x + ph_w / 2, ph_y + 0.012),
                    xytext=(side_x, ph_y + 0.012),
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                   lw=1.2, mutation_scale=11,
                                   linestyle="dashed"), zorder=2)

        box(ax, ph_x, ph_y, ph_w, 0.048,
            "Presence head: Lin(2048→128) → ReLU → Drop → Lin(128→6)",
            color=C_HEAD2, fontsize=7.5)

        sig_y = 0.105
        arrow(ax, cx, ph_y - 0.024, cx, sig_y + 0.022)
        box(ax, cx, sig_y, 0.36, 0.042, "Sigmoid (per class)", color="#884EA0", fontsize=8)

        pres_y = 0.038
        arrow(ax, cx, sig_y - 0.021, cx, pres_y + 0.022)
        box(ax, cx, pres_y, 0.58, 0.040,
            "Presence probabilities  (B, 6)  ∈ [0, 1]  (independent)",
            color="#884EA0", fontsize=7.5)

        # Label the dashed branch
        ax.text(side_x + 0.012, (ph_y0 + ph_y) / 2, "aux\nbranch",
                ha="left", va="center", fontsize=6.5, color=C_ARROW,
                fontfamily=FONT, style="italic")

    # ── Shared legend / training notes ────────────────────────────────────────
    note = (
        "Phase 1 (5 ep):   freeze backbone, train heads only  •  lr_head = 1e-3\n"
        "Phase 2 (25 ep):  unfreeze all  •  lr_backbone = 1e-4,  lr_head = 1e-3\n"
        "Loss:  KL-divergence (composition) + 0.3 × BCE (presence)    Early stopping: patience = 7"
    )
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8,
             fontfamily="monospace", color="#444444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                       edgecolor="#CCCCCC", lw=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = load_config()
    fig_dir = ensure_dir(resolve_path(cfg, "figures_dir"))

    draw_model_a(fig_dir / "arch_model_a.png")
    draw_model_b(fig_dir / "arch_model_b.png")
    print("Done.")


if __name__ == "__main__":
    main()
