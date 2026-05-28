"""
07_visualize.py — Grad-CAM heatmaps for model interpretability.

Grad-CAM highlights which regions of a stone photo the model attends to when
making its pure/mixed prediction. Useful for verifying the model is looking at
the stone surface rather than image artefacts (background, ruler, label).

What this script produces:
  - One figure per outcome category (TP, TN, FP, FN) showing up to N stones,
    each with their 3 photos and Grad-CAM overlays side by side.
  - A combined summary grid with one example per category.

Outputs (figures/gradcam/):
  true_positives.png   — correctly predicted mixed stones
  true_negatives.png   — correctly predicted pure stones
  false_positives.png  — pure stones predicted as mixed
  false_negatives.png  — mixed stones predicted as pure
  summary_grid.png     — one example of each outcome side by side

Usage:
  python scripts/07_visualize.py                 # fold 0, 4 examples per category
  python scripts/07_visualize.py --fold 2        # use fold 2 checkpoint
  python scripts/07_visualize.py --n 8           # 8 examples per category
  python scripts/07_visualize.py --fold 0 --n 2  # quick check

Requires: run 05_train.py and 06_evaluate.py first.
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("visualize")


# -----------------------------------------------------------------------------
# Grad-CAM implementation (no extra dependencies)
# -----------------------------------------------------------------------------
class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Registers forward/backward hooks on a target layer to capture activations
    and gradients, then combines them into a spatial heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute Grad-CAM for a single image tensor (1, C, H, W).

        Returns a (H, W) heatmap in [0, 1], upsampled to the input resolution.
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        # Forward pass with gradient tracking on the output
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool the gradients over spatial dims → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        """Clean up hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model) -> torch.nn.Module:
    """Return the last convolutional block for the configured backbone.

    This is the layer whose activations carry the richest spatial information
    before global pooling collapses it.
    """
    name = model.backbone_name
    if name in ("resnet18", "resnet50", "resnet101"):
        return model.backbone.layer4
    if name == "efficientnet_b0":
        return model.backbone.features[-1]
    if name == "densenet121":
        return model.backbone.features.denseblock4
    raise ValueError(f"No target layer defined for backbone '{name}'. Add it to get_target_layer().")


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def load_display_image(image_path: str, size: int) -> np.ndarray:
    """Load original image as a (H, W, 3) uint8 array for display."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an RGB image.

    Args:
        image:   (H, W, 3) uint8 array
        heatmap: (H, W) float array in [0, 1]
        alpha:   heatmap opacity

    Returns (H, W, 3) uint8 blended array.
    """
    colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = ((1 - alpha) * image + alpha * heatmap_rgb).astype(np.uint8)
    return blended


# -----------------------------------------------------------------------------
# Inference + Grad-CAM for a stone
# -----------------------------------------------------------------------------
def process_stone(
    stone_id: str,
    stone_images: list[dict],
    model: torch.nn.Module,
    grad_cam: GradCAM,
    eval_transform: transforms.Compose,
    display_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int, int]:
    """Run model + Grad-CAM for all images of one stone.

    Returns:
        originals:  list of (H, W, 3) original images
        overlays:   list of (H, W, 3) Grad-CAM overlay images
        probs:      list of P(mixed) per image
        true_label: stone's true label (0=pure, 1=mixed)
        stone_pred: stone-level prediction after mean-probability aggregation
    """
    originals, overlays, probs = [], [], []
    true_label = stone_images[0]["label"]

    for row in stone_images:
        orig = load_display_image(row["image_path"], display_size)

        tensor = eval_transform(Image.open(row["image_path"]).convert("RGB"))
        tensor = tensor.unsqueeze(0).to(device)

        # Grad-CAM toward the predicted class
        heatmap = grad_cam(tensor)
        prob = torch.softmax(model(tensor), dim=1)[0, 1].item()

        overlays.append(overlay_heatmap(orig, heatmap))
        originals.append(orig)
        probs.append(prob)

    mean_prob = float(np.mean(probs))
    stone_pred = int(mean_prob >= 0.5)
    return originals, overlays, probs, true_label, stone_pred


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_category_figure(
    stones: list[dict],         # list of processed stone dicts
    title: str,
    output_path: Path,
    display_size: int,
) -> None:
    """Save a figure showing original + Grad-CAM overlay for each stone.

    Layout: one row per stone, columns = [orig_A, cam_A, orig_B, cam_B, orig_C, cam_C].
    """
    n_stones = len(stones)
    n_cols = 6   # orig + cam for each of 3 photos
    n_rows = n_stones

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Photo A", "CAM A", "Photo B", "CAM B", "Photo C", "CAM C"]
    for col_i, label in enumerate(col_labels):
        axes[0, col_i].set_title(label, fontsize=9)

    for row_i, stone in enumerate(stones):
        originals = stone["originals"]
        overlays  = stone["overlays"]
        probs     = stone["probs"]
        true_lbl  = "pure" if stone["true_label"] == 0 else "mixed"
        pred_lbl  = "pure" if stone["stone_pred"] == 0 else "mixed"
        mean_prob = float(np.mean(probs))

        row_label = (
            f"{stone['stone_id']}\n"
            f"true={true_lbl}  pred={pred_lbl}\n"
            f"P(mixed)={mean_prob:.2f}"
        )
        axes[row_i, 0].set_ylabel(row_label, fontsize=7.5, rotation=0,
                                   ha="right", va="center", labelpad=60)

        for photo_i in range(min(3, len(originals))):
            col_orig = photo_i * 2
            col_cam  = photo_i * 2 + 1
            axes[row_i, col_orig].imshow(originals[photo_i])
            axes[row_i, col_cam].imshow(overlays[photo_i])
            axes[row_i, col_cam].set_xlabel(f"P={probs[photo_i]:.2f}", fontsize=7)

        for ax in axes[row_i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def save_summary_grid(
    category_examples: dict[str, dict | None],
    output_path: Path,
) -> None:
    """One example stone per outcome category, in a 2×2 grid."""
    categories = ["TP", "TN", "FP", "FN"]
    titles = {
        "TP": "True Positive\n(mixed → mixed)",
        "TN": "True Negative\n(pure → pure)",
        "FP": "False Positive\n(pure → mixed)",
        "FN": "False Negative\n(mixed → pure)",
    }
    n_cols = 6
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2.2, 4 * 2.8))

    col_labels = ["Photo A", "CAM A", "Photo B", "CAM B", "Photo C", "CAM C"]
    for col_i, label in enumerate(col_labels):
        axes[0, col_i].set_title(label, fontsize=9)

    for row_i, cat in enumerate(categories):
        stone = category_examples.get(cat)
        label = titles[cat]
        if stone is None:
            for ax in axes[row_i]:
                ax.axis("off")
            axes[row_i, 0].set_ylabel(f"{label}\n(no examples)", fontsize=8,
                                       rotation=0, ha="right", va="center", labelpad=60)
            continue

        originals = stone["originals"]
        overlays  = stone["overlays"]
        probs     = stone["probs"]
        mean_prob = float(np.mean(probs))

        axes[row_i, 0].set_ylabel(
            f"{label}\n{stone['stone_id']}  P(mixed)={mean_prob:.2f}",
            fontsize=7.5, rotation=0, ha="right", va="center", labelpad=60,
        )
        for photo_i in range(min(3, len(originals))):
            axes[row_i, photo_i * 2].imshow(originals[photo_i])
            axes[row_i, photo_i * 2 + 1].imshow(overlays[photo_i])
            axes[row_i, photo_i * 2 + 1].set_xlabel(f"P={probs[photo_i]:.2f}", fontsize=7)

        for ax in axes[row_i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Grad-CAM summary — one example per outcome", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold's checkpoint to use (default: 0)")
    parser.add_argument("--n", type=int, default=4,
                        help="Max stones per outcome category (default: 4)")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    requested = cfg["training"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    ckpt_dir = resolve_path(cfg, "checkpoints_dir")
    ckpt_path = ckpt_dir / f"fold_{args.fold}" / "best.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint at {ckpt_path}. Run 05_train.py first.")
        sys.exit(1)

    model_module = import_module("04_model_binary")
    model = model_module.build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded fold {args.fold} checkpoint (epoch {checkpoint['epoch']})")

    # -------------------------------------------------------------------------
    # Build val set for this fold
    # -------------------------------------------------------------------------
    ds_module = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    samples = ds_module.load_image_samples(images_csv, require_files_exist=True)
    folds = ds_module.make_stratified_folds(
        samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
    )
    _, val_idx = folds[args.fold]
    val_samples = [samples[i] for i in val_idx]

    # Group val images by stone
    stone_groups: dict[str, list[dict]] = {}
    for s in val_samples:
        stone_groups.setdefault(s.stone_id, []).append({
            "image_path": s.image_path,
            "label": s.label,
        })

    # -------------------------------------------------------------------------
    # Set up Grad-CAM
    # -------------------------------------------------------------------------
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    eval_transform = ds_module.build_transforms(cfg, train=False)
    display_size = cfg["image"]["input_size"]

    # -------------------------------------------------------------------------
    # Run Grad-CAM for every stone in the val set
    # -------------------------------------------------------------------------
    log.info(f"Running Grad-CAM for {len(stone_groups)} stones in fold {args.fold} val set...")
    processed: list[dict] = []
    for stone_id, images in stone_groups.items():
        originals, overlays, probs, true_label, stone_pred = process_stone(
            stone_id, images, model, grad_cam, eval_transform, display_size, device
        )
        processed.append({
            "stone_id":   stone_id,
            "true_label": true_label,
            "stone_pred": stone_pred,
            "originals":  originals,
            "overlays":   overlays,
            "probs":      probs,
        })

    grad_cam.remove()

    # -------------------------------------------------------------------------
    # Sort into outcome categories
    # -------------------------------------------------------------------------
    def category(s: dict) -> str:
        t, p = s["true_label"], s["stone_pred"]
        if   t == 1 and p == 1: return "TP"
        elif t == 0 and p == 0: return "TN"
        elif t == 0 and p == 1: return "FP"
        else:                   return "FN"

    buckets: dict[str, list[dict]] = {"TP": [], "TN": [], "FP": [], "FN": []}
    for s in processed:
        buckets[category(s)].append(s)

    for cat, stones in buckets.items():
        log.info(f"  {cat}: {len(stones)} stones")

    # -------------------------------------------------------------------------
    # Save figures
    # -------------------------------------------------------------------------
    gradcam_dir = ensure_dir(resolve_path(cfg, "figures_dir") / "gradcam")

    category_titles = {
        "TP": "True Positives — mixed stones correctly predicted as mixed",
        "TN": "True Negatives — pure stones correctly predicted as pure",
        "FP": "False Positives — pure stones incorrectly predicted as mixed",
        "FN": "False Negatives — mixed stones incorrectly predicted as pure",
    }
    category_files = {
        "TP": "true_positives.png",
        "TN": "true_negatives.png",
        "FP": "false_positives.png",
        "FN": "false_negatives.png",
    }

    summary_examples: dict[str, dict | None] = {}
    for cat, stones in buckets.items():
        subset = stones[: args.n]
        summary_examples[cat] = subset[0] if subset else None
        if subset:
            save_category_figure(
                subset,
                title=f"{category_titles[cat]} (fold {args.fold}, n={len(subset)})",
                output_path=gradcam_dir / category_files[cat],
                display_size=display_size,
            )
        else:
            log.info(f"  Skipping {cat} figure — no examples in this fold's val set")

    save_summary_grid(summary_examples, gradcam_dir / "summary_grid.png")
    log.info("Grad-CAM visualization complete.")


if __name__ == "__main__":
    main()
