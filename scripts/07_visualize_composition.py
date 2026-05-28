"""
07_visualize_composition.py — Grad-CAM heatmaps for Model B (composition model).

Adapts the binary Grad-CAM script for SimpleCompositionModel. Backpropagation
is through the predicted dominant class (argmax of the composition output).

Produces figures grouped by prediction correctness of the dominant component:
  - Correct dominant predictions
  - Incorrect dominant predictions
  - High within-stone variance stones (model is uncertain between photos)

Outputs (figures/gradcam_composition/):
  correct_dominant.png    — stones where argmax(pred) == argmax(true)
  incorrect_dominant.png  — stones where argmax(pred) != argmax(true)
  high_variance.png       — stones with highest prediction variance across photos
  summary_grid.png        — one example per category

Usage:
  python scripts/07_visualize_composition.py                # fold 0, 4 examples
  python scripts/07_visualize_composition.py --fold 2
  python scripts/07_visualize_composition.py --n 6

Requires: run 05_train_composition.py first.
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("visualize_composition")


# -----------------------------------------------------------------------------
# Grad-CAM (identical to 07_visualize_binary.py — reused directly)
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _m, _i, output):
        self._activations = output.detach()

    def _save_gradients(self, _m, _gi, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        comp_logits, _ = self.model(image_tensor)
        if class_idx is None:
            class_idx = comp_logits.argmax(dim=1).item()
        self.model.zero_grad()
        comp_logits[0, class_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image_tensor.shape[2], image_tensor.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model) -> torch.nn.Module:
    name = model.backbone_name
    if name in ("resnet50", "resnet101"):
        return model.backbone.layer4
    if name == "efficientnet_b0":
        return model.backbone.features[-1]
    if name == "densenet121":
        return model.backbone.features.denseblock4
    raise ValueError(f"No target layer for backbone '{name}'")


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def load_display_image(image_path: str, size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(img)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * image + alpha * heatmap_rgb).astype(np.uint8)


# -----------------------------------------------------------------------------
# Per-stone inference + Grad-CAM
# -----------------------------------------------------------------------------
def process_stone_composition(
    stone_images: list[dict],
    model: torch.nn.Module,
    grad_cam: GradCAM,
    eval_transform: transforms.Compose,
    class_names: list[str],
    display_size: int,
    device: torch.device,
) -> dict:
    """Run Model B + Grad-CAM for all images of one stone.

    Returns a dict with originals, overlays, per-image predictions,
    true composition, stone-level prediction, and within-stone variance.
    """
    originals, overlays = [], []
    per_image_preds: list[np.ndarray] = []
    true_comp = stone_images[0]["composition"]
    true_dominant = int(np.array(true_comp).argmax())

    for row in stone_images:
        orig = load_display_image(row["image_path"], display_size)

        tensor = eval_transform(Image.open(row["image_path"]).convert("RGB"))
        tensor = tensor.unsqueeze(0).to(device)

        # Backpropagate through the predicted dominant class
        heatmap = grad_cam(tensor)
        with torch.no_grad():
            comp_logits, _ = model(tensor)
            pred_comp = F.softmax(comp_logits, dim=1)[0].cpu().numpy()

        overlays.append(overlay_heatmap(orig, heatmap))
        originals.append(orig)
        per_image_preds.append(pred_comp)

    # Stone-level: mean then re-normalise
    mean_pred = np.mean(per_image_preds, axis=0)
    mean_pred /= mean_pred.sum()
    stone_dominant = int(mean_pred.argmax())

    # Within-stone variance: mean std across classes
    within_std = float(np.stack(per_image_preds).std(axis=0).mean())

    return {
        "originals":      originals,
        "overlays":       overlays,
        "per_image_preds": per_image_preds,
        "mean_pred":      mean_pred,
        "true_comp":      np.array(true_comp),
        "true_dominant":  true_dominant,
        "stone_dominant": stone_dominant,
        "correct":        stone_dominant == true_dominant,
        "within_std":     within_std,
        "class_names":    class_names,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_stone_figure(
    stones: list[dict],
    title: str,
    output_path: Path,
    display_size: int,
    class_names: list[str],
) -> None:
    """One row per stone: orig/CAM for each photo + bar chart of mean composition."""
    n_stones = len(stones)
    n_cols = 7  # orig + cam for 3 photos + composition bar
    fig, axes = plt.subplots(n_stones, n_cols, figsize=(n_cols * 2.2, n_stones * 2.6))
    if n_stones == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Photo A", "CAM A", "Photo B", "CAM B", "Photo C", "CAM C", "Composition"]
    for col_i, label in enumerate(col_labels):
        axes[0, col_i].set_title(label, fontsize=8)

    for row_i, stone in enumerate(stones):
        true_name = class_names[stone["true_dominant"]]
        pred_name = class_names[stone["stone_dominant"]]
        row_label = (
            f"{stone['stone_id']}\n"
            f"true={true_name}\npred={pred_name}\n"
            f"var={stone['within_std']:.3f}"
        )
        axes[row_i, 0].set_ylabel(row_label, fontsize=7, rotation=0,
                                   ha="right", va="center", labelpad=65)

        for photo_i in range(min(3, len(stone["originals"]))):
            axes[row_i, photo_i * 2].imshow(stone["originals"][photo_i])
            axes[row_i, photo_i * 2 + 1].imshow(stone["overlays"][photo_i])

        # Composition bar chart
        ax_bar = axes[row_i, 6]
        x = np.arange(len(class_names))
        ax_bar.bar(x, stone["true_comp"],  alpha=0.5, color="steelblue",  label="True")
        ax_bar.bar(x, stone["mean_pred"],  alpha=0.5, color="firebrick", label="Pred")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(class_names, fontsize=6, rotation=45, ha="right")
        ax_bar.set_ylim(0, 1)
        if row_i == 0:
            ax_bar.legend(fontsize=6)

        for ax in axes[row_i, :6]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_i, 6].set_yticks([0, 0.5, 1.0])
        axes[row_i, 6].tick_params(axis="y", labelsize=7)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n",    type=int, default=4,
                        help="Max stones per category")
    args = parser.parse_args()

    cfg  = load_config()
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
    log.info(f"Device: {device}")

    final_classes = cfg["class_remapping"]["final_classes"]
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    final_classes = [c for c in final_classes if c not in exclude]
    cfg["class_remapping"]["final_classes"] = final_classes
    class_names = final_classes

    # Load model
    ckpt_dir  = resolve_path(cfg, "checkpoints_composition_dir")
    ckpt_path = ckpt_dir / f"fold_{args.fold}" / "best_simple.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint at {ckpt_path}. Run 05_train_composition.py first.")
        sys.exit(1)

    model_module = import_module("04_model_composition")
    model = model_module.build_simple_composition_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded fold {args.fold} checkpoint (epoch {checkpoint['epoch']})")

    # Build val set
    ds_module  = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds_module.load_compositional_samples(
        images_csv, class_names, require_files_exist=True
    )
    test_fraction = cfg["cv"]["test_fraction"]
    train_val_idx, _ = ds_module.make_test_holdout(all_samples, test_fraction, seed)
    train_val_samples = [all_samples[i] for i in train_val_idx]
    folds = ds_module.make_stratified_folds(
        train_val_samples, cfg["cv"]["n_folds"], seed, cfg["cv"]["shuffle"]
    )
    _, val_idx = folds[args.fold]
    val_samples = [train_val_samples[i] for i in val_idx]

    # Group by stone
    stone_groups: dict[str, list[dict]] = {}
    for s in val_samples:
        stone_groups.setdefault(s.stone_id, []).append({
            "image_path":  s.image_path,
            "composition": s.composition,
        })

    # Grad-CAM setup
    target_layer = get_target_layer(model)
    grad_cam     = GradCAM(model, target_layer)
    eval_tf      = ds_module.build_transforms(cfg, train=False)
    display_size = cfg["image"]["input_size"]

    log.info(f"Running Grad-CAM for {len(stone_groups)} stones (fold {args.fold} val set)...")
    processed: list[dict] = []
    for stone_id, images in stone_groups.items():
        result = process_stone_composition(
            images, model, grad_cam, eval_tf, class_names, display_size, device
        )
        result["stone_id"] = stone_id
        processed.append(result)

    grad_cam.remove()

    # Split into categories
    correct   = [s for s in processed if     s["correct"]]
    incorrect = [s for s in processed if not s["correct"]]
    high_var  = sorted(processed, key=lambda s: s["within_std"], reverse=True)[: args.n]

    log.info(f"  Correct dominant:   {len(correct)}")
    log.info(f"  Incorrect dominant: {len(incorrect)}")

    gradcam_dir = ensure_dir(resolve_path(cfg, "figures_dir") / "gradcam_composition")

    if correct:
        save_stone_figure(
            correct[: args.n],
            title=f"Correct dominant prediction (fold {args.fold}, n={min(args.n, len(correct))})",
            output_path=gradcam_dir / "correct_dominant.png",
            display_size=display_size,
            class_names=class_names,
        )
    if incorrect:
        save_stone_figure(
            incorrect[: args.n],
            title=f"Incorrect dominant prediction (fold {args.fold}, n={min(args.n, len(incorrect))})",
            output_path=gradcam_dir / "incorrect_dominant.png",
            display_size=display_size,
            class_names=class_names,
        )
    if high_var:
        save_stone_figure(
            high_var,
            title=f"High within-stone variance (fold {args.fold}, top {len(high_var)})",
            output_path=gradcam_dir / "high_variance.png",
            display_size=display_size,
            class_names=class_names,
        )

    log.info("Grad-CAM visualization complete.")


if __name__ == "__main__":
    main()
