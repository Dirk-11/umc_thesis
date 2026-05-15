"""
07_visualize_multiclass.py — Grad-CAM heatmaps for Model C (5-class classifier).

Grad-CAM highlights which regions of a stone photo the model attends to when
predicting the dominant mineral component. Useful for verifying the model looks
at the stone surface rather than background artefacts.

What this script produces:
  - One figure per class showing correct predictions (model attended to right features)
  - One figure per class showing incorrect predictions (what confused the model)
  - A summary grid with one correct and one incorrect example per class

Outputs (figures/gradcam_multiclass/):
  correct_<class>.png     — correctly predicted stones for each class
  incorrect_<class>.png   — incorrectly predicted stones for each class
  summary_grid.png        — one correct + one incorrect example per class

Usage:
  python scripts/07_visualize_multiclass.py              # fold 0, 4 examples per class
  python scripts/07_visualize_multiclass.py --fold 2     # use fold 2 checkpoint
  python scripts/07_visualize_multiclass.py --n 8        # 8 examples per class

Requires: run 05_train_multiclass.py first.
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

log = setup_logging("visualize_multiclass")


# -----------------------------------------------------------------------------
# Grad-CAM (identical to 07_visualize_binary.py)
# -----------------------------------------------------------------------------
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

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
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model) -> torch.nn.Module:
    name = model.backbone_name
    if name in ("resnet18", "resnet50", "resnet101"):
        return model.backbone.layer4
    if name == "efficientnet_b0":
        return model.backbone.features[-1]
    if name == "densenet121":
        return model.backbone.features.denseblock4
    raise ValueError(f"No target layer defined for backbone '{name}'.")


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def load_display_image(image_path: str, size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * image + alpha * heatmap_rgb).astype(np.uint8)


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
    class_names: list[str],
) -> dict:
    """Run model + Grad-CAM for all images of one stone.

    Grad-CAM is computed toward the predicted class for each image.
    Returns a dict with originals, overlays, per-image predicted classes,
    confidence scores, the true label, and the stone-level prediction.
    """
    originals, overlays, pred_classes, confidences = [], [], [], []
    true_label = stone_images[0]["label"]

    for row in stone_images:
        orig = load_display_image(row["image_path"], display_size)
        tensor = eval_transform(Image.open(row["image_path"]).convert("RGB"))
        tensor = tensor.unsqueeze(0).to(device)

        # Predict and compute Grad-CAM toward the predicted class
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        # Re-run with grad for Grad-CAM
        heatmap = grad_cam(tensor, class_idx=pred_idx)

        originals.append(orig)
        overlays.append(overlay_heatmap(orig, heatmap))
        pred_classes.append(pred_idx)
        confidences.append(confidence)

    # Stone-level prediction: majority vote across images
    from collections import Counter
    stone_pred = Counter(pred_classes).most_common(1)[0][0]

    return {
        "stone_id":    stone_id,
        "true_label":  true_label,
        "stone_pred":  stone_pred,
        "originals":   originals,
        "overlays":    overlays,
        "pred_classes": pred_classes,
        "confidences": confidences,
        "class_names": class_names,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_class_figure(
    stones: list[dict],
    title: str,
    output_path: Path,
    display_size: int,
    class_names: list[str],
) -> None:
    """Save a figure for a set of stones — one row per stone."""
    n_stones = len(stones)
    n_cols = 6  # orig + cam for each of 3 photos

    fig, axes = plt.subplots(n_stones, n_cols, figsize=(n_cols * 2.2, n_stones * 2.4))
    if n_stones == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Photo A", "CAM A", "Photo B", "CAM B", "Photo C", "CAM C"]
    for col_i, label in enumerate(col_labels):
        axes[0, col_i].set_title(label, fontsize=9)

    for row_i, stone in enumerate(stones):
        true_name = class_names[stone["true_label"]]
        pred_name = class_names[stone["stone_pred"]]
        mean_conf = float(np.mean(stone["confidences"]))

        row_label = (
            f"{stone['stone_id']}\n"
            f"true={true_name}\n"
            f"pred={pred_name}  ({mean_conf:.2f})"
        )
        axes[row_i, 0].set_ylabel(row_label, fontsize=7.5, rotation=0,
                                   ha="right", va="center", labelpad=70)

        for photo_i in range(min(3, len(stone["originals"]))):
            col_orig = photo_i * 2
            col_cam  = photo_i * 2 + 1
            axes[row_i, col_orig].imshow(stone["originals"][photo_i])
            axes[row_i, col_cam].imshow(stone["overlays"][photo_i])
            img_pred = class_names[stone["pred_classes"][photo_i]]
            axes[row_i, col_cam].set_xlabel(
                f"{img_pred} ({stone['confidences'][photo_i]:.2f})", fontsize=7
            )

        for ax in axes[row_i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def save_summary_grid(
    correct_examples: dict[str, dict | None],
    incorrect_examples: dict[str, dict | None],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Two rows per class: one correct, one incorrect example."""
    n_cols = 6
    n_rows = len(class_names) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))

    col_labels = ["Photo A", "CAM A", "Photo B", "CAM B", "Photo C", "CAM C"]
    for col_i, label in enumerate(col_labels):
        axes[0, col_i].set_title(label, fontsize=9)

    for cls_i, cls_name in enumerate(class_names):
        for kind_i, (kind, examples) in enumerate([("correct", correct_examples),
                                                    ("incorrect", incorrect_examples)]):
            row_i = cls_i * 2 + kind_i
            stone = examples.get(cls_name)

            if stone is None:
                for ax in axes[row_i]:
                    ax.axis("off")
                axes[row_i, 0].set_ylabel(
                    f"{cls_name}\n({kind}, no examples)",
                    fontsize=8, rotation=0, ha="right", va="center", labelpad=70,
                )
                continue

            true_name = class_names[stone["true_label"]]
            pred_name = class_names[stone["stone_pred"]]
            mean_conf = float(np.mean(stone["confidences"]))
            axes[row_i, 0].set_ylabel(
                f"{cls_name} [{kind}]\n{stone['stone_id']}\ntrue={true_name} pred={pred_name} ({mean_conf:.2f})",
                fontsize=7.5, rotation=0, ha="right", va="center", labelpad=70,
            )

            for photo_i in range(min(3, len(stone["originals"]))):
                axes[row_i, photo_i * 2].imshow(stone["originals"][photo_i])
                axes[row_i, photo_i * 2 + 1].imshow(stone["overlays"][photo_i])
                img_pred = class_names[stone["pred_classes"][photo_i]]
                axes[row_i, photo_i * 2 + 1].set_xlabel(
                    f"{img_pred} ({stone['confidences'][photo_i]:.2f})", fontsize=7
                )

            for ax in axes[row_i]:
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle("Grad-CAM summary — correct vs incorrect per class", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold checkpoint to use (default: 0)")
    parser.add_argument("--n", type=int, default=4,
                        help="Max stones per class per category (default: 4)")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = cfg["training_multiclass"]["device"]
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Resolve class names (excluding OTH if configured)
    # -------------------------------------------------------------------------
    class_names = cfg["class_remapping"]["final_classes"]
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    class_names = [c for c in class_names if c not in exclude]
    n_classes = len(class_names)
    log.info(f"Classes ({n_classes}): {class_names}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    ckpt_dir = resolve_path(cfg, "checkpoints_multiclass_dir")
    ckpt_path = ckpt_dir / f"fold_{args.fold}" / "best.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint at {ckpt_path}. Run 05_train_multiclass.py first.")
        sys.exit(1)

    model_module = import_module("04_model_binary")
    cfg["model"]["num_classes"] = n_classes
    model = model_module.build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded fold {args.fold} checkpoint (epoch {checkpoint['epoch']})")

    # -------------------------------------------------------------------------
    # Build val set for this fold
    # -------------------------------------------------------------------------
    ds_module = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds_module.load_multiclass_samples(
        images_csv, class_names, require_files_exist=True
    )

    # Drop excluded classes
    excl_idx = {i for i, c in enumerate(class_names) if c in exclude}
    all_samples = [s for s in all_samples if s.label not in excl_idx]

    # Use binary labels for stratification to match training splits
    binary_labels = [s.binary_label for s in all_samples]
    folds = ds_module.make_stratified_folds(
        all_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
        stratify_labels=binary_labels,
    )
    _, val_idx = folds[args.fold]
    val_samples = [all_samples[i] for i in val_idx]

    # Group by stone
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
    # Run Grad-CAM for every stone
    # -------------------------------------------------------------------------
    log.info(f"Running Grad-CAM for {len(stone_groups)} stones in fold {args.fold} val set...")
    processed: list[dict] = []
    for stone_id, images in stone_groups.items():
        result = process_stone(
            stone_id, images, model, grad_cam,
            eval_transform, display_size, device, class_names,
        )
        processed.append(result)

    grad_cam.remove()

    # -------------------------------------------------------------------------
    # Sort into correct / incorrect per class
    # -------------------------------------------------------------------------
    correct:   dict[str, list[dict]] = {c: [] for c in class_names}
    incorrect: dict[str, list[dict]] = {c: [] for c in class_names}

    for stone in processed:
        true_name = class_names[stone["true_label"]]
        if stone["stone_pred"] == stone["true_label"]:
            correct[true_name].append(stone)
        else:
            incorrect[true_name].append(stone)

    for cls_name in class_names:
        log.info(f"  {cls_name}: {len(correct[cls_name])} correct, "
                 f"{len(incorrect[cls_name])} incorrect")

    # -------------------------------------------------------------------------
    # Save figures
    # -------------------------------------------------------------------------
    gradcam_dir = ensure_dir(resolve_path(cfg, "figures_dir") / "gradcam_multiclass")

    for cls_name in class_names:
        subset = correct[cls_name][: args.n]
        if subset:
            save_class_figure(
                subset,
                title=f"Correct predictions — {cls_name} (fold {args.fold})",
                output_path=gradcam_dir / f"correct_{cls_name}.png",
                display_size=display_size,
                class_names=class_names,
            )

        subset = incorrect[cls_name][: args.n]
        if subset:
            save_class_figure(
                subset,
                title=f"Incorrect predictions — true {cls_name} (fold {args.fold})",
                output_path=gradcam_dir / f"incorrect_{cls_name}.png",
                display_size=display_size,
                class_names=class_names,
            )

    save_summary_grid(
        correct_examples={c: correct[c][0] if correct[c] else None for c in class_names},
        incorrect_examples={c: incorrect[c][0] if incorrect[c] else None for c in class_names},
        class_names=class_names,
        output_path=gradcam_dir / "summary_grid.png",
    )

    log.info("Grad-CAM visualization complete.")


if __name__ == "__main__":
    main()
