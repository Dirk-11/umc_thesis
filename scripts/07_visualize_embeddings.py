"""
07_visualize_embeddings.py — t-SNE embedding plots of backbone features.

Extracts image features from a trained backbone (before the classification head),
reduces them to 2D with t-SNE, and plots each stone colored by:
  - Pure vs mixed (Model A perspective)
  - Dominant mineral class (Model C perspective)

This shows whether the backbone has learned a meaningful feature space where
stones of the same type cluster together.

Outputs (figures/embeddings/):
  tsne_binary.png      — colored by pure/mixed label
  tsne_multiclass.png  — colored by dominant mineral class

Usage:
  python scripts/07_visualize_embeddings.py              # fold 0 binary checkpoint
  python scripts/07_visualize_embeddings.py --fold 2     # fold 2 checkpoint
  python scripts/07_visualize_embeddings.py --tag wd001_lrlow  # tagged experiment

Requires: run 05_train_binary.py first.
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("visualize_embeddings")

CLASS_NAMES = ["CaOx", "CHP", "UA", "MAP", "CYS"]
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
BINARY_COLORS = ["#377eb8", "#e41a1c"]  # blue=pure, red=mixed


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def extract_features(
    model: torch.nn.Module,
    samples: list,
    transform,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Extract backbone features for all samples, averaged per stone.

    Returns:
        features:  (n_stones, feature_dim) array
        stone_ids: list of stone IDs in the same order
    """
    model.eval()

    # Hook to capture backbone output before the classification head
    features_dict: dict[str, list] = {}

    def hook_fn(module, input, output):
        # Global average pool the spatial features → (batch, feature_dim)
        pooled = output.mean(dim=[2, 3]) if output.dim() == 4 else output
        features_dict["out"] = pooled.detach().cpu()

    # Attach hook to the last backbone layer before the head
    backbone_name = model.backbone_name
    if backbone_name in ("resnet18", "resnet50", "resnet101"):
        hook = model.backbone.layer4.register_forward_hook(hook_fn)
    elif backbone_name == "efficientnet_b0":
        hook = model.backbone.features[-1].register_forward_hook(hook_fn)
    elif backbone_name == "densenet121":
        hook = model.backbone.features.denseblock4.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Group samples by stone
    stone_groups: dict[str, list] = {}
    for s in samples:
        stone_groups.setdefault(s.stone_id, []).append(s)

    stone_ids = list(stone_groups.keys())
    all_features = []

    with torch.no_grad():
        for stone_id in stone_ids:
            stone_feats = []
            for s in stone_groups[stone_id]:
                img = Image.open(s.image_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                model(tensor)
                stone_feats.append(features_dict["out"].squeeze(0).numpy())
            # Average across the 3 photos of this stone
            all_features.append(np.mean(stone_feats, axis=0))

    hook.remove()
    return np.array(all_features), stone_ids


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_tsne(
    embedding: np.ndarray,
    labels: list[int],
    stone_ids: list[str],
    color_list: list[str],
    label_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for label_idx, (color, name) in enumerate(zip(color_list, label_names)):
        mask = [i for i, l in enumerate(labels) if l == label_idx]
        if not mask:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=name,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold checkpoint to use (default: 0)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag (must match --tag used during training)")
    parser.add_argument("--perplexity", type=float, default=20,
                        help="t-SNE perplexity (default: 20, try 10-50 for small datasets)")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = cfg["training"]["device"]
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Load binary model checkpoint
    # -------------------------------------------------------------------------
    ckpt_dir = resolve_path(cfg, "checkpoints_dir")
    if args.tag:
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.name + f"_{args.tag}")
    ckpt_path = ckpt_dir / f"fold_{args.fold}" / "best.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint at {ckpt_path}. Run 05_train_binary.py first.")
        sys.exit(1)

    model_module = import_module("04_model_binary")
    model = model_module.build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded checkpoint: {ckpt_path} (epoch {checkpoint['epoch']})")

    # -------------------------------------------------------------------------
    # Load all samples (train + val + test) for a rich embedding
    # -------------------------------------------------------------------------
    ds_module = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")

    purity_threshold = float(cfg["data"]["purity_threshold"])
    final_classes = cfg["class_remapping"]["final_classes"]
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    class_names = [c for c in final_classes if c not in exclude]

    # Binary samples (for pure/mixed labels)
    binary_samples = ds_module.load_image_samples(
        images_csv,
        require_files_exist=True,
        purity_threshold=purity_threshold,
        final_classes=final_classes,
    )

    # Multiclass samples (for dominant mineral labels)
    multiclass_samples = ds_module.load_multiclass_samples(
        images_csv, class_names, require_files_exist=True
    )
    excl_idx = {i for i, c in enumerate(class_names) if c in exclude}
    multiclass_samples = [s for s in multiclass_samples if s.label not in excl_idx]

    # Build stone-level label lookups
    binary_stone_label: dict[str, int] = {}
    for s in binary_samples:
        binary_stone_label[s.stone_id] = s.label

    multiclass_stone_label: dict[str, int] = {}
    for s in multiclass_samples:
        multiclass_stone_label[s.stone_id] = s.label

    eval_transform = ds_module.build_transforms(cfg, train=False)

    # -------------------------------------------------------------------------
    # Extract features using binary samples (all stones)
    # -------------------------------------------------------------------------
    log.info(f"Extracting features for {len({s.stone_id for s in binary_samples})} stones...")
    features, stone_ids = extract_features(model, binary_samples, eval_transform, device)
    log.info(f"Feature matrix: {features.shape}")

    # -------------------------------------------------------------------------
    # t-SNE
    # -------------------------------------------------------------------------
    log.info(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=seed,
                max_iter=1000, init="pca")
    embedding = tsne.fit_transform(features)
    log.info("t-SNE done.")

    # -------------------------------------------------------------------------
    # Plot 1: colored by pure/mixed
    # -------------------------------------------------------------------------
    binary_labels = [binary_stone_label.get(sid, -1) for sid in stone_ids]
    valid = [i for i, l in enumerate(binary_labels) if l >= 0]
    emb_bin = embedding[valid]
    lbl_bin = [binary_labels[i] for i in valid]

    output_dir = ensure_dir(resolve_path(cfg, "figures_dir") / "embeddings")
    suffix = f"_{args.tag}" if args.tag else ""

    plot_tsne(
        emb_bin, lbl_bin, [stone_ids[i] for i in valid],
        color_list=BINARY_COLORS,
        label_names=["Pure", "Mixed"],
        title=f"t-SNE — backbone features colored by pure/mixed (fold {args.fold}){' — ' + args.tag if args.tag else ''}",
        output_path=output_dir / f"tsne_binary{suffix}.png",
    )

    # -------------------------------------------------------------------------
    # Plot 2: colored by dominant mineral class
    # -------------------------------------------------------------------------
    mc_labels = [multiclass_stone_label.get(sid, -1) for sid in stone_ids]
    valid_mc = [i for i, l in enumerate(mc_labels) if l >= 0]
    emb_mc = embedding[valid_mc]
    lbl_mc = [mc_labels[i] for i in valid_mc]

    plot_tsne(
        emb_mc, lbl_mc, [stone_ids[i] for i in valid_mc],
        color_list=CLASS_COLORS[:len(class_names)],
        label_names=class_names,
        title=f"t-SNE — backbone features colored by mineral class (fold {args.fold}){' — ' + args.tag if args.tag else ''}",
        output_path=output_dir / f"tsne_multiclass{suffix}.png",
    )

    log.info("Embedding visualization complete.")


if __name__ == "__main__":
    main()
