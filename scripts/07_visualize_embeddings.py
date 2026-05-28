"""
07_visualize_embeddings.py — UMAP/t-SNE embedding plots of backbone features.

Extracts image features from a backbone (before the classification head) and
reduces them to 2D with UMAP (or t-SNE as fallback), colored by:
  - Pure vs mixed (Model A perspective)
  - Dominant mineral class (Model C perspective)

Can compare pretrained (ImageNet only) vs fine-tuned features side by side,
showing what the model learned during training.

Outputs (figures/embeddings/):
  umap_binary.png           — colored by pure/mixed (fine-tuned model)
  umap_multiclass.png       — colored by mineral class (fine-tuned model)
  umap_comparison_binary.png    — pretrained vs fine-tuned, colored by pure/mixed
  umap_comparison_multiclass.png — pretrained vs fine-tuned, colored by mineral class

Usage:
  python scripts/07_visualize_embeddings.py                    # fine-tuned, fold 0
  python scripts/07_visualize_embeddings.py --pretrained       # ImageNet weights only
  python scripts/07_visualize_embeddings.py --compare          # side-by-side comparison
  python scripts/07_visualize_embeddings.py --fold 2 --tag resnet18
  python scripts/07_visualize_embeddings.py --method tsne      # use t-SNE instead of UMAP

Requires: run 05_train_binary.py first (unless using --pretrained).
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

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("visualize_embeddings")

CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
BINARY_COLORS = ["#377eb8", "#e41a1c"]  # blue=pure, red=mixed


# -----------------------------------------------------------------------------
# Dimensionality reduction
# -----------------------------------------------------------------------------
def reduce_features(features: np.ndarray, method: str, seed: int,
                    n_neighbors: int = 15, min_dist: float = 0.1,
                    perplexity: float = 20) -> np.ndarray:
    """Reduce (n_stones, feat_dim) to (n_stones, 2).

    Always applies PCA to 50 components first — reduces noise and makes UMAP/t-SNE
    faster and more stable on high-dimensional backbone features.
    """
    from sklearn.decomposition import PCA
    n_components = min(50, features.shape[0] - 1, features.shape[1])
    features = PCA(n_components=n_components, random_state=seed).fit_transform(features)
    log.info(f"PCA → {features.shape[1]} components")

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=seed)
            return reducer.fit_transform(features)
        except ImportError:
            log.warning("umap-learn not installed — falling back to t-SNE. "
                        "Install with: pip install umap-learn")
            method = "tsne"

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=1000, init="pca")
    return tsne.fit_transform(features)


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def extract_features(
    model: torch.nn.Module,
    samples: list,
    transform,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Extract backbone features for all samples, averaged per stone."""
    model.eval()
    features_dict: dict = {}

    def hook_fn(module, input, output):
        pooled = output.mean(dim=[2, 3]) if output.dim() == 4 else output
        features_dict["out"] = pooled.detach().cpu()

    backbone_name = model.backbone_name
    if backbone_name in ("resnet18", "resnet50", "resnet101"):
        hook = model.backbone.layer4.register_forward_hook(hook_fn)
    elif backbone_name == "efficientnet_b0":
        hook = model.backbone.features[-1].register_forward_hook(hook_fn)
    elif backbone_name == "densenet121":
        hook = model.backbone.features.denseblock4.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

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
            all_features.append(np.mean(stone_feats, axis=0))

    hook.remove()
    return np.array(all_features), stone_ids


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def scatter_embedding(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: list[int],
    color_list: list[str],
    label_names: list[str],
    title: str,
    method: str,
) -> None:
    """Draw a single scatter plot on the given axes."""
    for label_idx, (color, name) in enumerate(zip(color_list, label_names)):
        mask = [i for i, l in enumerate(labels) if l == label_idx]
        if not mask:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=name, alpha=0.75,
            edgecolors="white", linewidths=0.4, s=60,
        )
    dim_name = "UMAP" if method == "umap" else "t-SNE"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"{dim_name} 1", fontsize=10)
    ax.set_ylabel(f"{dim_name} 2", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])


def save_single(
    embedding: np.ndarray,
    labels: list[int],
    color_list: list[str],
    label_names: list[str],
    title: str,
    output_path: Path,
    method: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_embedding(ax, embedding, labels, color_list, label_names, title, method)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def save_comparison(
    emb_pre: np.ndarray,
    emb_ft: np.ndarray,
    labels: list[int],
    color_list: list[str],
    label_names: list[str],
    output_path: Path,
    method: str,
    label_type: str,
) -> None:
    """Side-by-side pretrained vs fine-tuned plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_embedding(axes[0], emb_pre, labels, color_list, label_names,
                      f"Pretrained (ImageNet only)", method)
    scatter_embedding(axes[1], emb_ft, labels, color_list, label_names,
                      f"Fine-tuned", method)
    fig.suptitle(
        f"Backbone feature space — {label_type} — before vs after training",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------------
def load_pretrained_model(cfg: dict, device: torch.device):
    """Build model with ImageNet weights only — no fine-tuning checkpoint."""
    model_module = import_module("04_model_binary")
    model = model_module.build_model(cfg).to(device)
    log.info("Loaded pretrained (ImageNet) model — no fine-tuning checkpoint")
    return model


def load_finetuned_model(cfg: dict, device: torch.device, ckpt_path: Path):
    """Build Model A and load fine-tuned checkpoint."""
    model_module = import_module("04_model_binary")
    model = model_module.build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded fine-tuned checkpoint: {ckpt_path} (epoch {checkpoint['epoch']})")
    return model


def load_finetuned_multiclass_model(cfg: dict, device: torch.device, ckpt_path: Path):
    """Build Model C (multiclass) and load fine-tuned checkpoint."""
    model_module = import_module("04_model_binary")
    mc_cfg = cfg["model_multiclass"]
    model = model_module.StoneClassifier(
        backbone_name=mc_cfg["backbone"],
        weights=mc_cfg["pretrained_weights"],
        num_classes=mc_cfg["num_classes"],
        dropout=mc_cfg["dropout"],
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded multiclass checkpoint: {ckpt_path} (epoch {checkpoint['epoch']})")
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold checkpoint to use (default: 0)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag matching --tag used during training")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights only (no fine-tuning checkpoint)")
    parser.add_argument("--compare", action="store_true",
                        help="Generate side-by-side pretrained vs fine-tuned comparison plots")
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap",
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--perplexity", type=float, default=20,
                        help="t-SNE perplexity (default: 20)")
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
    # Resolve checkpoint paths (Model A for binary plot, Model C for mineral plot)
    # -------------------------------------------------------------------------
    ckpt_dir_a = resolve_path(cfg, "checkpoints_dir")
    if args.tag:
        ckpt_dir_a = ckpt_dir_a.parent / (ckpt_dir_a.name + f"_{args.tag}")
    ckpt_path_a = ckpt_dir_a / f"fold_{args.fold}" / "best.pt"

    ckpt_dir_c = resolve_path(cfg, "checkpoints_multiclass_dir")
    ckpt_path_c = ckpt_dir_c / f"fold_{args.fold}" / "best.pt"

    if not args.pretrained and not ckpt_path_a.exists():
        log.error(f"No Model A checkpoint at {ckpt_path_a}. Run 05_train_binary.py first "
                  f"or use --pretrained.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load samples
    # -------------------------------------------------------------------------
    ds_module = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    purity_threshold = float(cfg["data"]["purity_threshold"])
    final_classes = cfg["class_remapping"]["final_classes"]
    exclude = cfg["class_remapping"].get("exclude_classes", [])
    class_names = [c for c in final_classes if c not in exclude]

    binary_samples = ds_module.load_image_samples(
        images_csv, require_files_exist=True,
        purity_threshold=purity_threshold, final_classes=final_classes,
    )
    multiclass_samples = ds_module.load_multiclass_samples(
        images_csv, class_names, require_files_exist=True,
    )
    excl_idx = {i for i, c in enumerate(class_names) if c in exclude}
    multiclass_samples = [s for s in multiclass_samples if s.label not in excl_idx]

    binary_stone_label = {s.stone_id: s.label for s in binary_samples}
    multiclass_stone_label = {s.stone_id: s.label for s in multiclass_samples}

    eval_transform = ds_module.build_transforms(cfg, train=False)
    output_dir = ensure_dir(resolve_path(cfg, "figures_dir") / "embeddings")
    suffix = f"_{args.tag}" if args.tag else ""
    method = args.method

    # -------------------------------------------------------------------------
    # Extract features
    # -------------------------------------------------------------------------
    def get_embedding(model) -> tuple[np.ndarray, list[str]]:
        n_stones = len({s.stone_id for s in binary_samples})
        log.info(f"Extracting features for {n_stones} stones...")
        features, stone_ids = extract_features(model, binary_samples, eval_transform, device)
        log.info(f"Feature matrix: {features.shape} — running {method.upper()}...")
        embedding = reduce_features(
            features, method, seed,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            perplexity=args.perplexity,
        )
        log.info(f"{method.upper()} done.")
        return embedding, stone_ids

    def get_labels(stone_ids, label_dict):
        labels = [label_dict.get(sid, -1) for sid in stone_ids]
        valid = [i for i, l in enumerate(labels) if l >= 0]
        return valid, [labels[i] for i in valid]

    # -------------------------------------------------------------------------
    # Single model plots (pretrained OR fine-tuned)
    # -------------------------------------------------------------------------
    if not args.compare:
        model_a = (load_pretrained_model(cfg, device) if args.pretrained
                   else load_finetuned_model(cfg, device, ckpt_path_a))
        embedding_a, stone_ids_a = get_embedding(model_a)
        del model_a
        model_label = "pretrained" if args.pretrained else f"fine-tuned fold {args.fold}"

        # Binary plot — Model A features
        valid, lbl = get_labels(stone_ids_a, binary_stone_label)
        save_single(
            embedding_a[valid], lbl, BINARY_COLORS, ["Pure", "Mixed"],
            title=f"{method.upper()} — pure/mixed — {model_label}{' — ' + args.tag if args.tag else ''}",
            output_path=output_dir / f"{method}_binary{'_pretrained' if args.pretrained else ''}{suffix}.png",
            method=method,
        )

        # Mineral class plot — Model C features (trained to separate mineral types)
        if not args.pretrained and ckpt_path_c.exists():
            model_c = load_finetuned_multiclass_model(cfg, device, ckpt_path_c)
            embedding_c, stone_ids_c = get_embedding(model_c)
            del model_c
            valid_mc, lbl_mc = get_labels(stone_ids_c, multiclass_stone_label)
            mc_model_label = f"Model C fine-tuned fold {args.fold}"
        else:
            embedding_c, stone_ids_c = embedding_a, stone_ids_a
            valid_mc, lbl_mc = get_labels(stone_ids_a, multiclass_stone_label)
            mc_model_label = model_label
            if not args.pretrained:
                log.warning(f"No Model C checkpoint at {ckpt_path_c} — using Model A features for mineral plot")

        save_single(
            embedding_c[valid_mc], lbl_mc, CLASS_COLORS[:len(class_names)], class_names,
            title=f"{method.upper()} — mineral class — {mc_model_label}{' — ' + args.tag if args.tag else ''}",
            output_path=output_dir / f"{method}_multiclass{'_pretrained' if args.pretrained else ''}{suffix}.png",
            method=method,
        )

    # -------------------------------------------------------------------------
    # Comparison plots (pretrained vs fine-tuned side by side)
    # Model A features for binary; Model C features for mineral class
    # -------------------------------------------------------------------------
    else:
        log.info("=== Extracting pretrained features (Model A backbone) ===")
        model_pre = load_pretrained_model(cfg, device)
        emb_pre, stone_ids = get_embedding(model_pre)
        del model_pre

        log.info("=== Extracting fine-tuned Model A features ===")
        model_ft_a = load_finetuned_model(cfg, device, ckpt_path_a)
        emb_ft_a, _ = get_embedding(model_ft_a)
        del model_ft_a

        # Binary comparison — pretrained vs Model A
        valid, lbl = get_labels(stone_ids, binary_stone_label)
        save_comparison(
            emb_pre[valid], emb_ft_a[valid], lbl,
            BINARY_COLORS, ["Pure", "Mixed"],
            output_path=output_dir / f"{method}_comparison_binary{suffix}.png",
            method=method, label_type="pure/mixed",
        )

        # Mineral class comparison — pretrained vs Model C (if available)
        if ckpt_path_c.exists():
            log.info("=== Extracting fine-tuned Model C features ===")
            model_ft_c = load_finetuned_multiclass_model(cfg, device, ckpt_path_c)
            emb_ft_c, stone_ids_c = get_embedding(model_ft_c)
            del model_ft_c
            valid_mc, lbl_mc = get_labels(stone_ids_c, multiclass_stone_label)
            emb_pre_mc = emb_pre  # pretrained baseline is the same backbone
        else:
            log.warning(f"No Model C checkpoint at {ckpt_path_c} — using Model A features for mineral comparison")
            emb_ft_c = emb_ft_a
            valid_mc, lbl_mc = get_labels(stone_ids, multiclass_stone_label)
            emb_pre_mc = emb_pre

        save_comparison(
            emb_pre_mc[valid_mc], emb_ft_c[valid_mc], lbl_mc,
            CLASS_COLORS[:len(class_names)], class_names,
            output_path=output_dir / f"{method}_comparison_multiclass{suffix}.png",
            method=method, label_type="mineral class",
        )

    log.info("Embedding visualization complete.")


if __name__ == "__main__":
    main()
