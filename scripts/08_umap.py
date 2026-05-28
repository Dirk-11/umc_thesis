"""
08_umap.py — UMAP of Model C backbone embeddings coloured by mineral class.

Loads the best checkpoint for each cross-validation fold, extracts the
512-dim backbone features (before the classification head) for every stone
photo in that fold's val set, averages the 3 photos per stone, then
projects all stones to 2D with UMAP.

Points are coloured by true primary mineral class.

Outputs (figures/):
  umap_model_c.png        (val set, default)
  umap_model_c_test.png   (held-out test set, --test flag)

Usage:
  python scripts/08_umap.py
  python scripts/08_umap.py --test
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("umap")

# One colour per mineral class (up to 6)
CLASS_COLORS = ["#2166AC", "#D6604D", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628"]


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (embeddings, labels, stone_ids) for every image in loader."""
    model.eval()
    embs, labs, sids = [], [], []
    for images, labels, stone_ids in loader:
        images = images.to(device, non_blocking=True)
        feats  = model.backbone(images).cpu().numpy()   # (B, 512)
        embs.append(feats)
        labs.extend(labels.tolist())
        sids.extend(list(stone_ids))
    return np.concatenate(embs, axis=0), np.array(labs), sids


def aggregate_to_stone(
    embs: np.ndarray,
    labels: np.ndarray,
    stone_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per-image embeddings to stone level."""
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(stone_ids):
        groups[sid].append(i)

    stone_embs, stone_labels, stone_ids_out = [], [], []
    for sid, idxs in groups.items():
        stone_embs.append(embs[idxs].mean(axis=0))
        stone_labels.append(int(labels[idxs[0]]))
        stone_ids_out.append(sid)
    return (
        np.stack(stone_embs),
        np.array(stone_labels),
        np.array(stone_ids_out),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Use held-out test set instead of cross-validation val sets.")
    args = parser.parse_args()

    try:
        import umap as umap_lib
    except ImportError:
        log.error("umap-learn is not installed. Run: pip install umap-learn")
        sys.exit(1)

    cfg  = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = cfg["training_multiclass"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    class_names = cfg["class_remapping"]["final_classes"]

    ds_module  = import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    all_samples = ds_module.load_multiclass_samples(
        images_csv, class_names, require_files_exist=True
    )

    exclude = cfg["class_remapping"].get("exclude_classes", [])
    if exclude:
        excl_idx  = {i for i, c in enumerate(class_names) if c in exclude}
        all_samples = [s for s in all_samples if s.label not in excl_idx]
        class_names = [c for c in class_names if c not in exclude]
        cfg["model_multiclass"]["num_classes"] = len(class_names)

    dominant_labels = [s.label for s in all_samples]
    test_fraction   = cfg["cv"]["test_fraction"]
    train_val_idx, test_idx = ds_module.make_test_holdout(
        all_samples, test_fraction, seed, stratify_labels=dominant_labels
    )
    train_val_samples   = [all_samples[i] for i in train_val_idx]
    train_val_dominant  = [dominant_labels[i] for i in train_val_idx]

    folds = ds_module.make_stratified_folds(
        train_val_samples,
        n_folds=cfg["cv"]["n_folds"],
        seed=seed,
        shuffle=cfg["cv"]["shuffle"],
        stratify_labels=train_val_dominant,
    )

    ckpt_dir     = resolve_path(cfg, "checkpoints_multiclass_dir")
    figures_dir  = ensure_dir(resolve_path(cfg, "figures_dir"))
    model_module = import_module("04_model_binary")

    all_embs: list[np.ndarray] = []
    all_labs: list[np.ndarray] = []

    if args.test:
        # Single pass: average embeddings across folds for test stones
        fold_embs: dict[str, list[np.ndarray]] = {}
        fold_labs: dict[str, int] = {}

        for fold_i in range(len(folds)):
            ckpt_path = ckpt_dir / f"fold_{fold_i}" / "best.pt"
            if not ckpt_path.exists():
                log.warning(f"Fold {fold_i}: checkpoint not found, skipping")
                continue
            mc = cfg["model_multiclass"]
            model = model_module.StoneClassifier(
                backbone_name=mc["backbone"],
                weights=mc["pretrained_weights"],
                num_classes=mc["num_classes"],
                dropout=mc["dropout"],
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])

            loader = ds_module.make_multiclass_dataloader(
                all_samples, test_idx, cfg, train=False
            )
            embs, labs, sids = extract_embeddings(model, loader, device)
            stone_embs, stone_labs, stone_sid_arr = aggregate_to_stone(embs, labs, sids)
            for sid, emb, lab in zip(stone_sid_arr, stone_embs, stone_labs):
                fold_embs.setdefault(sid, []).append(emb)
                fold_labs[sid] = int(lab)

        for sid, emb_list in fold_embs.items():
            all_embs.append(np.stack(emb_list).mean(axis=0))
            all_labs.append(fold_labs[sid])

        set_label = "held-out test set"
        out_name  = "umap_model_c_test.png"
    else:
        # Cross-validation: each stone appears once (its own fold's val set)
        for fold_i, (_, val_idx) in enumerate(folds):
            ckpt_path = ckpt_dir / f"fold_{fold_i}" / "best.pt"
            if not ckpt_path.exists():
                log.warning(f"Fold {fold_i}: checkpoint not found, skipping")
                continue
            mc = cfg["model_multiclass"]
            model = model_module.StoneClassifier(
                backbone_name=mc["backbone"],
                weights=mc["pretrained_weights"],
                num_classes=mc["num_classes"],
                dropout=mc["dropout"],
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])

            loader = ds_module.make_multiclass_dataloader(
                train_val_samples, val_idx, cfg, train=False
            )
            embs, labs, sids = extract_embeddings(model, loader, device)
            stone_embs, stone_labs, _ = aggregate_to_stone(embs, labs, sids)
            all_embs.append(stone_embs)
            all_labs.append(stone_labs)

        set_label = "cross-validation"
        out_name  = "umap_model_c.png"

    if not all_embs:
        log.error("No embeddings collected — are checkpoints present?")
        sys.exit(1)

    X = np.concatenate(all_embs, axis=0)   # (N_stones, 512)
    y = np.concatenate(all_labs, axis=0)   # (N_stones,)
    log.info(f"Running UMAP on {len(X)} stones...")

    reducer = umap_lib.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(X)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        mask = y == cls_idx
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, label=f"{cls_name} (n={mask.sum()})",
            s=40, alpha=0.75, edgecolors="none",
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"Model C backbone embeddings — {set_label}", fontsize=13)
    ax.legend(fontsize=9, markerscale=1.4, framealpha=0.9)
    fig.tight_layout()

    out_path = figures_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
