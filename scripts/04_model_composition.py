"""
04_model_composition.py — Compositional regression model with multi-label detection (Model B).

Architecture:
  - Shared ResNet50 backbone (same structure as Model A in 04_model_binary.py)
  - Parallel regression heads: one per class (CaOx, CHP, UA, MAP, CYS, OTH)
    Each head: Linear(feat_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1)
    Outputs softmax probabilities for composition (sums to 1.0)
  - Auxiliary multi-label detection head:
    Linear(feat_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, n_classes)
    Outputs class presence/absence probabilities (sigmoid per class, independent)

Usage:
  model = build_composition_model(cfg)
  model.freeze_backbone()       # train heads only (Phase 1)
  model.unfreeze_backbone()     # full fine-tuning (Phase 2)
  comp_logits, pres_logits = model(x)   # raw logits — apply softmax / sigmoid externally

Run as script to verify model builds and forward-pass is valid:
  python scripts/04_model_composition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

log = setup_logging("model_composition")


# -----------------------------------------------------------------------------
# Backbone — reuse builders from 04_model_binary.py
# -----------------------------------------------------------------------------
def _get_backbones():
    from importlib import import_module
    return import_module("04_model_binary").BACKBONES


# -----------------------------------------------------------------------------
# Expert head
# -----------------------------------------------------------------------------
class ExpertHead(nn.Module):
    """Single expert head: produces one scalar logit for one composition class.

    Linear(feat_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 1)
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, 1)


# -----------------------------------------------------------------------------
# Compositional model with multi-task heads
# -----------------------------------------------------------------------------
class CompositionModel(nn.Module):
    """Shared backbone + composition regression + presence detection heads.

    Forward pass:
      1. Backbone extracts feature vector (batch, feat_dim)
      2. Composition heads → raw logits (batch, n_classes)
      3. Presence head → class-presence logits (batch, n_classes)

    Returns:
      (composition_logits, presence_logits)
      Apply softmax to composition (sums to 1.0) and sigmoid to presence.
    """

    def __init__(
        self,
        backbone_name: str,
        weights: str,
        class_names: list[str],
        hidden_dim: int,
        dropout: float,
        aux_head_hidden_dim: int | None = None,
    ):
        super().__init__()
        backbones = _get_backbones()
        if backbone_name not in backbones:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(backbones.keys())}"
            )
        self.backbone, feat_dim = backbones[backbone_name](weights)
        self.backbone_name = backbone_name
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Composition regression heads (one per class)
        self.composition_heads = nn.ModuleList([
            ExpertHead(feat_dim, hidden_dim, dropout)
            for _ in class_names
        ])

        # Auxiliary multi-label detection head (presence/absence per class)
        if aux_head_hidden_dim is None:
            aux_head_hidden_dim = hidden_dim
        self.presence_head = nn.Sequential(
            nn.Linear(feat_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(aux_head_hidden_dim, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (composition_logits, presence_logits).

        composition_logits: (batch, n_classes) — apply softmax for composition probs
        presence_logits:    (batch, n_classes) — apply sigmoid for presence probs
        """
        features = self.backbone(x)  # (B, feat_dim)
        comp_logits = torch.cat(
            [h(features) for h in self.composition_heads], dim=1
        )  # (B, n_classes)
        pres_logits = self.presence_head(features)  # (B, n_classes)
        return comp_logits, pres_logits

    def predict_composition(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax composition probabilities (sums to 1.0)."""
        comp_logits, _ = self.forward(x)
        return F.softmax(comp_logits, dim=1)

    def predict_presence(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid presence probabilities (independent per class)."""
        _, pres_logits = self.forward(x)
        return torch.sigmoid(pres_logits)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def head_parameters(self):
        """Return parameters for both composition and presence heads (not backbone)."""
        params = list(self.composition_heads.parameters())
        params += list(self.presence_head.parameters())
        return iter(params)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Simple baseline: single shared regression head
# -----------------------------------------------------------------------------
class SimpleCompositionModel(nn.Module):
    """Shared backbone + single regression head + presence detection head.

    Baseline for comparison against CompositionModel (parallel expert heads).
    The only difference: one Linear(feat_dim → n_classes) replaces the 6
    parallel expert heads. All other architecture, training, and evaluation
    code is identical — making this a clean ablation.
    """

    def __init__(
        self,
        backbone_name: str,
        weights: str,
        class_names: list[str],
        hidden_dim: int,
        dropout: float,
        aux_head_hidden_dim: int | None = None,
    ):
        super().__init__()
        backbones = _get_backbones()
        if backbone_name not in backbones:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(backbones.keys())}"
            )
        self.backbone, feat_dim = backbones[backbone_name](weights)
        self.backbone_name = backbone_name
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Single shared composition head
        self.composition_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.n_classes),
        )

        # Auxiliary presence head (identical to CompositionModel)
        if aux_head_hidden_dim is None:
            aux_head_hidden_dim = hidden_dim
        self.presence_head = nn.Sequential(
            nn.Linear(feat_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(aux_head_hidden_dim, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (composition_logits, presence_logits) — same interface as CompositionModel."""
        features = self.backbone(x)
        comp_logits = self.composition_head(features)   # (B, n_classes)
        pres_logits = self.presence_head(features)      # (B, n_classes)
        return comp_logits, pres_logits

    def predict_composition(self, x: torch.Tensor) -> torch.Tensor:
        comp_logits, _ = self.forward(x)
        return F.softmax(comp_logits, dim=1)

    def predict_presence(self, x: torch.Tensor) -> torch.Tensor:
        _, pres_logits = self.forward(x)
        return torch.sigmoid(pres_logits)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def head_parameters(self):
        params = list(self.composition_head.parameters())
        params += list(self.presence_head.parameters())
        return iter(params)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Factory + warm-start
# -----------------------------------------------------------------------------
def build_composition_model(cfg: dict) -> CompositionModel:
    """Build CompositionModel from config, optionally warm-starting backbone."""
    m = cfg["model_composition"]
    classes = cfg["class_remapping"]["final_classes"]

    model = CompositionModel(
        backbone_name=m["backbone"],
        weights=m["pretrained_weights"],
        class_names=classes,
        hidden_dim=m["head_hidden_dim"],
        dropout=m["head_dropout"],
        aux_head_hidden_dim=m.get("aux_head_hidden_dim"),
    )
    log.info(
        f"Built CompositionModel: {m['backbone']} backbone + "
        f"{len(classes)} regression heads + presence detection head "
        f"({model.total_param_count():,} params total)"
    )

    warm_start = m.get("warm_start_from")
    if warm_start:
        load_backbone_from_binary_checkpoint(model, warm_start)

    return model


def build_simple_composition_model(cfg: dict) -> SimpleCompositionModel:
    """Build SimpleCompositionModel (single shared head) from config.

    Uses the same model_composition config section as build_composition_model
    so the two models are trained under identical hyperparameter conditions.
    """
    m = cfg["model_composition"]
    classes = cfg["class_remapping"]["final_classes"]

    model = SimpleCompositionModel(
        backbone_name=m["backbone"],
        weights=m["pretrained_weights"],
        class_names=classes,
        hidden_dim=m["head_hidden_dim"],
        dropout=m["head_dropout"],
        aux_head_hidden_dim=m.get("aux_head_hidden_dim"),
    )
    log.info(
        f"Built SimpleCompositionModel: {m['backbone']} backbone + "
        f"single shared head + presence detection head "
        f"({model.total_param_count():,} params total)"
    )

    warm_start = m.get("warm_start_from")
    if warm_start:
        load_backbone_from_binary_checkpoint(model, warm_start)

    return model


def load_backbone_from_binary_checkpoint(
    model: CompositionModel,
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> None:
    """Copy backbone weights from a Model A (StoneClassifier) checkpoint.

    Only backbone.* keys are transferred; heads are left freshly initialised.
    This lets Model B inherit visual features already tuned for kidney stones.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state, strict=True)
    log.info(
        f"Warm-started backbone from Model A checkpoint: {ckpt_path} "
        f"(epoch {ckpt.get('epoch', '?')})"
    )


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def main() -> None:
    from utils import load_config

    cfg = load_config()
    model = build_composition_model(cfg)

    log.info(f"Total params:     {model.total_param_count():,}")
    log.info(f"Trainable params: {model.trainable_param_count():,}")

    model.freeze_backbone()
    log.info(f"After freeze:     {model.trainable_param_count():,} trainable (heads only)")

    model.unfreeze_backbone()
    log.info(f"After unfreeze:   {model.trainable_param_count():,} trainable (all)")

    size = cfg["image"]["input_size"]
    x = torch.randn(4, 3, size, size)
    comp_logits, pres_logits = model(x)
    probs_comp = model.predict_composition(x)
    probs_pres = model.predict_presence(x)

    n_classes = len(cfg["class_remapping"]["final_classes"])
    log.info(f"Forward pass OK:")
    log.info(f"  Input:              {tuple(x.shape)}")
    log.info(f"  Composition logits: {tuple(comp_logits.shape)}")
    log.info(f"  Presence logits:    {tuple(pres_logits.shape)}")
    log.info(f"  Composition probs sum: {probs_comp.sum(dim=1).tolist()}")
    log.info(f"  Presence probs range:  [{probs_pres.min():.4f}, {probs_pres.max():.4f}]")

    assert probs_comp.shape == (4, n_classes)
    assert probs_pres.shape == (4, n_classes)
    assert torch.allclose(probs_comp.sum(dim=1), torch.ones(4), atol=1e-5), \
        "Composition probabilities must sum to 1"
    assert (probs_pres >= 0).all() and (probs_pres <= 1).all(), \
        "Presence probabilities must be in [0, 1]"
    log.info("All assertions passed.")


if __name__ == "__main__":
    main()
