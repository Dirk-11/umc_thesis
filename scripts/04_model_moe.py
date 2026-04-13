"""
04_model_moe.py — Mixture-of-Experts model for compositional regression (Model B).

Architecture:
  - Shared ResNet50 backbone (same structure as Model A in 04_model.py)
  - N parallel expert heads, one per final class (CaOx, CHP, UA, MAP, CYS, OTH)
  - Each head: Linear(feat_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1)
  - Forward pass returns raw logits (shape: batch × n_classes)
  - Apply softmax for probabilities, log_softmax for KL divergence loss

Optional warm-start: copy backbone weights from a Model A checkpoint so Model B
inherits visual features already tuned for kidney stones. Only backbone weights
are transferred; the expert heads are always freshly initialised.

Usage:
  model = build_moe_model(cfg)
  model.freeze_backbone()     # train heads only (Phase 1)
  model.unfreeze_backbone()   # full fine-tuning (Phase 2)

Run as script to verify model builds and forward-pass is valid:
  python scripts/04_model_moe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

log = setup_logging("model_moe")


# -----------------------------------------------------------------------------
# Backbone — reuse builders from 04_model.py
# -----------------------------------------------------------------------------
def _get_backbones():
    from importlib import import_module
    return import_module("04_model").BACKBONES


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
# MoE compositor model
# -----------------------------------------------------------------------------
class MoEStoneCompositor(nn.Module):
    """Shared backbone + parallel expert heads for compositional regression.

    Forward pass:
      1. Backbone extracts a feature vector (batch, feat_dim)
      2. Each expert head independently maps features → scalar logit
      3. Cat all logits → (batch, n_classes) raw logits
      4. Caller applies softmax (inference) or log_softmax (KL loss training)

    The softmax over the n_classes logits enforces the simplex constraint:
    predicted composition sums to 1 with all components ≥ 0.
    """

    def __init__(
        self,
        backbone_name: str,
        weights: str,
        class_names: list[str],
        hidden_dim: int,
        dropout: float,
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

        self.experts = nn.ModuleList([
            ExpertHead(feat_dim, hidden_dim, dropout)
            for _ in class_names
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, n_classes). Apply softmax externally."""
        features = self.backbone(x)                                    # (B, feat_dim)
        logits = torch.cat([h(features) for h in self.experts], dim=1)  # (B, n_classes)
        return logits

    def predict_composition(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return softmax probabilities summing to 1."""
        return F.softmax(self.forward(x), dim=1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def head_parameters(self):
        return self.experts.parameters()

    def backbone_parameters(self):
        return self.backbone.parameters()

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Factory + warm-start
# -----------------------------------------------------------------------------
def build_moe_model(cfg: dict) -> MoEStoneCompositor:
    """Build MoEStoneCompositor from config, optionally warm-starting backbone."""
    m = cfg["model_moe"]
    classes = cfg["class_remapping"]["final_classes"]

    model = MoEStoneCompositor(
        backbone_name=m["backbone"],
        weights=m["pretrained_weights"],
        class_names=classes,
        hidden_dim=m["head_hidden_dim"],
        dropout=m["head_dropout"],
    )
    log.info(
        f"Built MoE model: {m['backbone']} backbone + "
        f"{len(classes)} expert heads {classes} "
        f"({model.total_param_count():,} params total)"
    )

    warm_start = m.get("warm_start_from")
    if warm_start:
        load_backbone_from_m1_checkpoint(model, warm_start)

    return model


def load_backbone_from_m1_checkpoint(
    model: MoEStoneCompositor,
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> None:
    """Copy backbone weights from a Model A (StoneClassifier) checkpoint.

    Only backbone.* keys are transferred; expert heads are left as-is.
    This lets Model B inherit visual features already tuned for kidney stones.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    # Extract backbone keys, strip the "backbone." prefix
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
    import torch

    cfg = load_config()
    model = build_moe_model(cfg)

    log.info(f"Total params:     {model.total_param_count():,}")
    log.info(f"Trainable params: {model.trainable_param_count():,}")

    model.freeze_backbone()
    log.info(f"After freeze:     {model.trainable_param_count():,} trainable (heads only)")

    model.unfreeze_backbone()
    log.info(f"After unfreeze:   {model.trainable_param_count():,} trainable (all)")

    # Forward pass with dummy input
    size = cfg["image"]["input_size"]
    x = torch.randn(4, 3, size, size)
    logits = model(x)
    probs = model.predict_composition(x)

    log.info(f"Forward pass OK: input {tuple(x.shape)} → logits {tuple(logits.shape)}")
    log.info(f"Composition probs row sums: {probs.sum(dim=1).tolist()}")
    assert probs.shape == (4, len(cfg["class_remapping"]["final_classes"]))
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5), \
        "Composition probabilities must sum to 1"
    log.info("All assertions passed.")


if __name__ == "__main__":
    main()
