"""
04_model.py — Backbone wrappers for binary pure/mixed classification.

Designed so the backbone is swappable via config.yaml without changing
training code. The classifier head is replaced with a small dropout +
linear layer producing num_classes logits.

Currently supported backbones (via torchvision):
  - resnet50, resnet101
  - efficientnet_b0
  - densenet121

To add a new backbone, add a builder function and register it in BACKBONES.

Usage:
  model = build_model(cfg)
  model.freeze_backbone()        # train head only
  model.unfreeze_backbone()      # full fine-tuning
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

log = setup_logging("model")


# -----------------------------------------------------------------------------
# Backbone builders
# -----------------------------------------------------------------------------
def _build_resnet50(weights_str: str) -> tuple[nn.Module, int]:
    """Returns (backbone_with_identity_head, feature_dim)."""
    weights = getattr(models.ResNet50_Weights, weights_str)
    net = models.resnet50(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


def _build_resnet101(weights_str: str) -> tuple[nn.Module, int]:
    weights = getattr(models.ResNet101_Weights, weights_str)
    net = models.resnet101(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


def _build_efficientnet_b0(weights_str: str) -> tuple[nn.Module, int]:
    weights = getattr(models.EfficientNet_B0_Weights, weights_str)
    net = models.efficientnet_b0(weights=weights)
    feat_dim = net.classifier[-1].in_features
    net.classifier = nn.Identity()
    return net, feat_dim


def _build_densenet121(weights_str: str) -> tuple[nn.Module, int]:
    weights = getattr(models.DenseNet121_Weights, weights_str)
    net = models.densenet121(weights=weights)
    feat_dim = net.classifier.in_features
    net.classifier = nn.Identity()
    return net, feat_dim


BACKBONES: dict[str, Callable[[str], tuple[nn.Module, int]]] = {
    "resnet50": _build_resnet50,
    "resnet101": _build_resnet101,
    "efficientnet_b0": _build_efficientnet_b0,
    "densenet121": _build_densenet121,
}


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------
class StoneClassifier(nn.Module):
    """Backbone + classifier head wrapper.

    The backbone outputs a feature vector (its native pooled output). The
    head is a single Dropout + Linear producing class logits.

    Methods:
      freeze_backbone()    — sets backbone params to requires_grad=False
      unfreeze_backbone()  — sets all params to requires_grad=True
      head_parameters()    — returns just the head params (for separate optimizer LR)
      backbone_parameters()— returns just the backbone params
    """

    def __init__(self, backbone_name: str, weights: str, num_classes: int, dropout: float):
        super().__init__()
        if backbone_name not in BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(BACKBONES.keys())}"
            )
        self.backbone, feat_dim = BACKBONES[backbone_name](weights)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.backbone_name = backbone_name
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Lock backbone weights so only the head trains."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unlock all weights for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def head_parameters(self):
        return self.head.parameters()

    def backbone_parameters(self):
        return self.backbone.parameters()

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def build_model(cfg: dict) -> StoneClassifier:
    """Build a StoneClassifier from the config dict."""
    m = cfg["model"]
    model = StoneClassifier(
        backbone_name=m["backbone"],
        weights=m["pretrained_weights"],
        num_classes=m["num_classes"],
        dropout=m["dropout"],
    )
    log.info(
        f"Built {m['backbone']} ({model.total_param_count():,} params total). "
        f"Pretrained weights: {m['pretrained_weights']}"
    )
    return model


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_config
    cfg = load_config()

    model = build_model(cfg)
    log.info(f"Total params:     {model.total_param_count():,}")
    log.info(f"Trainable params: {model.trainable_param_count():,} (head + backbone)")

    model.freeze_backbone()
    log.info(f"After freeze:     {model.trainable_param_count():,} trainable (head only)")

    model.unfreeze_backbone()
    log.info(f"After unfreeze:   {model.trainable_param_count():,} trainable (head + backbone)")

    # Forward pass with dummy input
    x = torch.randn(2, 3, cfg["image"]["input_size"], cfg["image"]["input_size"])
    out = model(x)
    log.info(f"Forward pass OK: input {tuple(x.shape)} → output {tuple(out.shape)}")


if __name__ == "__main__":
    main()
