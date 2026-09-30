"""Model definition for Member D's transfer-learning experiment."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_pretrained_resnet18(num_classes: int = 500, freeze_backbone: bool = False) -> nn.Module:
    """Build ResNet18 initialised with ImageNet-pretrained weights.

    freeze_backbone=True freezes every pretrained layer and only trains the
    new classifier head (linear probe). freeze_backbone=False fine-tunes all
    layers starting from the pretrained weights.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
