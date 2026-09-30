"""Model definition for Member C's from-scratch experiment."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def build_scratch_resnet18(num_classes: int = 500) -> nn.Module:
    """Build ResNet18 with random initialization and a new classifier head.

    Important: weights=None is what makes this a genuine from-scratch model.
    Do not replace it with ImageNet weights in Member C's experiment.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
