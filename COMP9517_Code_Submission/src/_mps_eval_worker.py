"""Standalone validation worker, invoked as a subprocess by train_pretrained.py.

On this project's development machine (Apple Silicon, torch MPS backend),
calling a torch.no_grad()-scoped validation function in the SAME process
that just ran a training function (e.g. train_one_epoch) produces near-random
accuracy: predictions look like noise even though the trained weights are
correct (verified by reloading the exact same checkpoint in a fresh process,
which reliably gives the expected accuracy). Running validation in a fresh
subprocess, loading the checkpoint from disk, sidesteps this environment bug.
See PROJECT_TEAM_GUIDE_CN.md discussion / commit notes for the isolation
experiments that pinned this down. This is unrelated to model correctness.

Prints a single JSON line of metrics to stdout so the caller can parse it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from dataset import make_loader
from pretrained_model import build_pretrained_resnet18


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = safe_torch_load(Path(args.checkpoint), device)
    model = build_pretrained_resnet18(args.num_classes, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loader = make_loader(
        csv_path=args.val_csv,
        split="val",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    sample_count = 0
    top1_count = 0
    top5_count = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            k = min(5, logits.shape[1])
            topk = logits.topk(k=k, dim=1).indices
            predictions = topk[:, 0]

            batch_size = targets.size(0)
            top1_count += int((predictions == targets).sum().item())
            top5_count += int(topk.eq(targets.view(-1, 1)).any(dim=1).sum().item())
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            all_true.extend(targets.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "val_loss": loss_sum / sample_count,
        "val_top1": top1_count / sample_count,
        "val_top5": top5_count / sample_count,
        "val_macro_precision": float(precision),
        "val_macro_recall": float(recall),
        "val_macro_f1": float(macro_f1),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
