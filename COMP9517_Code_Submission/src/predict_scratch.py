"""Generate clean and robustness predictions for Member C's scratch model.

Run from the project root:

    python src/predict_scratch.py \
        --checkpoint results/models/scratch_aug_best.pth \
        --mode all

Outputs:
    results/clean/scratch_predictions.csv
    results/robustness/predictions/scratch_{degradation}_{severity}_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import build_transform
from degradations import DEGRADATION_LEVELS, apply_degradation
from scratch_model import build_scratch_resnet18


PROJECT_ROOT = Path.cwd()


def check_project_root() -> None:
    required = [
        PROJECT_ROOT / "data" / "splits" / "test_paths.csv",
        PROJECT_ROOT / "src" / "dataset.py",
        PROJECT_ROOT / "src" / "degradations.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the project root. Missing:\n  - "
            + "\n  - ".join(missing)
        )


def resolve_image_path(raw_path: str) -> Path:
    original = Path(raw_path)
    if original.exists():
        return original

    converted = Path(raw_path.replace("\\", os.sep).replace("/", os.sep))
    if converted.exists():
        return converted

    raise FileNotFoundError(
        "Could not open image. Check data/raw placement and nested folders:\n"
        f"  CSV path: {raw_path}\n"
        f"  Tried: {converted}"
    )


class PredictionDataset(Dataset):
    """Read test images, optionally degrade them, and retain CSV image paths."""

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int,
        degradation: Optional[str] = None,
        severity=None,
    ):
        self.df = pd.read_csv(csv_path)
        required = {"image_path", "class_idx"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        self.transform = build_transform("test", image_size=image_size, augment=False)
        self.degradation = degradation
        self.severity = severity

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        original_csv_path = str(row["image_path"])
        actual_path = resolve_image_path(original_csv_path)

        with Image.open(actual_path) as image_file:
            image = image_file.convert("RGB")

        if self.degradation is not None:
            image = apply_degradation(image, self.degradation, self.severity)

        image = self.transform(image)
        label = int(row["class_idx"])
        return image, label, original_csv_path


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def predict_variant(
    model: torch.nn.Module,
    device: torch.device,
    test_csv: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    out_csv: Path,
    degradation: Optional[str] = None,
    severity=None,
) -> float:
    dataset = PredictionDataset(
        csv_path=test_csv,
        image_size=image_size,
        degradation=degradation,
        severity=severity,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    model.eval()
    start = time.time()

    for batch_index, (images, labels, image_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.cpu()
        pred = top5[:, 0]
        labels = labels.cpu()

        for path, true_idx, pred_idx, top5_row in zip(
            image_paths,
            labels.tolist(),
            pred.tolist(),
            top5.tolist(),
        ):
            rows.append(
                {
                    "image_path": path,
                    "true_idx": int(true_idx),
                    "pred_idx": int(pred_idx),
                    "top5_idx": " ".join(str(int(value)) for value in top5_row),
                }
            )

        if batch_index % 50 == 0:
            print(f"    processed {min(batch_index * batch_size, len(dataset))}/{len(dataset)}")

    elapsed = time.time() - start
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} predictions: {out_csv}")
    print(f"Elapsed: {elapsed:.1f}s")
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", default="data/splits/test_paths.csv")
    parser.add_argument("--mode", choices=["clean", "robustness", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_project_root()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = safe_torch_load(checkpoint_path, device)

    num_classes = int(checkpoint.get("num_classes", 500))
    checkpoint_image_size = int(checkpoint.get("image_size", 224))
    image_size = args.image_size or checkpoint_image_size

    model = build_scratch_resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Image size: {image_size}")

    test_csv = Path(args.test_csv)
    timing = {}

    if args.mode in {"clean", "all"}:
        timing["clean_sec"] = predict_variant(
            model=model,
            device=device,
            test_csv=test_csv,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            out_csv=PROJECT_ROOT / "results" / "clean" / "scratch_predictions.csv",
        )

    if args.mode in {"robustness", "all"}:
        for degradation, severities in DEGRADATION_LEVELS.items():
            for severity in severities:
                variant = f"{degradation}_{severity}"
                print(f"\nRobustness variant: {variant}")
                out_csv = (
                    PROJECT_ROOT
                    / "results"
                    / "robustness"
                    / "predictions"
                    / f"scratch_{variant}_predictions.csv"
                )
                timing[f"{variant}_sec"] = predict_variant(
                    model=model,
                    device=device,
                    test_csv=test_csv,
                    image_size=image_size,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    out_csv=out_csv,
                    degradation=degradation,
                    severity=severity,
                )

    timing["total_sec"] = sum(timing.values())
    timing_path = PROJECT_ROOT / "results" / "logs" / "scratch_prediction_timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"\nSaved timing: {timing_path}")


if __name__ == "__main__":
    main()
