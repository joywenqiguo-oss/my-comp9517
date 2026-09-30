"""PyTorch Dataset helpers for COMP9517 project CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

try:
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dataset.py requires torch and torchvision. Install requirements.txt first."
    ) from exc


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class INatCsvDataset(Dataset):
    """Dataset reading image_path and class_idx from a split CSV.

    The image paths stored in data/splits/*.csv are project-relative paths such
    as data/raw/train_mini/....  This dataset resolves them relative to the
    project root, so it works both from the project root and from notebooks/.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[object] = None,
        project_root: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path).resolve()
        self.df = pd.read_csv(self.csv_path)

        required = {"image_path", "class_idx"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        # If csv_path is PROJECT_ROOT/data/splits/train_paths.csv,
        # parents[2] is PROJECT_ROOT.
        if project_root is None:
            try:
                self.project_root = self.csv_path.parents[2]
            except IndexError:
                self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root).resolve()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["image_path"])

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {img_path}\n"
                f"CSV path: {self.csv_path}\n"
                f"Project root used: {self.project_root}\n"
                "Check that raw data is placed under data/raw/ and that there is no extra train_mini/train_mini or val/val layer."
            )

        image = Image.open(img_path).convert("RGB")
        label = int(row["class_idx"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transform(split: str, image_size: int = 224, augment: bool = True):
    """Create standard transforms for train/val/test."""
    if split == "train" and augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_loader(
    csv_path: str | Path,
    split: str,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    augment: bool = True,
    shuffle: Optional[bool] = None,
    project_root: str | Path | None = None,
):
    """Build a DataLoader from one of the project split CSV files."""
    if shuffle is None:
        shuffle = split == "train"

    dataset = INatCsvDataset(
        csv_path,
        transform=build_transform(split, image_size, augment),
        project_root=project_root,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
