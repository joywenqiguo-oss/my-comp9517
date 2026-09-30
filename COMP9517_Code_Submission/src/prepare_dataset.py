"""Prepare reproducible train/validation/test CSV files for the COMP9517 project.

This script assumes iNaturalist-style COCO JSON files with at least:
- images: [{"id": ..., "file_name": ...}, ...]
- annotations: [{"image_id": ..., "category_id": ...}, ...]
- categories: [{"id": ..., "name": ...}, ...]

Training images are split into train and validation subsets.
Official validation images are used as the held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_lookup(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(img["id"]): img for img in data.get("images", [])}


def build_category_lookup(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(cat["id"]): cat for cat in data.get("categories", [])}


def group_images_by_category(data: Dict[str, Any]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        grouped[category_id].append(image_id)
    return grouped


def category_label(cat: Dict[str, Any], category_id: int) -> str:
    """Return a readable category label from an iNat/COCO category entry."""
    for key in ("name", "scientific_name", "common_name"):
        value = cat.get(key)
        if value:
            return str(value)
    return str(category_id)


def image_path(root: Path, img_entry: Dict[str, Any]) -> str:
    return str(root / str(img_entry["file_name"]))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "image_path",
        "image_id",
        "category_id",
        "label",
        "class_idx",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_rows(
    image_ids: List[int],
    image_lookup: Dict[int, Dict[str, Any]],
    image_root: Path,
    category_id: int,
    label: str,
    class_idx: int,
    split: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image_id in image_ids:
        img = image_lookup[image_id]
        rows.append(
            {
                "image_path": image_path(image_root, img),
                "image_id": image_id,
                "category_id": category_id,
                "label": label,
                "class_idx": class_idx,
                "split": split,
            }
        )
    return rows


def prepare_splits(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    train_json = load_json(Path(args.train_json))
    val_json = load_json(Path(args.val_json))

    train_images = build_image_lookup(train_json)
    val_images = build_image_lookup(val_json)
    train_categories = build_category_lookup(train_json)

    train_by_cat = group_images_by_category(train_json)
    val_by_cat = group_images_by_category(val_json)

    eligible_categories = []
    for category_id, train_ids in train_by_cat.items():
        val_ids = val_by_cat.get(category_id, [])
        if len(train_ids) >= args.train_per_class + args.val_per_class and len(val_ids) >= args.test_per_class:
            eligible_categories.append(category_id)

    if len(eligible_categories) < args.num_classes:
        raise ValueError(
            f"Only {len(eligible_categories)} categories have enough images, "
            f"but --num-classes={args.num_classes} was requested. "
            "Try reducing num_classes or per-class counts."
        )

    selected_category_ids = sorted(rng.sample(eligible_categories, args.num_classes))
    class_to_idx = {str(category_id): idx for idx, category_id in enumerate(selected_category_ids)}
    idx_to_class = {str(idx): str(category_id) for category_id, idx in class_to_idx.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    selected_lines: List[str] = []

    for category_id in selected_category_ids:
        class_idx = class_to_idx[str(category_id)]
        cat = train_categories.get(category_id, {"id": category_id})
        label = category_label(cat, category_id)

        train_ids = list(train_by_cat[category_id])
        official_val_ids = list(val_by_cat[category_id])
        rng.shuffle(train_ids)
        rng.shuffle(official_val_ids)

        selected_train = train_ids[: args.train_per_class]
        selected_val = train_ids[args.train_per_class : args.train_per_class + args.val_per_class]
        selected_test = official_val_ids[: args.test_per_class]

        train_rows.extend(
            make_rows(
                selected_train,
                train_images,
                Path(args.train_root),
                category_id,
                label,
                class_idx,
                "train",
            )
        )
        val_rows.extend(
            make_rows(
                selected_val,
                train_images,
                Path(args.train_root),
                category_id,
                label,
                class_idx,
                "val",
            )
        )
        test_rows.extend(
            make_rows(
                selected_test,
                val_images,
                Path(args.val_root),
                category_id,
                label,
                class_idx,
                "test",
            )
        )
        selected_lines.append(f"{class_idx}\t{category_id}\t{label}")

    write_csv(out_dir / "train_paths.csv", train_rows)
    write_csv(out_dir / "val_paths.csv", val_rows)
    write_csv(out_dir / "test_paths.csv", test_rows)

    (out_dir / "selected_classes.txt").write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    (out_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")
    (out_dir / "idx_to_class.json").write_text(json.dumps(idx_to_class, indent=2), encoding="utf-8")

    summary = {
        "seed": args.seed,
        "num_classes": args.num_classes,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "test_per_class": args.test_per_class,
        "num_train_images": len(train_rows),
        "num_val_images": len(val_rows),
        "num_test_images": len(test_rows),
        "note": "Training and validation splits are sampled from train_mini. Test split is sampled from official validation images.",
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done. Generated split files in:", out_dir)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COMP9517 iNaturalist subset splits.")
    parser.add_argument("--train-json", required=True, help="Path to train_mini annotation JSON.")
    parser.add_argument("--val-json", required=True, help="Path to official validation annotation JSON.")
    parser.add_argument("--train-root", required=True, help="Root folder of train_mini images.")
    parser.add_argument("--val-root", required=True, help="Root folder of validation images.")
    parser.add_argument("--out-dir", default="data/splits", help="Output directory for split files.")
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9517)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_splits(parse_args())
