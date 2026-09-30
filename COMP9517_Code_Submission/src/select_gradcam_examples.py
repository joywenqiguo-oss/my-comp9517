"""Select correct/incorrect and most-confused examples for Member E's Grad-CAM work.

Reads results/clean/pretrained_predictions.csv and data/splits/selected_classes.txt,
and writes:
    results/clean/pretrained_correct_examples.csv
    results/clean/pretrained_incorrect_examples.csv
    results/clean/pretrained_confused_pairs.csv

Run from the project root:

    python src/select_gradcam_examples.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_class_labels(selected_classes_path: Path) -> dict[int, str]:
    labels = {}
    with selected_classes_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            class_idx, _category_id, species_name = parts[0], parts[1], parts[2]
            labels[int(class_idx)] = species_name
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", default="results/clean/pretrained_predictions.csv")
    parser.add_argument("--selected-classes", default="data/splits/selected_classes.txt")
    parser.add_argument("--out-dir", default="results/clean")
    parser.add_argument("--num-correct", type=int, default=30)
    parser.add_argument("--num-incorrect", type=int, default=30)
    parser.add_argument("--num-confused-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9517)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    labels = load_class_labels(Path(args.selected_classes))

    df["true_label"] = df["true_idx"].map(labels)
    df["pred_label"] = df["pred_idx"].map(labels)
    df["correct"] = df["true_idx"] == df["pred_idx"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_df = df[df["correct"]].sample(
        n=min(args.num_correct, int(df["correct"].sum())), random_state=args.seed
    )
    incorrect_df = df[~df["correct"]].sample(
        n=min(args.num_incorrect, int((~df["correct"]).sum())), random_state=args.seed
    )

    cols = ["image_path", "true_idx", "true_label", "pred_idx", "pred_label", "top5_idx"]
    correct_df[cols].to_csv(out_dir / "pretrained_correct_examples.csv", index=False)
    incorrect_df[cols].to_csv(out_dir / "pretrained_incorrect_examples.csv", index=False)

    confused = (
        df[~df["correct"]]
        .groupby(["true_idx", "true_label", "pred_idx", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(args.num_confused_pairs)
    )
    confused.to_csv(out_dir / "pretrained_confused_pairs.csv", index=False)

    print(f"Saved {len(correct_df)} correct examples: {out_dir / 'pretrained_correct_examples.csv'}")
    print(f"Saved {len(incorrect_df)} incorrect examples: {out_dir / 'pretrained_incorrect_examples.csv'}")
    print(f"Saved top {len(confused)} confused pairs: {out_dir / 'pretrained_confused_pairs.csv'}")
    print("\nHardest confused pairs:")
    print(confused.to_string(index=False))


if __name__ == "__main__":
    main()
