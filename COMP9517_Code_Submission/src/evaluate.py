"""Evaluate model prediction CSVs with unified metrics.

Expected input CSV columns:
- image_path
- true_idx
- pred_idx
- top5_idx optional, as space-separated class indices, e.g. "12 4 7 9 3"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from metrics import evaluate_classification, save_metrics


def parse_top5(value) -> List[int]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [int(x) for x in value.replace(",", " ").split() if x.strip()]
    return []


def evaluate_prediction_csv(pred_csv: Path, out_json: Path, num_classes: int | None) -> None:
    df = pd.read_csv(pred_csv)
    required = {"true_idx", "pred_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {pred_csv}: {sorted(missing)}")

    y_true = df["true_idx"].astype(int).tolist()
    y_pred = df["pred_idx"].astype(int).tolist()
    y_top5 = None
    if "top5_idx" in df.columns:
        y_top5 = [parse_top5(v) for v in df["top5_idx"].tolist()]

    metrics = evaluate_classification(y_true, y_pred, y_top5=y_top5, num_classes=num_classes)
    save_metrics(metrics, out_json)
    print(f"Saved metrics to {out_json}")
    print({k: v for k, v in metrics.items() if k not in {"confusion_matrix", "per_class_accuracy"}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a prediction CSV.")
    parser.add_argument("--pred-csv", required=True, help="Path to prediction CSV.")
    parser.add_argument("--out-json", required=True, help="Where to save metrics JSON.")
    parser.add_argument("--num-classes", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_prediction_csv(Path(args.pred_csv), Path(args.out_json), args.num_classes)
