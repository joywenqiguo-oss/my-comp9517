"""Create a one-row-per-run ablation summary from training summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Example: results/logs/scratch_no_aug_summary.json results/logs/scratch_aug_summary.json",
    )
    parser.add_argument(
        "--out-csv",
        default="results/logs/scratch_ablation_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    for summary_path in args.summaries:
        path = Path(summary_path)
        with path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        rows.append(
            {
                "run_name": summary["run_name"],
                "augmentation": summary["augment"],
                "best_epoch": summary["best_epoch"],
                "best_val_top1": summary["best_val_top1"],
                "best_val_top5": summary["best_val_top5"],
                "best_val_macro_f1": summary["best_val_macro_f1"],
                "training_time_sec": summary["total_training_time_sec"],
            }
        )

    df = pd.DataFrame(rows).sort_values("best_val_macro_f1", ascending=False)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
