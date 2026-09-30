"""Small helper to plot a confusion matrix from a metrics JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--top-n", type=int, default=30, help="Plot only the first/top N classes for readability.")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    cm = np.asarray(metrics["confusion_matrix"])
    cm = cm[: args.top_n, : args.top_n]

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix Overview (first {args.top_n} classes)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved {args.out_png}")


if __name__ == "__main__":
    main()
