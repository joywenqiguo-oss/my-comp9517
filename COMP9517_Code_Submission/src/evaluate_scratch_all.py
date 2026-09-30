"""Evaluate all clean and degraded prediction CSVs produced by Member C."""

from __future__ import annotations

from pathlib import Path

from degradations import DEGRADATION_LEVELS
from evaluate import evaluate_prediction_csv


def main() -> None:
    clean_prediction = Path("results/clean/scratch_predictions.csv")
    clean_metrics = Path("results/clean/scratch_metrics.json")

    if not clean_prediction.exists():
        raise FileNotFoundError(
            "Missing clean predictions. Run src/predict_scratch.py first."
        )

    evaluate_prediction_csv(clean_prediction, clean_metrics, num_classes=500)

    prediction_dir = Path("results/robustness/predictions")
    metrics_dir = Path("results/robustness/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for degradation, severities in DEGRADATION_LEVELS.items():
        for severity in severities:
            stem = f"scratch_{degradation}_{severity}"
            pred_path = prediction_dir / f"{stem}_predictions.csv"
            metrics_path = metrics_dir / f"{stem}_metrics.json"

            if not pred_path.exists():
                missing.append(str(pred_path))
                continue

            evaluate_prediction_csv(pred_path, metrics_path, num_classes=500)

    if missing:
        print("\nThe following robustness prediction files were not found:")
        for path in missing:
            print(f"  - {path}")
    else:
        print("\nAll 15 robustness variants were evaluated successfully.")


if __name__ == "__main__":
    main()
