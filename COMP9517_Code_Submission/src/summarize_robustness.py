"""Summarize clean and robustness metrics into one CSV.

This script reads metrics JSON files produced by src/evaluate.py and creates a
single robustness summary table.

Expected clean metrics files:
    results/clean/traditional_metrics.json
    results/clean/scratch_metrics.json
    results/clean/pretrained_metrics.json

Expected robustness metrics files:
    results/robustness/metrics/{model}_{degradation}_{severity}_metrics.json

Example:
    python src/summarize_robustness.py \
        --clean-dir results/clean \
        --metrics-dir results/robustness/metrics \
        --out-csv results/robustness/combined_robustness_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

KNOWN_MODELS = {"traditional", "scratch", "pretrained"}


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_robustness_name(path: Path) -> Optional[Tuple[str, str, str]]:
    """Parse {model}_{degradation}_{severity}_metrics.json.

    The degradation part may contain underscores, but the current project uses
    simple names such as noise, blur, brightness, contrast, and jpeg.
    """
    name = path.name
    suffix = "_metrics.json"
    if not name.endswith(suffix):
        return None

    stem = name[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    model = parts[0]
    if model not in KNOWN_MODELS:
        return None

    severity = parts[-1]
    degradation = "_".join(parts[1:-1])
    return model, degradation, severity


def add_row(rows: List[Dict[str, Any]], model: str, degradation: str, severity: str, metrics_path: Path) -> None:
    metrics = load_metrics(metrics_path)
    rows.append(
        {
            "model": model,
            "degradation": degradation,
            "severity": severity,
            "top1_accuracy": get_metric(metrics, "top1_accuracy"),
            "top5_accuracy": get_metric(metrics, "top5_accuracy"),
            "macro_precision": get_metric(metrics, "macro_precision"),
            "macro_recall": get_metric(metrics, "macro_recall"),
            "macro_f1": get_metric(metrics, "macro_f1"),
            "metrics_json": str(metrics_path).replace("\\", "/"),
        }
    )


def summarize(clean_dir: Path, metrics_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Add clean metrics if available.
    for model in sorted(KNOWN_MODELS):
        clean_metrics = clean_dir / f"{model}_metrics.json"
        if clean_metrics.exists():
            add_row(rows, model=model, degradation="clean", severity="0", metrics_path=clean_metrics)

    # Add robustness metrics if available.
    if metrics_dir.exists():
        for metrics_path in sorted(metrics_dir.glob("*_metrics.json")):
            parsed = parse_robustness_name(metrics_path)
            if parsed is None:
                print(f"[skip] Could not parse metrics filename: {metrics_path}")
                continue
            model, degradation, severity = parsed
            add_row(rows, model=model, degradation=degradation, severity=severity, metrics_path=metrics_path)

    if not rows:
        raise FileNotFoundError(
            "No metrics JSON files found. Please run src/evaluate.py first for clean and/or robustness predictions."
        )

    df = pd.DataFrame(rows)

    # Sort for readability.
    model_order = {"traditional": 0, "scratch": 1, "pretrained": 2}
    degradation_order = {"clean": 0, "noise": 1, "blur": 2, "brightness": 3, "contrast": 4, "jpeg": 5}
    df["_model_order"] = df["model"].map(model_order).fillna(99)
    df["_degradation_order"] = df["degradation"].map(degradation_order).fillna(99)
    df["_severity_numeric"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
    df = df.sort_values(["_model_order", "_degradation_order", "_severity_numeric"])
    df = df.drop(columns=["_model_order", "_degradation_order", "_severity_numeric"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize robustness metrics into one CSV.")
    parser.add_argument("--clean-dir", type=Path, default=Path("results/clean"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/robustness/metrics"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/robustness/combined_robustness_results.csv"))
    args = parser.parse_args()

    df = summarize(args.clean_dir, args.metrics_dir, args.out_csv)
    print(f"Saved summary to: {args.out_csv}")
    print(df[["model", "degradation", "severity", "top1_accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
