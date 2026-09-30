"""
Build the final comparison table for the three project models.

The table combines:
1. Clean test performance
2. Average robustness performance
3. Training time
4. Clean test prediction time

Outputs:
    results/final_comparison_table.csv
    results/figures/final_comparison_table.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = PROJECT_ROOT / "results" / "clean"
ROBUSTNESS_CSV = (
    PROJECT_ROOT
    / "results"
    / "robustness"
    / "combined_robustness_results.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_comparison_table.csv"
)

OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "final_comparison_table.png"
)


# ============================================================
# 2. Input files for each model
# ============================================================

MODEL_FILES = {
    "traditional": {
        "display_name": "Traditional",
        "clean_metrics": CLEAN_DIR / "traditional_metrics.json",
        "training_timing": (
            PROJECT_ROOT
            / "results"
            / "models"
            / "traditional_train_timing.json"
        ),
        "prediction_timing": (
            PROJECT_ROOT
            / "results"
            / "models"
            / "traditional_predict_timing.json"
        ),
        "training_time_key": "total_sec",
        "testing_time_key": "clean_encode_sec",
    },
    "scratch": {
        "display_name": "Scratch ResNet18",
        "clean_metrics": CLEAN_DIR / "scratch_metrics.json",
        "training_timing": (
            PROJECT_ROOT
            / "results"
            / "logs"
            / "scratch_aug_summary.json"
        ),
        "prediction_timing": (
            PROJECT_ROOT
            / "results"
            / "logs"
            / "scratch_clean_prediction_timing.json"
        ),
        "training_time_key": "total_training_time_sec",
        "testing_time_key": "clean_sec",
    },
    "pretrained": {
        "display_name": "Pretrained ResNet18",
        "clean_metrics": CLEAN_DIR / "pretrained_metrics.json",
        "training_timing": (
            PROJECT_ROOT
            / "results"
            / "logs"
            / "pretrained_finetune_summary.json"
        ),
        "prediction_timing": (
            PROJECT_ROOT
            / "results"
            / "logs"
            / "pretrained_prediction_timing.json"
        ),
        "training_time_key": "total_training_time_sec",
        "testing_time_key": "clean_sec",
    },
}


# ============================================================
# 3. Helper functions
# ============================================================

def load_json(path: Path) -> dict:
    """Load one JSON file and report a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_column(dataframe: pd.DataFrame, column_name: str) -> None:
    """Check that a required column exists in a CSV."""
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Missing column '{column_name}' in robustness CSV. "
            f"Available columns: {list(dataframe.columns)}"
        )


def calculate_robustness_summary(
    robustness_results: pd.DataFrame,
    model_name: str,
    clean_top1: float,
) -> tuple[float, float, float]:
    """
    Calculate average performance across all 15 degraded test conditions.

    Returns:
        mean_degraded_top1
        mean_degraded_macro_f1
        top1_retention
    """
    model_results = robustness_results[
        robustness_results["model"] == model_name
    ].copy()

    degraded_results = model_results[
        model_results["degradation"] != "clean"
    ].copy()

    if degraded_results.empty:
        raise ValueError(
            f"No degraded robustness rows found for model: {model_name}"
        )

    mean_degraded_top1 = float(
        degraded_results["top1_accuracy"].mean()
    )

    mean_degraded_macro_f1 = float(
        degraded_results["macro_f1"].mean()
    )

    if clean_top1 > 0:
        top1_retention = mean_degraded_top1 / clean_top1
    else:
        top1_retention = 0.0

    return (
        mean_degraded_top1,
        mean_degraded_macro_f1,
        top1_retention,
    )


def build_final_table() -> pd.DataFrame:
    """Read all input files and build the final model comparison table."""
    if not ROBUSTNESS_CSV.exists():
        raise FileNotFoundError(
            f"Robustness summary file not found: {ROBUSTNESS_CSV}"
        )

    robustness_results = pd.read_csv(ROBUSTNESS_CSV)

    for required_column in [
        "model",
        "degradation",
        "top1_accuracy",
        "macro_f1",
    ]:
        require_column(
            robustness_results,
            required_column,
        )

    table_rows = []

    for model_name, file_config in MODEL_FILES.items():
        # ----------------------------------------------------
        # Load clean test metrics
        # ----------------------------------------------------
        clean_metrics = load_json(
            file_config["clean_metrics"]
        )

        clean_top1 = float(
            clean_metrics["top1_accuracy"]
        )

        clean_top5 = float(
            clean_metrics["top5_accuracy"]
        )

        clean_macro_f1 = float(
            clean_metrics["macro_f1"]
        )

        # ----------------------------------------------------
        # Calculate average robustness metrics
        # ----------------------------------------------------
        (
            mean_degraded_top1,
            mean_degraded_macro_f1,
            top1_retention,
        ) = calculate_robustness_summary(
            robustness_results=robustness_results,
            model_name=model_name,
            clean_top1=clean_top1,
        )

        # ----------------------------------------------------
        # Load training and clean testing times
        # ----------------------------------------------------
        training_timing = load_json(
            file_config["training_timing"]
        )

        prediction_timing = load_json(
            file_config["prediction_timing"]
        )

        training_time_sec = float(
            training_timing[
                file_config["training_time_key"]
            ]
        )

        testing_time_sec = float(
            prediction_timing[
                file_config["testing_time_key"]
            ]
        )

        table_rows.append(
            {
                "Model": file_config["display_name"],
                "Clean Top-1": clean_top1,
                "Clean Top-5": clean_top5,
                "Clean Macro-F1": clean_macro_f1,
                "Mean Degraded Top-1": mean_degraded_top1,
                "Mean Degraded Macro-F1": mean_degraded_macro_f1,
                "Top-1 Retention": top1_retention,
                "Training Time (h)": training_time_sec / 3600,
                "Clean Testing Time (s)": testing_time_sec,
            }
        )

    return pd.DataFrame(table_rows)


def save_table_figure(
    comparison_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a report-ready PNG version of the final comparison table."""
    display_table = comparison_table.copy()

    percentage_columns = [
        "Clean Top-1",
        "Clean Top-5",
        "Clean Macro-F1",
        "Mean Degraded Top-1",
        "Mean Degraded Macro-F1",
        "Top-1 Retention",
    ]

    for column_name in percentage_columns:
        display_table[column_name] = display_table[
            column_name
        ].map(lambda value: f"{value * 100:.2f}%")

    display_table["Training Time (h)"] = display_table[
        "Training Time (h)"
    ].map(lambda value: f"{value:.2f}")

    display_table["Clean Testing Time (s)"] = display_table[
        "Clean Testing Time (s)"
    ].map(lambda value: f"{value:.2f}")

    # 将较长的表头换成两行，避免文字挤出单元格
    display_table = display_table.rename(
        columns={
            "Mean Degraded Top-1": "Mean Degraded\nTop-1",
            "Mean Degraded Macro-F1": "Mean Degraded\nMacro-F1",
            "Clean Testing Time (s)": "Clean Testing\nTime (s)",
        }
    )

    figure, axis = plt.subplots(
        figsize=(19, 4.2)
    )

    axis.axis("off")

    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.3)

    # Make the header row bold
    for column_index in range(
        len(display_table.columns)
    ):
        table[(0, column_index)].set_text_props(
            weight="bold"
        )

    axis.set_title(
        "Final Model Comparison",
        fontsize=16,
        pad=18,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 4. Main program
# ============================================================

def main() -> None:
    comparison_table = build_final_table()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_table.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    save_table_figure(
        comparison_table=comparison_table,
        output_path=OUTPUT_FIGURE,
    )

    print("\nFinal comparison table")
    print("=" * 120)

    printable_table = comparison_table.copy()

    percentage_columns = [
        "Clean Top-1",
        "Clean Top-5",
        "Clean Macro-F1",
        "Mean Degraded Top-1",
        "Mean Degraded Macro-F1",
        "Top-1 Retention",
    ]

    for column_name in percentage_columns:
        printable_table[column_name] = (
            printable_table[column_name] * 100
        ).round(2)

    printable_table["Training Time (h)"] = (
        printable_table["Training Time (h)"]
        .round(2)
    )

    printable_table["Clean Testing Time (s)"] = (
        printable_table["Clean Testing Time (s)"]
        .round(2)
    )

    print(printable_table.to_string(index=False))

    print("\nFinal comparison table completed.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()