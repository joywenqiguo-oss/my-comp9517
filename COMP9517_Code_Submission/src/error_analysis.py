from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# 1. 路径和分析参数
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILES = {
    "traditional": PROJECT_ROOT / "results/clean/traditional_predictions.csv",
    "scratch": PROJECT_ROOT / "results/clean/scratch_predictions.csv",
    "pretrained": PROJECT_ROOT / "results/clean/pretrained_predictions.csv",
}

# CSV 和分析数据的输出目录
ANALYSIS_OUTPUT_ROOT = PROJECT_ROOT / "results/error_analysis"

# 所有生成图片的输出目录
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "results/figures/error_analysis"

# 每类案例最多导出 20 条代表样本；完整统计仍使用全部预测结果
NUM_EXAMPLES = 20

# confusion matrix 展示错误最多的类别数量
NUM_CONFUSION_CLASSES = 10

# 高置信度错误和正确样本使用的阈值
HIGH_CONFIDENCE_THRESHOLD = 0.80

# difficult but correct：预测正确但置信度较低
DIFFICULT_CORRECT_THRESHOLD = 0.50

# 最常见混淆类别组合的数量
NUM_CONFUSED_PAIRS = 10


# ============================================================
# 2. 辅助函数
# ============================================================

def parse_top5(top5_value):
    """
    将 CSV 中的 Top-5 字符串转换成整数列表。

    例如：
        "433 470 434 469 406"
    转换为：
        [433, 470, 434, 469, 406]
    """
    return [int(class_idx) for class_idx in str(top5_value).split()]


def extract_species_name(image_path):
    """
    从图片所在文件夹名称中提取物种名称。

    例如：
        00044_Animalia_Arthropoda_..._Trichonephila_clavata

    返回：
        Trichonephila clavata
    """
    normalised_path = str(image_path).replace("\\", "/")
    folder_name = normalised_path.split("/")[-2]
    name_parts = folder_name.split("_")

    return " ".join(name_parts[-2:])


def prepare_predictions(predictions):
    """
    为预测数据增加 Error Analysis 所需的辅助列。
    """
    predictions = predictions.copy()

    # 将主要数值列转换成统一的数据类型
    predictions["true_idx"] = predictions["true_idx"].astype(int)
    predictions["pred_idx"] = predictions["pred_idx"].astype(int)
    predictions["top1_confidence"] = predictions["top1_confidence"].astype(float)

    # 将 Top-5 字符串转换成整数列表
    predictions["top5_list"] = predictions["top5_idx"].apply(parse_top5)

    # 判断 Top-1 和 Top-5 是否预测正确
    predictions["top1_correct"] = predictions["true_idx"] == predictions["pred_idx"]
    predictions["top5_correct"] = predictions.apply(
        lambda row: row["true_idx"] in row["top5_list"],
        axis=1,
    )

    # 从图片路径中提取真实物种名称
    predictions["true_species"] = predictions["image_path"].apply(extract_species_name)

    return predictions


def save_case_samples(samples, output_path, num_examples=NUM_EXAMPLES):
    """
    按当前排序保存指定数量的代表性案例。
    """
    columns_to_save = [
        "image_path",
        "true_idx",
        "pred_idx",
        "top5_idx",
        "top1_confidence",
        "true_species",
        "top1_correct",
        "top5_correct",
    ]

    samples.head(num_examples)[columns_to_save].to_csv(output_path, index=False)

def resolve_image_path(image_path):
    """
    将预测 CSV 中的图片路径转换成本地可读取的路径。

    CSV 可能使用 Windows 风格的反斜杠，
    因此统一转换成当前项目下的相对路径。
    """
    normalised_path = str(image_path).replace("\\", "/")
    resolved_path = PROJECT_ROOT / normalised_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Cannot find image for example gallery:\n"
            f"{resolved_path}"
        )

    return resolved_path


def plot_example_gallery(
    model_name,
    case_groups,
    class_name_map,
    output_path,
    figure_title,
    num_examples_per_case=2,
):
    """
    为成功或失败案例生成原图总览。

    每种案例默认展示 2 张代表图片。
    如果某类案例为空，则自动跳过该类别。
    """
    non_empty_groups = [
        (case_label, samples)
        for case_label, samples in case_groups
        if not samples.empty
    ]

    if not non_empty_groups:
        print(
            f"[Warning] No samples available for gallery: "
            f"{figure_title}"
        )
        return

    num_rows = len(non_empty_groups)
    num_columns = num_examples_per_case

    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(6 * num_columns, 5.5 * num_rows),
    )

    # 保证 axes 始终是二维数组，方便统一索引
    axes = np.asarray(axes)

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    if num_columns == 1:
        axes = axes.reshape(-1, 1)

    # for row_index, (case_label, samples) in enumerate(
    #     non_empty_groups
    # ):
    #     selected_samples = samples.head(
    #         num_examples_per_case
    #     )

    used_image_paths = set()

    for row_index, (case_label, samples) in enumerate(
        non_empty_groups
    ):
        # 排除已经在前面类别中展示过的图片，避免 gallery 内重复
        available_samples = samples[
            ~samples["image_path"].astype(str).isin(used_image_paths)
        ]

        selected_samples = available_samples.head(
            num_examples_per_case
        )

        used_image_paths.update(
            selected_samples["image_path"].astype(str).tolist()
        )

        for column_index in range(num_columns):
            axis = axes[row_index, column_index]
            axis.axis("off")

            if column_index >= len(selected_samples):
                continue

            sample = selected_samples.iloc[column_index]

            image_path = resolve_image_path(
                sample["image_path"]
            )

            with Image.open(image_path) as image_file:
                original_image = image_file.convert("RGB")

            true_idx = int(sample["true_idx"])
            pred_idx = int(sample["pred_idx"])

            true_species = str(sample["true_species"])
            pred_species = class_name_map.get(
                pred_idx,
                f"class_{pred_idx}",
            )

            confidence = float(
                sample["top1_confidence"]
            )

            axis.imshow(original_image)

            axis.set_title(
                f"{case_label}\n"
                f"True: {true_species}\n"
                f"Predicted: {pred_species}\n"
                f"Confidence: {confidence:.4f}",
                fontsize=10,
                pad=12,
            )

    figure.suptitle(
        f"{model_name.capitalize()}: {figure_title}",
        fontsize=15,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96],
        h_pad=4.0,
        w_pad=2.0,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

# ============================================================
# 3. 单个模型的案例分析
# ============================================================

def analyse_prediction_cases( model_name, predictions, model_output_dir, model_figure_dir):
    """
    提取并保存一个模型的不同类型预测案例。
    """

    # --------------------------------------------------------
    # Top-1 正确样本
    # 优先保留置信度最高、最有代表性的正确预测
    # --------------------------------------------------------
    correct_samples = predictions[predictions["top1_correct"]].sort_values(
        "top1_confidence",
        ascending=False,
    )

    save_case_samples(
        correct_samples,
        model_output_dir / "top1_correct.csv",
    )

    # --------------------------------------------------------
    # Top-1 错误样本
    # 优先保留置信度最高的错误预测
    # --------------------------------------------------------
    wrong_samples = predictions[~predictions["top1_correct"]].sort_values(
        "top1_confidence",
        ascending=False,
    )

    save_case_samples(
        wrong_samples,
        model_output_dir / "top1_wrong.csv",
    )

    # --------------------------------------------------------
    # Top-1 错误但 Top-5 正确
    # 说明正确类别已经进入候选范围，但排序不是第一名
    # --------------------------------------------------------
    top1_wrong_top5_correct = predictions[
        (~predictions["top1_correct"]) & predictions["top5_correct"]
    ].sort_values(
        "top1_confidence",
        ascending=False,
    )

    save_case_samples(
        top1_wrong_top5_correct,
        model_output_dir / "top1_wrong_top5_correct.csv",
    )

    # --------------------------------------------------------
    # Difficult but correct
    # 模型最终预测正确，但置信度较低
    # --------------------------------------------------------
    difficult_correct = predictions[
        predictions["top1_correct"]
        & (predictions["top1_confidence"] < DIFFICULT_CORRECT_THRESHOLD)
    ].sort_values(
        "top1_confidence",
        ascending=True,
    )

    save_case_samples(
        difficult_correct,
        model_output_dir / "difficult_but_correct.csv",
    )

    # --------------------------------------------------------
    # High-confidence correct
    # 模型预测正确且非常有把握
    # --------------------------------------------------------
    high_confidence_correct = predictions[
        predictions["top1_correct"]
        & (predictions["top1_confidence"] >= HIGH_CONFIDENCE_THRESHOLD)
    ].sort_values(
        "top1_confidence",
        ascending=False,
    )

    save_case_samples(
        high_confidence_correct,
        model_output_dir / "high_confidence_correct.csv",
    )

    # --------------------------------------------------------
    # High-confidence wrong
    # 最值得分析的失败案例：模型很有把握但预测错误
    # --------------------------------------------------------
    high_confidence_wrong = predictions[
        (~predictions["top1_correct"])
        & (predictions["top1_confidence"] >= HIGH_CONFIDENCE_THRESHOLD)
    ].sort_values(
        "top1_confidence",
        ascending=False,
    )

    save_case_samples(
        high_confidence_wrong,
        model_output_dir / "high_confidence_wrong.csv",
    )

    # 保存各类案例的数量，便于之后写报告
    case_summary = pd.DataFrame(
        {
            "case_type": [
                "top1_correct",
                "top1_wrong",
                "top1_wrong_top5_correct",
                "difficult_but_correct",
                "high_confidence_correct",
                "high_confidence_wrong",
            ],
            "count": [
                len(correct_samples),
                len(wrong_samples),
                len(top1_wrong_top5_correct),
                len(difficult_correct),
                len(high_confidence_correct),
                len(high_confidence_wrong),
            ],
        }
    )

    case_summary.to_csv(model_output_dir / "case_summary.csv", index=False)

    print(f"\n[{model_name}] Case analysis")
    print(case_summary.to_string(index=False))

    # 建立类别编号到物种名称的映射，
    # 用于在 gallery 中显示预测物种名称
    class_name_map = (
        predictions[["true_idx", "true_species"]]
        .drop_duplicates("true_idx")
        .set_index("true_idx")["true_species"]
        .to_dict()
    )

    # --------------------------------------------------------
    # 成功案例原图总览
    # --------------------------------------------------------
    success_case_groups = [
        ("Top-1 Correct", correct_samples),
        (
            "High-confidence Correct",
            high_confidence_correct,
        ),
        (
            "Difficult but Correct",
            difficult_correct,
        ),
    ]

    plot_example_gallery(
        model_name=model_name,
        case_groups=success_case_groups,
        class_name_map=class_name_map,
        output_path=(
            model_figure_dir
            / "successful_examples_gallery.png"
        ),
        figure_title="Successful Prediction Examples",
        num_examples_per_case=2,
    )

    # --------------------------------------------------------
    # 失败案例原图总览
    # --------------------------------------------------------
    failure_case_groups = [
        ("Top-1 Wrong", wrong_samples),
        (
            "Top-1 Wrong but Top-5 Correct",
            top1_wrong_top5_correct,
        ),
        (
            "High-confidence Wrong",
            high_confidence_wrong,
        ),
    ]

    plot_example_gallery(
        model_name=model_name,
        case_groups=failure_case_groups,
        class_name_map=class_name_map,
        output_path=(
            model_figure_dir
            / "failure_examples_gallery.png"
        ),
        figure_title="Failure Prediction Examples",
        num_examples_per_case=2,
    )


# ============================================================
# 4. 最常混淆的类别组合
# ============================================================

def analyse_confused_pairs(model_name, predictions, model_output_dir, model_figure_dir):
    """
    统计最常见的真实类别和错误预测类别组合，
    并同时保存类别编号、物种名称和错误次数。
    """

    # 只分析 Top-1 错误的样本
    wrong_predictions = predictions[~predictions["top1_correct"]].copy()

    # 建立类别编号到物种名称的映射
    class_name_map = (
        predictions[["true_idx", "true_species"]]
        .drop_duplicates("true_idx")
        .set_index("true_idx")["true_species"]
        .to_dict()
    )

    # 统计每一种 true class → predicted class 错误组合的次数
    confused_pairs = (
        wrong_predictions.groupby(["true_idx", "pred_idx"])
        .size()
        .reset_index(name="error_count")
        .sort_values(
            ["error_count", "true_idx", "pred_idx"],
            ascending=[False, True, True],
        )
    )

    # 加入真实类别和预测类别的物种名称
    confused_pairs["true_species"] = confused_pairs["true_idx"].map(class_name_map)
    confused_pairs["pred_species"] = confused_pairs["pred_idx"].map(class_name_map)

    # 计算该错误占真实类别全部样本的比例
    true_class_counts = predictions["true_idx"].value_counts()

    confused_pairs["error_rate_within_true_class"] = (
        confused_pairs["error_count"]
        / confused_pairs["true_idx"].map(true_class_counts)
    )

    # 只保留错误次数最高的若干类别组合
    top_pairs = confused_pairs.head(NUM_CONFUSED_PAIRS).copy()

    # 保存 CSV
    top_pairs.to_csv(
        model_output_dir / "most_confused_class_pairs.csv",
        index=False,
    )

    # 图中同时展示类别编号和物种名称
    top_pairs["pair_label"] = (
        top_pairs["true_idx"].astype(str)
        + ": "
        + top_pairs["true_species"]
        + " → "
        + top_pairs["pred_idx"].astype(str)
        + ": "
        + top_pairs["pred_species"]
    )

    # 绘制最常见混淆类别组合
    plt.figure(figsize=(14, 8))
    plt.barh(
        top_pairs["pair_label"][::-1],
        top_pairs["error_count"][::-1],
    )

    plt.xlabel("Number of Misclassified Images")
    plt.ylabel("True Class → Predicted Class")
    plt.title(f"{model_name.capitalize()}: Most Frequent Confusion Pairs")
    plt.tight_layout()

    plt.savefig(
        model_figure_dir / "most_confused_class_pairs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 5. Selected Confusion Matrix
# ============================================================

def plot_selected_confusion_matrix(
    model_name,
    predictions,
    model_output_dir,
    model_figure_dir,
):
    """
    选择错误数量最多的真实类别，并绘制归一化 confusion matrix。

    对于不在主要显示类别中的预测，将其归入 Other，
    从而避免错误样本被直接丢弃。
    """

    # 统计每一个真实类别的 Top-1 错误数量
    class_error_counts = (
        predictions[~predictions["top1_correct"]]
        .groupby("true_idx")
        .size()
        .sort_values(ascending=False)
    )

    # 选择错误数量最多的真实类别
    selected_true_classes = (
        class_error_counts
        .head(NUM_CONFUSION_CLASSES)
        .index
        .tolist()
    )

    # 保留这些真实类别的全部样本，包括正确和错误预测
    selected_predictions = predictions[
        predictions["true_idx"].isin(selected_true_classes)
    ].copy()

    # 找出这些真实类别最常被错误预测成的类别
    top_error_targets = (
        selected_predictions[~selected_predictions["top1_correct"]]
        ["pred_idx"]
        .value_counts()
        .head(NUM_CONFUSION_CLASSES)
        .index
        .tolist()
    )

    # 显示列必须包括真实类别本身，以便保留对角线上的正确预测
    display_pred_classes = list(
        dict.fromkeys(selected_true_classes + top_error_targets)
    )

    # 不在主要显示类别中的预测统一归为 Other
    selected_predictions["predicted_group"] = selected_predictions[
        "pred_idx"
    ].apply(
        lambda pred_idx: str(pred_idx)
        if pred_idx in display_pred_classes
        else "Other"
    )

    row_labels = [str(class_idx) for class_idx in selected_true_classes]
    column_labels = [
        str(class_idx) for class_idx in display_pred_classes
    ] + ["Other"]

    # 生成完整计数矩阵
    matrix_counts = pd.crosstab(
        selected_predictions["true_idx"].astype(str),
        selected_predictions["predicted_group"],
    ).reindex(
        index=row_labels,
        columns=column_labels,
        fill_value=0,
    )

    # 每一行除以该真实类别的全部样本数量
    normalised_matrix = (
        matrix_counts
        .div(matrix_counts.sum(axis=1), axis=0)
        .fillna(0)
    )

    # 保存原始计数矩阵
    matrix_counts.to_csv(
        model_output_dir / "selected_confusion_matrix_counts.csv"
    )

    # 保存归一化矩阵
    normalised_matrix.to_csv(
        model_output_dir / "selected_confusion_matrix_normalised.csv"
    )

    # 绘制归一化 confusion matrix
    plt.figure(figsize=(14, 8))

    image = plt.imshow(
        normalised_matrix.values,
        interpolation="nearest",
        cmap="Blues",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(image, fraction=0.046, pad=0.04)

    plt.xticks(
        range(len(column_labels)),
        column_labels,
        rotation=90,
    )

    plt.yticks(
        range(len(row_labels)),
        row_labels,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.title(
        f"{model_name.capitalize()}: Normalised Confusion Matrix\n"
        f"Top {NUM_CONFUSION_CLASSES} True Classes with Most Errors"
    )

    plt.tight_layout()

    plt.savefig(
        model_figure_dir / "selected_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 6. 置信度分布
# ============================================================

def plot_confidence_distribution(model_name, predictions, model_figure_dir):
    """
    比较正确预测和错误预测的置信度分布。
    """

    correct_confidence = predictions.loc[
        predictions["top1_correct"],
        "top1_confidence",
    ]

    wrong_confidence = predictions.loc[
        ~predictions["top1_correct"],
        "top1_confidence",
    ]

    plt.figure(figsize=(9, 6))
    plt.hist(
        correct_confidence,
        bins=20,
        alpha=0.6,
        label="Correct predictions",
    )
    plt.hist(
        wrong_confidence,
        bins=20,
        alpha=0.6,
        label="Wrong predictions",
    )

    plt.xlabel("Top-1 Confidence")
    plt.ylabel("Number of Images")
    plt.title(f"{model_name.capitalize()}: Confidence Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        model_figure_dir / "confidence_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 7. 单个模型的总体指标
# ============================================================

def calculate_model_summary(model_name, predictions):
    """
    计算一个模型在 clean test set 上的主要 Error Analysis 指标。
    """

    total_samples = len(predictions)

    top1_correct_count = int(predictions["top1_correct"].sum())
    top5_correct_count = int(predictions["top5_correct"].sum())

    top1_wrong_top5_correct_count = int(
        ((~predictions["top1_correct"]) & predictions["top5_correct"]).sum()
    )

    high_confidence_wrong_count = int(
        (
            (~predictions["top1_correct"])
            & (predictions["top1_confidence"] >= HIGH_CONFIDENCE_THRESHOLD)
        ).sum()
    )

    difficult_correct_count = int(
        (
            predictions["top1_correct"]
            & (predictions["top1_confidence"] < DIFFICULT_CORRECT_THRESHOLD)
        ).sum()
    )

    return {
        "model": model_name,
        "total_samples": total_samples,
        "top1_correct": top1_correct_count,
        "top1_accuracy": top1_correct_count / total_samples,
        "top5_correct": top5_correct_count,
        "top5_accuracy": top5_correct_count / total_samples,
        "top1_wrong_top5_correct": top1_wrong_top5_correct_count,
        "mean_confidence": predictions["top1_confidence"].mean(),
        "mean_correct_confidence": predictions.loc[
            predictions["top1_correct"],
            "top1_confidence",
        ].mean(),
        "mean_wrong_confidence": predictions.loc[
            ~predictions["top1_correct"],
            "top1_confidence",
        ].mean(),
        "difficult_correct": difficult_correct_count,
        "high_confidence_wrong": high_confidence_wrong_count,
    }


# ============================================================
# 8. 三个模型对比图
# ============================================================

def plot_model_comparison(summary_df):
    """
    绘制三个模型的 Top-1 和 Top-5 accuracy 对比图。
    """

    x_positions = np.arange(len(summary_df))
    bar_width = 0.35

    plt.figure(figsize=(9, 6))

    top1_bars = plt.bar(
        x_positions - bar_width / 2,
        summary_df["top1_accuracy"],
        width=bar_width,
        label="Top-1 Accuracy",
    )

    top5_bars = plt.bar(
        x_positions + bar_width / 2,
        summary_df["top5_accuracy"],
        width=bar_width,
        label="Top-5 Accuracy",
    )

    plt.xticks(x_positions, summary_df["model"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Clean Test Performance Comparison")

    # 在每个柱子上方显示具体准确率
    for bar in top1_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in top5_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_OUTPUT_ROOT / "model_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 9. 主程序
# ============================================================

def main():
    """
    依次分析 traditional、scratch 和 pretrained 三个模型。
    """

    ANALYSIS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model_summaries = []

    for model_name, prediction_path in PREDICTION_FILES.items():
        print("\n" + "=" * 70)
        print(f"Analysing model: {model_name}")
        print(f"Prediction file: {prediction_path}")
        print("=" * 70)

        # 读取该模型的完整预测结果
        predictions = pd.read_csv(prediction_path)

        # 生成 Error Analysis 所需的辅助列
        predictions = prepare_predictions(predictions)

        # CSV 和分析结果目录
        model_output_dir = ANALYSIS_OUTPUT_ROOT / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # PNG 图片目录
        model_figure_dir = FIGURE_OUTPUT_ROOT / model_name
        model_figure_dir.mkdir(parents=True, exist_ok=True)

        # 保存加入辅助分析列后的完整预测文件
        predictions.to_csv(
            model_output_dir / "predictions_with_analysis.csv",
            index=False,
        )

        # 分析不同类型的预测案例
        analyse_prediction_cases(
            model_name,
            predictions,
            model_output_dir,
            model_figure_dir,
        )

        # 分析模型最容易混淆的类别组合
        analyse_confused_pairs(
            model_name,
            predictions,
            model_output_dir,
            model_figure_dir,
        )

        # 绘制 selected confusion matrix
        plot_selected_confusion_matrix(
            model_name,
            predictions,
            model_output_dir,
            model_figure_dir,
        )

        # 绘制正确和错误预测的置信度分布
        plot_confidence_distribution(
            model_name,
            predictions,
            model_figure_dir,
        )

        # 记录该模型的总体统计数据
        model_summaries.append(
            calculate_model_summary(model_name, predictions)
        )

    # 汇总三个模型的统计结果
    summary_df = pd.DataFrame(model_summaries)

    summary_df.to_csv(
        ANALYSIS_OUTPUT_ROOT / "model_comparison_summary.csv",
        index=False,
    )

    # 绘制三个模型的准确率对比图
    plot_model_comparison(summary_df)

    print("\n" + "=" * 70)
    print("Overall model comparison")
    print("=" * 70)

    print(
        summary_df[
            [
                "model",
                "total_samples",
                "top1_accuracy",
                "top5_accuracy",
                "mean_confidence",
                "mean_correct_confidence",
                "mean_wrong_confidence",
                "high_confidence_wrong",
            ]
        ].to_string(index=False)
    )

    print("\nError analysis completed.")

    print(f"\nCSV results were saved to:\n{ANALYSIS_OUTPUT_ROOT}")
    print(f"\nFigures were saved to:\n{FIGURE_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()