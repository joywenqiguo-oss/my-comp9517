from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import build_transform
from pretrained_model import build_pretrained_resnet18


# ============================================================
# 1. 路径和分析参数
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "pretrained_finetune_best.pth"
)

DEFAULT_CORRECT_CSV = (
    PROJECT_ROOT
    / "results"
    / "clean"
    / "pretrained_correct_examples.csv"
)

DEFAULT_INCORRECT_CSV = (
    PROJECT_ROOT
    / "results"
    / "clean"
    / "pretrained_incorrect_examples.csv"
)

DEFAULT_CONFUSED_PAIRS_CSV = (
    PROJECT_ROOT
    / "results"
    / "clean"
    / "pretrained_confused_pairs.csv"
)

PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "results"
    / "clean"
    / "pretrained_predictions.csv"
)

IDX_TO_CLASS_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "idx_to_class.json"
)

OUTPUT_FIGURE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "gradcam"
    / "pretrained"
)

OUTPUT_ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "gradcam_pretrained"
)

# 每类默认生成的 Grad-CAM 图片数量
DEFAULT_NUM_CORRECT = 5
DEFAULT_NUM_INCORRECT = 5
DEFAULT_NUM_CONFUSED = 3

# Grad-CAM 热力图覆盖到原图时的透明度
OVERLAY_ALPHA = 0.45


# ============================================================
# 2. 路径和数据辅助函数
# ============================================================

def resolve_image_path(raw_path: str) -> Path:
    """
    将 CSV 中的图片路径转换成当前系统能够使用的路径。

    数据文件可能保存 Windows 风格反斜杠路径，
    因此需要兼容 macOS、Linux 和 Windows。
    """
    original_path = Path(raw_path)

    if original_path.exists():
        return original_path

    converted_path = Path(
        raw_path.replace("\\", os.sep).replace("/", os.sep)
    )

    if converted_path.exists():
        return converted_path

    project_relative_path = PROJECT_ROOT / converted_path

    if project_relative_path.exists():
        return project_relative_path

    raise FileNotFoundError(
        "Cannot find image file.\n"
        f"CSV path: {raw_path}\n"
        f"Converted path: {converted_path}\n"
        f"Project-relative path: {project_relative_path}"
    )


def load_idx_to_class(mapping_path: Path) -> dict[int, str]:
    """
    读取类别编号到类别名称的映射。
    """
    with mapping_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    return {
        int(class_idx): str(class_name)
        for class_idx, class_name in raw_mapping.items()
    }


def extract_species_name(class_name: str) -> str:
    """
    从完整类别名称中提取较简洁的物种名称。

    如果映射文件中已经只包含物种名，则直接返回原名称。
    """
    parts = str(class_name).split("_")

    if len(parts) >= 2:
        return " ".join(parts[-2:])

    return str(class_name)

def extract_species_from_image_path(image_path: str) -> str:
    """
    从图片所在类别文件夹名称中提取物种名称。

    例如文件夹：
    00044_Animalia_Arthropoda_Arachnida_Araneae_Araneidae_Trichonephila_clavata

    最终返回：
    Trichonephila clavata
    """
    normalised_path = str(image_path).replace("\\", "/")
    class_folder = Path(normalised_path).parent.name
    parts = class_folder.split("_")

    # 文件夹末尾两个字段通常是属名和种名
    if len(parts) >= 2:
        return f"{parts[-2]} {parts[-1]}"

    return class_folder


def build_class_idx_to_species(predictions_path: Path) -> dict[int, str]:
    """
    根据完整预测 CSV 中的图片路径，建立类别编号到物种名称的映射。

    图片所在文件夹包含完整分类信息，文件夹名称最后两个字段
    通常分别是属名和种名，因此可以得到真正的物种名称。
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Prediction CSV was not found:\n"
            f"{predictions_path}"
        )

    predictions = pd.read_csv(predictions_path)

    required_columns = {"image_path", "true_idx"}
    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "Prediction CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    class_idx_to_species = {}

    for _, row in predictions.iterrows():
        class_idx = int(row["true_idx"])

        if class_idx not in class_idx_to_species:
            class_idx_to_species[class_idx] = (
                extract_species_from_image_path(
                    str(row["image_path"])
                )
            )

    return class_idx_to_species


def get_device() -> torch.device:
    """
    自动选择 CUDA、Apple MPS 或 CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def safe_torch_load(
    checkpoint_path: Path,
    device: torch.device,
):
    """
    兼容不同 PyTorch 版本地加载 checkpoint。
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


# ============================================================
# 3. Grad-CAM 实现
# ============================================================

class GradCAM:
    """
    对指定卷积层计算 Grad-CAM。

    Grad-CAM 使用：
    1. 指定卷积层的 feature maps
    2. 目标类别相对于 feature maps 的 gradients
    3. 对 gradients 做全局平均，得到每个通道的重要程度
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    def _save_activations(
        self,
        module,
        inputs,
        output,
    ) -> None:
        """
        保存目标卷积层前向传播输出的 feature maps。
        """
        self.activations = output.detach()

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output,
    ) -> None:
        """
        保存目标类别反向传播到目标卷积层的 gradients。
        """
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        为单张图片生成 Grad-CAM。

        返回：
            heatmap
            predicted class index
            predicted confidence
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        predicted_class = int(logits.argmax(dim=1).item())
        predicted_confidence = float(
            probabilities[0, predicted_class].item()
        )

        if target_class is None:
            target_class = predicted_class

        target_score = logits[0, target_class]
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations or gradients."
            )

        # 对空间维度求平均，得到每个 feature-map 通道的权重
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # 按权重组合所有 feature maps
        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # 只保留对目标类别有正向贡献的区域
        cam = F.relu(cam)

        # 将 Grad-CAM 放大到输入图片大小
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # 将数值归一化到 0–1
        cam_min = cam.min()
        cam_max = cam.max()

        if float(cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return (
            cam.detach().cpu().numpy(),
            predicted_class,
            predicted_confidence,
        )

    def close(self) -> None:
        """
        移除 PyTorch hooks，避免重复注册。
        """
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# 4. 模型和图片加载
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    """
    创建 D 的 ResNet18 结构，并加载训练完成的参数。
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Pretrained checkpoint was not found:\n"
            f"{checkpoint_path}\n\n"
            "Ask Member D to upload "
            "results/models/pretrained_finetune_best.pth."
        )

    checkpoint = safe_torch_load(
        checkpoint_path,
        device,
    )

    num_classes = int(
        checkpoint.get("num_classes", 500)
    )

    image_size = int(
        checkpoint.get("image_size", 224)
    )

    model = build_pretrained_resnet18(
        num_classes=num_classes,
        freeze_backbone=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")

    return model, image_size


def load_image_for_gradcam(
    image_path: Path,
    image_size: int,
) -> tuple[Image.Image, torch.Tensor]:
    """
    同时读取：
    1. 用于展示的 RGB 原图
    2. 使用 D 的测试预处理转换后的模型输入
    """
    with Image.open(image_path) as image_file:
        original_image = image_file.convert("RGB")

    transform = build_transform(
        "test",
        image_size=image_size,
        augment=False,
    )

    input_tensor = transform(original_image).unsqueeze(0)

    return original_image, input_tensor


# ============================================================
# 5. 案例数据选择
# ============================================================

def standardise_example_dataframe(
    dataframe: pd.DataFrame,
    case_type: str,
) -> pd.DataFrame:
    """
    将 D 提供的不同案例 CSV 转换成统一格式。
    """
    dataframe = dataframe.copy()

    rename_candidates = {
        "class_idx": "true_idx",
        "label": "true_idx",
        "prediction": "pred_idx",
        "predicted_idx": "pred_idx",
    }

    dataframe = dataframe.rename(
        columns={
            old_name: new_name
            for old_name, new_name in rename_candidates.items()
            if old_name in dataframe.columns
        }
    )

    if "image_path" not in dataframe.columns:
        raise ValueError(
            f"{case_type} CSV does not contain image_path."
        )

    dataframe["case_type"] = case_type

    return dataframe


def load_example_csv(
    csv_path: Path,
    case_type: str,
    num_examples: int,
) -> pd.DataFrame:
    """
    读取指定案例文件，并保留需要分析的样本数量。
    """
    if not csv_path.exists():
        print(
            f"[Warning] Example file not found and will be skipped:\n"
            f"{csv_path}"
        )

        return pd.DataFrame()

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        print(
            f"[Warning] Example file is empty and will be skipped:\n"
            f"{csv_path}"
        )

        return pd.DataFrame()

    dataframe = standardise_example_dataframe(
        dataframe,
        case_type,
    )

    return dataframe.head(num_examples)


# def build_confused_pair_examples(
#     confused_pairs_path: Path,
#     predictions_path: Path,
#     num_examples: int,
# ) -> pd.DataFrame:
#     """
#     根据最常混淆类别组合，从完整预测结果中挑选实际图片。

#     confused-pair CSV 通常只包含 true_idx、pred_idx 和 count，
#     因此还要回到 pretrained_predictions.csv 中找到对应图片。
#     """
#     if (
#         not confused_pairs_path.exists()
#         or not predictions_path.exists()
#     ):
#         print(
#             "[Warning] Confused-pair inputs are missing. "
#             "Confused-pair Grad-CAM will be skipped."
#         )

#         return pd.DataFrame()

#     confused_pairs = pd.read_csv(
#         confused_pairs_path
#     )

#     predictions = pd.read_csv(
#         predictions_path
#     )

#     required_pair_columns = {
#         "true_idx",
#         "pred_idx",
#     }

#     if not required_pair_columns.issubset(
#         confused_pairs.columns
#     ):
#         raise ValueError(
#             "Confused-pairs CSV must contain "
#             "true_idx and pred_idx."
#         )

#     selected_rows = []

#     for _, pair in confused_pairs.iterrows():
#         true_idx = int(pair["true_idx"])
#         pred_idx = int(pair["pred_idx"])

#         matching_predictions = predictions[
#             (predictions["true_idx"] == true_idx)
#             & (predictions["pred_idx"] == pred_idx)
#         ]

#         if matching_predictions.empty:
#             continue

#         selected_row = matching_predictions.iloc[0].copy()
#         selected_row["case_type"] = "confused_pair"
#         selected_rows.append(selected_row)

#         if len(selected_rows) >= num_examples:
#             break

#     return pd.DataFrame(selected_rows)


def build_confused_pair_comparisons(
    confused_pairs_path: Path,
    predictions_path: Path,
    num_pairs: int,
) -> list[dict]:
    """
    为每个易混淆物种对选择两张图片：

    1. Species A 的错误样本：
       真实类别是 A，但模型预测成 B。

    2. Species B 的正确代表样本：
       真实类别和预测类别都是 B。

    最终每个物种对会用于生成一张两行三列对比图。
    """
    if not confused_pairs_path.exists():
        print(
            "[Warning] Confused-pairs CSV was not found:\n"
            f"{confused_pairs_path}"
        )
        return []

    if not predictions_path.exists():
        print(
            "[Warning] Prediction CSV was not found:\n"
            f"{predictions_path}"
        )
        return []

    confused_pairs = pd.read_csv(confused_pairs_path)
    predictions = pd.read_csv(predictions_path)

    required_pair_columns = {"true_idx", "pred_idx"}
    missing_pair_columns = required_pair_columns - set(confused_pairs.columns)

    if missing_pair_columns:
        raise ValueError(
            "Confused-pairs CSV is missing columns: "
            f"{sorted(missing_pair_columns)}"
        )

    required_prediction_columns = {
        "image_path",
        "true_idx",
        "pred_idx",
    }
    missing_prediction_columns = (
        required_prediction_columns - set(predictions.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction CSV is missing columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    selected_pairs = []

    for _, pair in confused_pairs.iterrows():
        species_a_idx = int(pair["true_idx"])
        species_b_idx = int(pair["pred_idx"])

        # 找到 Species A 被错分成 Species B 的样本
        species_a_errors = predictions[
            (predictions["true_idx"] == species_a_idx)
            & (predictions["pred_idx"] == species_b_idx)
        ]

        # 找到 Species B 被正确分类的代表样本
        species_b_correct = predictions[
            (predictions["true_idx"] == species_b_idx)
            & (predictions["pred_idx"] == species_b_idx)
        ]

        if species_a_errors.empty or species_b_correct.empty:
            continue

        # 如果有 confidence，优先选择更有代表性的样本
        if "top1_confidence" in predictions.columns:
            species_a_row = (
                species_a_errors
                .sort_values("top1_confidence", ascending=False)
                .iloc[0]
                .copy()
            )

            species_b_row = (
                species_b_correct
                .sort_values("top1_confidence", ascending=False)
                .iloc[0]
                .copy()
            )
        else:
            species_a_row = species_a_errors.iloc[0].copy()
            species_b_row = species_b_correct.iloc[0].copy()

        selected_pairs.append(
            {
                "species_a_idx": species_a_idx,
                "species_b_idx": species_b_idx,
                "species_a_name": extract_species_from_image_path(
                    str(species_a_row["image_path"])
                ),
                "species_b_name": extract_species_from_image_path(
                    str(species_b_row["image_path"])
                ),
                "species_a_row": species_a_row,
                "species_b_row": species_b_row,
            }
        )

        if len(selected_pairs) >= num_pairs:
            break

    return selected_pairs

def generate_gradcam_result(
    row: pd.Series,
    gradcam: GradCAM,
    device: torch.device,
    image_size: int,
    idx_to_class: dict[int, str],
) -> dict:
    """
    对一张图片生成 Grad-CAM 所需的全部结果，
    但暂时不单独保存图片。
    """
    raw_image_path = str(row["image_path"])
    actual_image_path = resolve_image_path(raw_image_path)

    original_image, input_tensor = load_image_for_gradcam(
        actual_image_path,
        image_size,
    )

    input_tensor = input_tensor.to(device)

    heatmap, predicted_class, predicted_confidence = gradcam.generate(
        input_tensor=input_tensor,
        target_class=None,
    )

    true_idx = int(row["true_idx"])

    # 真实物种名称可以直接从当前图片路径中准确提取
    true_name = extract_species_from_image_path(raw_image_path)

    # 预测物种没有对应图片路径，因此仍然先使用类别映射
    predicted_name = extract_species_name(
        idx_to_class.get(
            predicted_class,
            f"class_{predicted_class}",
        )
    )

    overlay = create_overlay(
        original_image,
        heatmap,
    )

    return {
        "raw_image_path": raw_image_path,
        "resolved_image_path": str(actual_image_path),
        "original_image": original_image,
        "heatmap": heatmap,
        "overlay": overlay,
        "true_idx": true_idx,
        "true_species": true_name,
        "pred_idx": predicted_class,
        "pred_species": predicted_name,
        "confidence": predicted_confidence,
    }

# ============================================================
# 6. Grad-CAM 图片绘制
# ============================================================

def create_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """
    将 Grad-CAM 热力图覆盖到原图上。
    """
    resized_image = original_image.resize(
        (heatmap.shape[1], heatmap.shape[0])
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    ) / 255.0

    colormap = plt.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[..., :3]

    overlay = (
        (1 - alpha) * image_array
        + alpha * heatmap_rgb
    )

    return np.clip(overlay, 0, 1)


def save_gradcam_figure(
    original_image: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
    case_type: str,
    true_idx: int,
    pred_idx: int,
    confidence: float,
    true_name: str,
    pred_name: str,
) -> None:
    """
    保存原图、热力图和叠加图的三栏展示。
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    figure.suptitle(
        f"{case_type.replace('_', ' ').title()}\n"
        f"True: {true_idx} — {true_name} | "
        f"Predicted: {pred_idx} — {pred_name} | "
        f"Confidence: {confidence:.4f}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confused_pair_comparison(
    species_a_result: dict,
    species_b_result: dict,
    output_path: Path,
) -> None:
    """
    保存易混淆物种对的两行三列对比图。

    第一行：
        Species A 被错误预测为 Species B。

    第二行：
        Species B 的正确分类参考样本。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    # ========================================================
    # 第一行：Species A 被错分成 Species B
    # ========================================================

    axes[0, 0].imshow(species_a_result["original_image"])
    axes[0, 0].set_title(
        "Misclassified Species A\n"
        f"True: {species_a_result['true_species']}\n"
        f"Predicted: {species_a_result['pred_species']} | "
        f"Confidence: {species_a_result['confidence']:.4f}",
        fontsize=10,
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        species_a_result["heatmap"],
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Grad-CAM Heatmap")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(species_a_result["overlay"])
    axes[0, 2].set_title("Grad-CAM Overlay")
    axes[0, 2].axis("off")

    # ========================================================
    # 第二行：Species B 的正确参考样本
    # ========================================================

    axes[1, 0].imshow(species_b_result["original_image"])
    axes[1, 0].set_title(
        "Correct Species B Reference\n"
        f"True: {species_b_result['true_species']}\n"
        f"Predicted: {species_b_result['pred_species']} | "
        f"Confidence: {species_b_result['confidence']:.4f}",
        fontsize=10,
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        species_b_result["heatmap"],
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Grad-CAM Heatmap")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(species_b_result["overlay"])
    axes[1, 2].set_title("Grad-CAM Overlay")
    axes[1, 2].axis("off")

    # 整张图的标题直接展示两个物种名称
    figure.suptitle(
        "Frequently Confused Species Pair\n"
        f"{species_a_result['true_species']} → "
        f"{species_b_result['true_species']}",
        fontsize=14,
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 7. 单个案例分析
# ============================================================

def analyse_single_example(
    row: pd.Series,
    gradcam: GradCAM,
    device: torch.device,
    image_size: int,
    class_idx_to_species: dict[int, str],
    example_index: int,
) -> dict:
    """
    对一条案例生成 Grad-CAM，并返回结果记录。
    """
    raw_image_path = str(row["image_path"])
    actual_image_path = resolve_image_path(
        raw_image_path
    )

    original_image, input_tensor = (
        load_image_for_gradcam(
            actual_image_path,
            image_size,
        )
    )

    input_tensor = input_tensor.to(device)

    heatmap, predicted_class, predicted_confidence = (
        gradcam.generate(
            input_tensor=input_tensor,
            target_class=None,
        )
    )

    true_idx = int(
        row.get("true_idx", -1)
    )

    csv_pred_idx = int(
        row.get("pred_idx", predicted_class)
    )

    # 真实类别直接从当前图片路径中提取准确的物种名称
    true_name = extract_species_from_image_path(
        raw_image_path
    )

    # 预测类别通过完整预测结果建立的映射转换成物种名称
    pred_name = class_idx_to_species.get(
        predicted_class,
        f"class_{predicted_class}",
    )

    overlay = create_overlay(
        original_image,
        heatmap,
    )

    case_type = str(
        row.get("case_type", "unknown")
    )

    output_dir = (
        OUTPUT_FIGURE_ROOT
        / case_type
    )

    output_filename = (
        f"{example_index:02d}_"
        f"true_{true_idx}_"
        f"pred_{predicted_class}.png"
    )

    output_path = (
        output_dir
        / output_filename
    )

    save_gradcam_figure(
        original_image=original_image,
        heatmap=heatmap,
        overlay=overlay,
        output_path=output_path,
        case_type=case_type,
        true_idx=true_idx,
        pred_idx=predicted_class,
        confidence=predicted_confidence,
        true_name=true_name,
        pred_name=pred_name,
    )

    prediction_matches_csv = (
        predicted_class == csv_pred_idx
    )

    print(
        f"[{case_type}] "
        f"{actual_image_path.name} | "
        f"true={true_idx} | "
        f"pred={predicted_class} | "
        f"confidence={predicted_confidence:.4f} | "
        f"CSV match={prediction_matches_csv}"
    )

    return {
        "case_type": case_type,
        "image_path": raw_image_path,
        "resolved_image_path": str(actual_image_path),
        "true_idx": true_idx,
        "true_species": true_name,
        "csv_pred_idx": csv_pred_idx,
        "gradcam_pred_idx": predicted_class,
        "pred_species": pred_name,
        "top1_confidence": predicted_confidence,
        "prediction_matches_csv": prediction_matches_csv,
        "gradcam_figure": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
    }


# ============================================================
# 8. 主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--correct-csv",
        default=str(DEFAULT_CORRECT_CSV),
    )

    parser.add_argument(
        "--incorrect-csv",
        default=str(DEFAULT_INCORRECT_CSV),
    )

    parser.add_argument(
        "--confused-pairs-csv",
        default=str(DEFAULT_CONFUSED_PAIRS_CSV),
    )

    parser.add_argument(
        "--num-correct",
        type=int,
        default=DEFAULT_NUM_CORRECT,
    )

    parser.add_argument(
        "--num-incorrect",
        type=int,
        default=DEFAULT_NUM_INCORRECT,
    )

    parser.add_argument(
        "--num-confused",
        type=int,
        default=DEFAULT_NUM_CONFUSED,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    correct_csv_path = Path(args.correct_csv)
    incorrect_csv_path = Path(args.incorrect_csv)
    confused_pairs_csv_path = Path(
        args.confused_pairs_csv
    )

    OUTPUT_FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ANALYSIS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    model, image_size = load_model(
        checkpoint_path,
        device,
    )

    idx_to_class = load_idx_to_class(
        IDX_TO_CLASS_PATH
    )

    # 根据预测 CSV 的图片路径建立类别编号到真实物种名称的映射
    class_idx_to_species = build_class_idx_to_species(
        PREDICTIONS_CSV
    )

    # ResNet18 最后一个卷积阶段的最后一个残差块
    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    correct_examples = load_example_csv(
        csv_path=correct_csv_path,
        case_type="correct",
        num_examples=args.num_correct,
    )

    incorrect_examples = load_example_csv(
        csv_path=incorrect_csv_path,
        case_type="incorrect",
        num_examples=args.num_incorrect,
    )

    confused_pair_comparisons = build_confused_pair_comparisons(
        confused_pairs_path=confused_pairs_csv_path,
        predictions_path=PREDICTIONS_CSV,
        num_pairs=args.num_confused,
    )

    example_groups = [
        correct_examples,
        incorrect_examples,
    ]

    result_records = []

    try:
        for example_group in example_groups:
            if example_group.empty:
                continue

            for example_index, (_, row) in enumerate(
                example_group.iterrows(),
                start=1,
            ):
                record = analyse_single_example(
                    row=row,
                    gradcam=gradcam,
                    device=device,
                    image_size=image_size,
                    class_idx_to_species=class_idx_to_species,
                    example_index=example_index,
                )

                result_records.append(record)

        # 为易混淆物种对生成两行三列对比图
        for pair_index, pair_data in enumerate(
            confused_pair_comparisons,
            start=1,
        ):
            species_a_result = generate_gradcam_result(
                row=pair_data["species_a_row"],
                gradcam=gradcam,
                device=device,
                image_size=image_size,
                idx_to_class=idx_to_class,
            )

            species_b_result = generate_gradcam_result(
                row=pair_data["species_b_row"],
                gradcam=gradcam,
                device=device,
                image_size=image_size,
                idx_to_class=idx_to_class,
            )

            # 使用图片路径中提取出的真实物种名，避免类别映射只显示原始 ID
            species_a_result["true_species"] = pair_data["species_a_name"]
            species_b_result["true_species"] = pair_data["species_b_name"]

            # Species A 的预测目标就是 Species B
            species_a_result["pred_species"] = pair_data["species_b_name"]

            # Species B 参考样本应预测为 Species B
            species_b_result["pred_species"] = pair_data["species_b_name"]

            output_path = (
                OUTPUT_FIGURE_ROOT
                / "confused_pair"
                / (
                    f"{pair_index:02d}_"
                    f"true_{species_a_result['true_idx']}_"
                    f"confused_with_{species_b_result['true_idx']}.png"
                )
            )

            save_confused_pair_comparison(
                species_a_result=species_a_result,
                species_b_result=species_b_result,
                output_path=output_path,
            )

            # 将物种对信息保存到汇总 CSV
            result_records.append(
                {
                    "case_type": "confused_pair_misclassified",
                    "pair_index": pair_index,
                    "image_path": species_a_result["raw_image_path"],
                    "resolved_image_path": species_a_result["resolved_image_path"],
                    "true_idx": species_a_result["true_idx"],
                    "true_species": species_a_result["true_species"],
                    "csv_pred_idx": int(pair_data["species_a_row"]["pred_idx"]),
                    "gradcam_pred_idx": species_a_result["pred_idx"],
                    "pred_species": species_a_result["pred_species"],
                    "top1_confidence": species_a_result["confidence"],
                    "prediction_matches_csv": (
                        species_a_result["pred_idx"]
                        == int(pair_data["species_a_row"]["pred_idx"])
                    ),
                    "gradcam_figure": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

            result_records.append(
                {
                    "case_type": "confused_pair_reference",
                    "pair_index": pair_index,
                    "image_path": species_b_result["raw_image_path"],
                    "resolved_image_path": species_b_result["resolved_image_path"],
                    "true_idx": species_b_result["true_idx"],
                    "true_species": species_b_result["true_species"],
                    "csv_pred_idx": int(pair_data["species_b_row"]["pred_idx"]),
                    "gradcam_pred_idx": species_b_result["pred_idx"],
                    "pred_species": species_b_result["pred_species"],
                    "top1_confidence": species_b_result["confidence"],
                    "prediction_matches_csv": (
                        species_b_result["pred_idx"]
                        == int(pair_data["species_b_row"]["pred_idx"])
                    ),
                    "gradcam_figure": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

            print(
                f"[Confused pair {pair_index}] "
                f"{species_a_result['true_species']} → "
                f"{species_b_result['true_species']}"
            )

    finally:
        gradcam.close()

    results_dataframe = pd.DataFrame(
        result_records
    )

    summary_path = (
        OUTPUT_ANALYSIS_ROOT
        / "gradcam_summary.csv"
    )

    results_dataframe.to_csv(
        summary_path,
        index=False,
    )

    # Count unique Grad-CAM figure paths because each confused pair has two CSV records
    # but shares one combined comparison figure.
    num_figures = results_dataframe["gradcam_figure"].nunique()
    num_matches = results_dataframe["prediction_matches_csv"].sum()
    num_records = len(results_dataframe)

    print("\nGrad-CAM analysis completed.")
    print(f"Generated figures: {num_figures}")
    print(f"Summary records: {num_records}")
    print(f"Predictions matching CSV: {num_matches}/{num_records}")
    print(f"Summary CSV: {summary_path}")
    print(f"Figure directory: {OUTPUT_FIGURE_ROOT}")


if __name__ == "__main__":
    main()