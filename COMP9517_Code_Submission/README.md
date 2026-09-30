# COMP9517 Group Project

This repository contains the shared implementation and experimental results for fine-grained species classification on a fixed 500-class subset of the iNaturalist-2021 Mini dataset.

Three complete methods are compared:

1. **Traditional:** SIFT + Bag-of-Visual-Words + SVM
2. **From scratch:** ResNet18 trained from random initialisation
3. **Transfer learning:** ImageNet-pretrained ResNet18 with full fine-tuning

The project also includes robustness analysis, error analysis and Grad-CAM. All methods use the same fixed train/validation/test split and the same remapped `class_idx` label space.

---

## 1. Project Structure

```text
COMP9517-project/
鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ raw/                          # Local dataset, not submitted
鈹?  鈹斺攢鈹€ splits/                       # Fixed 500-class split
鈹溾攢鈹€ notebooks/
鈹?  鈹溾攢鈹€ A1_dataset_split_and_interface_check.ipynb
鈹?  鈹溾攢鈹€ A2_evaluation_and_robustness_summary.ipynb
鈹?  鈹溾攢鈹€ B_traditional_model.ipynb
鈹?  鈹溾攢鈹€ C_From_Scratch_ResNet.ipynb
鈹?  鈹斺攢鈹€ D_pretrained_model.ipynb
鈹溾攢鈹€ reports/
鈹溾攢鈹€ results/
鈹?  鈹溾攢鈹€ clean/
鈹?  鈹溾攢鈹€ logs/
鈹?  鈹溾攢鈹€ robustness/
鈹?  鈹斺攢鈹€ figures/
鈹溾攢鈹€ src/
鈹溾攢鈹€ PROJECT_TEAM_GUIDE_CN.md
鈹溾攢鈹€ requirements.txt
鈹斺攢鈹€ README.md
```

---

## 2. Dataset Placement

The raw dataset is not included. Place it locally as:

```text
data/raw/
鈹溾攢鈹€ train_mini/
鈹溾攢鈹€ val/
鈹溾攢鈹€ train_mini.json
鈹斺攢鈹€ val.json
```

Do not create an extra `train_mini/train_mini` or `val/val` directory layer.

The fixed split files are already under `data/splits/`. The CSVs contain paths and labels, but the images must still exist under `data/raw/` when training or displaying examples.

---

## 3. Fixed Dataset Split

Random seed: `9517`

| Split | Images/class | Total | Source |
|---|---:|---:|---|
| Train | 40 | 20,000 | `train_mini` |
| Validation | 10 | 5,000 | `train_mini` |
| Test | 10 | 5,000 | official validation split |

Important files:

```text
data/splits/selected_classes.txt
data/splits/class_to_idx.json
data/splits/idx_to_class.json
data/splits/split_summary.json
data/splits/train_paths.csv
data/splits/val_paths.csv
data/splits/test_paths.csv
```

Each split CSV contains:

```text
image_path,image_id,category_id,label,class_idx,split
```

All models must use `class_idx` from `0` to `499`.

`selected_classes.txt` stores:

```text
class_idx    original_category_id    scientific_species_name
```

`idx_to_class.json` stores the original category ID as a string. Use `selected_classes.txt` or the split CSV `label` column when scientific names are required.

---

## 4. Environment

```bash
pip install -r requirements.txt
```

PyTorch users should install a build suitable for their CPU, CUDA or Apple Silicon environment. On Windows, begin with `num_workers=0`.

---

## 5. Shared Dataloader

```python
from src.dataset import make_loader

train_loader = make_loader(
    csv_path="data/splits/train_paths.csv",
    split="train",
    batch_size=32,
    image_size=224,
    num_workers=0,
    augment=True,
)
```

Expected batch shapes:

```text
images: [batch_size, 3, 224, 224]
labels: [batch_size]
```

---

## 6. Prediction Interface

Clean predictions:

```text
results/clean/traditional_predictions.csv
results/clean/scratch_predictions.csv
results/clean/pretrained_predictions.csv
```

Degraded predictions:

```text
results/robustness/predictions/{model}_{degradation}_{severity}_predictions.csv
```

Format:

```text
image_path,true_idx,pred_idx,top5_idx
```

All indices must use the shared `0鈥?99` `class_idx` mapping.

---

## 7. Clean Test Results

| Model | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| Traditional SIFT + BoVW + SVM | 10.74% | 23.04% | 9.82% |
| Scratch ResNet18 | 32.10% | 57.66% | 29.73% |
| Pretrained ResNet18 | 68.24% | 86.98% | 67.95% |

---

## 8. Robustness Evaluation

Robustness is evaluated only at test time. Models are not retrained on degraded test images.

```text
noise:      0.05, 0.10, 0.20
blur:       3, 5, 7
brightness: 0.8, 0.6, 0.4
contrast:   0.8, 0.6, 0.4
jpeg:       70, 40, 20
```

Each model has one clean baseline and 15 degraded conditions. Across three models this gives 48 conditions.

Rebuild the unified summary:

```bash
python src/summarize_robustness.py     --clean-dir results/clean     --metrics-dir results/robustness/metrics     --out-csv results/robustness/combined_robustness_results.csv
```

The main integration notebook is:

```text
notebooks/A2_evaluation_and_robustness_summary.ipynb
```

It performs input validation, shared-test verification, clean comparison, degradation visualisation, robustness curves, absolute drop, retention, heatmaps, ranking, prediction-transition analysis, per-class sensitivity and representative failure analysis.

---

## 9. Notebook Order

```text
A1_dataset_split_and_interface_check.ipynb
B_traditional_model.ipynb
C_From_Scratch_ResNet.ipynb
D_pretrained_model.ipynb
A2_evaluation_and_robustness_summary.ipynb
```

---

## 10. Member Responsibilities

### Member A
Dataset split, dataloader/evaluation interface, robustness protocol, unified metrics, 48-condition integration, visualisation and A1/A2 notebooks.

### Member B
SIFT, Bag-of-Visual-Words, SVM, and traditional clean/degraded predictions.

### Member C
From-scratch ResNet18, augmentation ablation, training analysis, and clean/degraded predictions.

### Member D
ImageNet-pretrained ResNet18, freeze/fine-tune comparison, and clean/degraded predictions.

### Member E
Confusion analysis, difficult species pairs, Grad-CAM and integrated failure discussion.

---

## 11. Final Submission Exclusions

Do not include the raw dataset, model weights or generated result images in the final code ZIP.

```text
data/raw/
*.pt
*.pth
*.ckpt
*.pkl
*.npy
*.npz
*.jpg
*.jpeg
*.png
*.webp
*.gif
*.zip
*.tar
*.gz
*.7z
results/models/
results/features/
results/figures/
results/robustness/predictions/
__pycache__/
.ipynb_checkpoints/
```

Keep code, notebooks, split files, small summary CSV/JSON files and documentation.

---

## Repository and Submission Snapshot

The complete project repository is available at:

https://github.com/2393159898-afk/COMP9517-project

This submission ZIP was prepared from:

- Branch: `main`
- Repository baseline commit: `abb7885`
- The submission ZIP additionally includes selected final result summaries generated locally after this commit.

The submitted ZIP contains the project source code, notebooks, fixed dataset
split metadata, selected experiment summaries, technical documentation and
reproducibility instructions.

Raw iNaturalist images, trained model weights, large prediction files, metric
files containing full confusion matrices, and generated result images are
intentionally excluded from the submission ZIP.

The submitted ZIP should be treated as the official code submission. The
GitHub repository is provided as an additional reference for the complete
development history.

