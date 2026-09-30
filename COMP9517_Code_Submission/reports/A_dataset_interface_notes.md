# Dataset, Dataloader, Evaluation, and Robustness Interface Notes

This document explains the shared dataset split, dataloader interface, evaluation format, and robustness testing workflow prepared by Member A. All group members should follow this document so that the experiments are reproducible and comparable.

---

## 1. Current Dataset Split

We use a subset of the iNaturalist-2021 Mini dataset.

Current setting:

- Random seed: `9517`
- Number of selected species/classes: `500`
- Training images: `40` images per class, `20,000` images in total
- Validation images: `10` images per class, `5,000` images in total
- Test images: `10` images per class, `5,000` images in total
- Training and validation images are sampled from `train_mini`
- Test images are sampled from the official iNaturalist validation split

The training, validation, and test sets are kept strictly separate. The test set should only be used for final evaluation and should not be used for model selection or hyperparameter tuning.

---

## 2. Generated Split Files

All split files are stored in:

```text
data/splits/
```

Files included:

```text
selected_classes.txt
class_to_idx.json
idx_to_class.json
split_summary.json
train_paths.csv
val_paths.csv
test_paths.csv
```

Each CSV contains:

```text
image_path,image_id,category_id,label,class_idx,split
```

Column explanation:

- `image_path`: path to the image file
- `image_id`: original iNaturalist image ID
- `category_id`: original iNaturalist category ID
- `label`: species name
- `class_idx`: remapped class index used in this project, ranging from 0 to 499
- `split`: train / val / test

Important: all models should use `class_idx` as the training and evaluation label. Do not use the original `category_id`.

---

## 3. For Member B: Traditional Model

Member B is responsible for the SIFT + Bag-of-Visual-Words + SVM pipeline.

The traditional method can directly read the CSV files:

```python
import pandas as pd
import cv2

train_df = pd.read_csv("data/splits/train_paths.csv")

for _, row in train_df.iterrows():
    img_path = row["image_path"]
    label = row["class_idx"]

    img = cv2.imread(img_path)
    # Extract SIFT / BoVW features here
```

Expected clean output from Member B:

```text
results/clean/traditional_predictions.csv
```

Expected robustness prediction outputs from Member B:

```text
results/robustness/predictions/traditional_{degradation}_{severity}_predictions.csv
```

Examples:

```text
results/robustness/predictions/traditional_noise_0.05_predictions.csv
results/robustness/predictions/traditional_blur_5_predictions.csv
results/robustness/predictions/traditional_jpeg_40_predictions.csv
```

---

## 4. For Member C: Deep Learning from Scratch

Member C is responsible for the from-scratch CNN / ResNet18 model.

Member C can use the PyTorch dataloader directly:

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

val_loader = make_loader(
    csv_path="data/splits/val_paths.csv",
    split="val",
    batch_size=32,
    image_size=224,
    num_workers=0,
    augment=False,
)

test_loader = make_loader(
    csv_path="data/splits/test_paths.csv",
    split="test",
    batch_size=32,
    image_size=224,
    num_workers=0,
    augment=False,
)
```

Expected clean output from Member C:

```text
results/clean/scratch_predictions.csv
```

Expected robustness prediction outputs from Member C:

```text
results/robustness/predictions/scratch_{degradation}_{severity}_predictions.csv
```

Member C should also provide training logs, training curves, and the best checkpoint if available.

---

## 5. For Member D: Pretrained / Transfer Learning Model

Member D is responsible for the ImageNet-pretrained ResNet18 / EfficientNet model.

Member D should use the same dataloader interface as Member C.

Expected clean output from Member D:

```text
results/clean/pretrained_predictions.csv
```

Expected robustness prediction outputs from Member D:

```text
results/robustness/predictions/pretrained_{degradation}_{severity}_predictions.csv
```

Member D should also provide:

- Best pretrained checkpoint
- Model architecture information
- Class mapping
- Correctly classified examples
- Incorrectly classified examples

These files are needed by Member E for Grad-CAM visualisation.

---

## 6. For Member E: Error Analysis, Grad-CAM, and Robustness Discussion

Member E will use prediction files and metrics generated from B/C/D.

Clean files to use:

```text
results/clean/traditional_predictions.csv
results/clean/traditional_metrics.json
results/clean/scratch_predictions.csv
results/clean/scratch_metrics.json
results/clean/pretrained_predictions.csv
results/clean/pretrained_metrics.json
```

Robustness files to use:

```text
results/robustness/combined_robustness_results.csv
results/figures/robustness_{degradation}_top1_accuracy.png
results/figures/robustness_{degradation}_macro_f1.png
```

Member E will use these files for:

- Confusion matrix analysis
- Hardest confused species pairs
- Correct and incorrect examples
- Grad-CAM visualisation for the pretrained model
- Robustness curves and discussion

---

## 7. Required Clean Prediction CSV Format

All model owners must output a clean prediction CSV with the following format:

```text
image_path,true_idx,pred_idx,top5_idx
```

Example:

```text
data/raw/val/example.jpg,12,18,18 12 35 7 209
```

Important:

- `true_idx`, `pred_idx`, and `top5_idx` must use the remapped 0-499 `class_idx` labels.
- Do not use the original iNaturalist `category_id`.

---

## 8. Unified Clean Evaluation

If a model outputs a clean prediction file such as:

```text
results/clean/pretrained_predictions.csv
```

it can be evaluated using:

```bash
python src/evaluate.py --pred-csv results/clean/pretrained_predictions.csv --out-json results/clean/pretrained_metrics.json --num-classes 500
```

The evaluation script reports:

- Top-1 accuracy
- Top-5 accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix

---

## 9. Robustness Testing Template

A robustness template is provided at:

```text
results/robustness/robustness_result_template.csv
```

Members should use this file to check:

- The required degradation names
- The required severity levels
- The expected prediction CSV file names
- The expected metrics JSON file names

The final combined robustness table will be saved to:

```text
results/robustness/combined_robustness_results.csv
```

Its key columns are:

```text
model,degradation,severity,top1_accuracy,macro_f1
```

Example rows:

```text
traditional,clean,0,0.120,0.095
traditional,noise,0.05,0.098,0.081
scratch,clean,0,0.180,0.150
pretrained,clean,0,0.420,0.390
```

---

## 10. Robustness Prediction and Metrics Locations

For each degraded test condition, model owners should save prediction CSV files under:

```text
results/robustness/predictions/
```

Naming format:

```text
{model}_{degradation}_{severity}_predictions.csv
```

Examples:

```text
traditional_noise_0.05_predictions.csv
scratch_blur_5_predictions.csv
pretrained_jpeg_40_predictions.csv
```

Member A will run unified evaluation and save metrics JSON files under:

```text
results/robustness/metrics/
```

Naming format:

```text
{model}_{degradation}_{severity}_metrics.json
```

Examples:

```text
traditional_noise_0.05_metrics.json
scratch_blur_5_metrics.json
pretrained_jpeg_40_metrics.json
```

---

## 11. Robustness Summary and Plotting Scripts

Two helper scripts are provided:

```text
src/summarize_robustness.py
src/plot_robustness_curves.py
```

After metrics JSON files are available, summarize them using:

```bash
python src/summarize_robustness.py --clean-dir results/clean --metrics-dir results/robustness/metrics --out-csv results/robustness/combined_robustness_results.csv
```

Then generate robustness curves using:

```bash
python src/plot_robustness_curves.py --input-csv results/robustness/combined_robustness_results.csv --out-dir results/figures
```

---

## 12. Current Status of Member A's Part

Completed:

- 500-class dataset split
- Train / validation / test CSV generation
- Class mapping files
- Image path checking
- Image opening check
- Train / validation / test dataloader test
- Dummy metrics test
- Degradation function test
- Clean evaluation script
- Robustness summary template
- Robustness summary script
- Robustness plotting script

Remaining tasks:

- Support B/C/D in using the shared dataset interface
- Collect clean prediction CSV files from B/C/D
- Run unified clean evaluation
- Collect degraded prediction CSV files or metrics from B/C/D
- Generate the combined robustness table
- Generate robustness curves for the final report and presentation
