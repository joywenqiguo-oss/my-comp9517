# Member D Notes: ImageNet-Pretrained ResNet18 (Transfer Learning)

## Method

- Architecture: ResNet18
- Initialization: ImageNet-pretrained (`torchvision.models.ResNet18_Weights.IMAGENET1K_V1`)
- Classes: 500
- Training images: 20,000
- Validation images: 5,000
- Test images: 5,000
- Input size: 224 × 224
- Optimizer: AdamW
- Weight decay: 0.0001
- Batch size: 64
- Scheduler: CosineAnnealingLR
- Data augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter (same shared `src/dataset.py` transform used by C)

## Freeze vs. Fine-tune Ablation

Two transfer-learning strategies were compared, both trained for up to 20 epochs with early stopping (patience 6, selection on validation macro-F1):

| Setting | Trainable params | Initial LR | Val Top-1 | Val Top-5 | Val Macro F1 | Best epoch | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen backbone (linear probe) | 256,500 / 11.4M | 1e-3 | 56.72% | 78.18% | 56.09% | 20 | 2.1 h |
| Full fine-tune | 11.4M / 11.4M | 1e-4 | 68.62% | 87.44% | 68.32% | 20 | 7.8 h |

Full fine-tuning outperforms the frozen-backbone linear probe by **+11.9 points top-1** and **+12.2 points macro-F1** — a gap that holds from the very first epoch onward. The two runs converge differently: the frozen backbone was still creeping up slightly every epoch at the 20-epoch cap (56.06%→56.72% over the last 5 epochs) and had not fully plateaued, while fine-tuning's validation accuracy had largely plateaued/oscillated by epoch 20 (67.9%–68.6% over the last 5 epochs) even as training accuracy kept climbing (80.7%→82.6%) — i.e. fine-tuning was reaching a healthy stopping point with mild overfitting onset, not undertrained. Interestingly, at epoch 1 the frozen backbone was briefly *ahead* (31.5% vs. 28.4% val top-1), since it only has to learn the new linear head; fine-tuning overtakes it by epoch 4 and pulls steadily further ahead after that.

**Selected checkpoint: `results/models/pretrained_finetune_best.pth`** (epoch 20).

## Clean Test Results (held-out test set, `pretrained_finetune_best.pth`)

- Top-1 accuracy: 68.24%
- Top-5 accuracy: 86.98%
- Macro precision: 69.71%
- Macro recall: 68.24%
- Macro F1: 67.95%

This is a large improvement over Member C's from-scratch ResNet18 (32.10% top-1 / 29.73% macro-F1 on the same test set), demonstrating the value of ImageNet-pretrained features for this fine-grained, data-scarce (40 images/class) task.

## Robustness Summary

Full sweep in `results/robustness/combined_robustness_results.csv` and curves in `results/figures/robustness_*.png` (pretrained overlaid against scratch).

Strongest-severity results:

| Degradation | Severity | Top-1 | Macro F1 |
|---|---:|---:|---:|
| Gaussian noise | 0.20 | 15.48% | 15.66% |
| Gaussian blur | 7 | 32.98% | 33.12% |
| Brightness | 0.40 | 63.96% | 63.87% |
| Contrast | 0.40 | 60.52% | 61.06% |
| JPEG quality | 20 | 62.84% | 62.52% |

The pretrained model is far more robust than the scratch model in absolute terms at every severity (e.g. blur-7: 32.98% vs scratch's 15.46%), but it degrades sharply under strong Gaussian noise and blur — both drop below scratch's *clean* accuracy once noise reaches sigma=0.2. It is comparatively robust to brightness/contrast/JPEG changes, similar to the pattern C found for the scratch model. This is consistent with ImageNet-pretrained low-level filters being tuned to natural, low-noise photographs.

## Files for Member E (Grad-CAM / error analysis)

- Checkpoint: `results/models/pretrained_finetune_best.pth` (131 MB, not committed to git — excluded by `.gitignore` and over GitHub's 100 MB push limit). Download from the project's GitHub Release instead:
  `https://github.com/2393159898-afk/COMP9517-project/releases/download/d-pretrained-checkpoint/pretrained_finetune_best.pth`
  Save it to `results/models/pretrained_finetune_best.pth` in your local checkout before loading it.
- Architecture: `src/pretrained_model.py` (`build_pretrained_resnet18`)
- Class mapping: `data/splits/selected_classes.txt`, `data/splits/idx_to_class.json`, `data/splits/class_to_idx.json`
- Correct examples (30, random sample): `results/clean/pretrained_correct_examples.csv`
- Incorrect examples (30, random sample): `results/clean/pretrained_incorrect_examples.csv`
- Hardest confused species pairs (top 20 by count): `results/clean/pretrained_confused_pairs.csv` — mostly same-genus pairs (e.g. *Platalea leucorodia* vs *Platalea flavipes*, *Gopherus agassizii* vs *Testudo graeca*, *Quercus alba* vs *Quercus buckleyi*), good Grad-CAM candidates for the "confusable species pairs" analysis in the spec.
- Selection script (rerunnable): `src/select_gradcam_examples.py`

## Environment Note (MPS-specific, not relevant on Colab/CUDA)

Training was run locally on Apple Silicon (MPS backend). An environment-specific
bug was found and worked around: running the validation forward pass in-process,
immediately after a training epoch, silently produced near-random accuracy on
this machine's PyTorch/MPS build, even though the trained weights were correct
(confirmed by reloading the same checkpoint fresh, which always evaluated
correctly). `src/train_pretrained.py` now runs per-epoch validation in a fresh
subprocess (`src/_mps_eval_worker.py`) whenever `device.type == "mps"`; on
CUDA (e.g. Colab) it uses the normal in-process path and this workaround is a
no-op. If you rerun training on Colab, this won't trigger and isn't needed.

## Important Files

- `src/pretrained_model.py`: model definition (freeze/fine-tune)
- `src/train_pretrained.py`: training (freeze and fine-tune ablation)
- `src/predict_pretrained.py`: clean and robustness prediction
- `src/compare_pretrained_runs.py`: freeze vs. fine-tune summary table
- `src/select_gradcam_examples.py`: correct/incorrect/confused-pair example selection
- `results/models/pretrained_finetune_best.pth`, `pretrained_freeze_best.pth`: checkpoints
- `results/logs/pretrained_freeze_*`, `pretrained_finetune_*`: training logs, curves, summaries
- `results/clean/pretrained_predictions.csv`, `pretrained_metrics.json`: clean test results
- `results/robustness/predictions/pretrained_*`, `results/robustness/metrics/pretrained_*`: 15 degraded-test conditions
