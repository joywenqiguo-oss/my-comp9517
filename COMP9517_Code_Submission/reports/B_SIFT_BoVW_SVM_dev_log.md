# SIFT + BoVW + SVM: Development Log

*Traditional baseline classifier — engineering notes and experiment log*

## Purpose of this document

This log tracks the real engineering problems, technical decisions, and key experimental data encountered while building the traditional baseline classifier (SIFT + Bag-of-Visual-Words + SVM) for the 500-class species classification task. It serves two purposes:

1. Provides grounded material for the methodology / discussion sections of the final report — actual engineering reasoning rather than a bare statement of "we used SIFT and SVM."
2. Gives teammates (particularly whoever handles the cross-model analysis) a direct reference for why specific choices were made in this component.

**Maintenance note**: entries are appended to the relevant section as each stage is completed. This file is meant to stay a single, continuously updated log rather than being replaced or split into multiple files.

---

## Environment

- OpenCV 5.0.0 (`cv2.SIFT_create()` works directly, no opencv-contrib needed)
- System RAM: 16.0 GB (LPDDR4); available memory during normal use is often only 2-3 GB
- The laptop CPU throttles under sustained heavy load (observed drop from 3.41GHz to 2.25GHz), which introduces timing variance unrelated to code efficiency
- Dataset: 500 classes, 20,000 train images (40/class), 5,000 val images, 5,000 test images

---

## Issue 1: MemoryError from extracting all descriptors at once

### Symptom
After extracting SIFT descriptors from 2,000 training images, calling `np.vstack(all_descriptors)` to concatenate them into a single array raised:
```
MemoryError: Unable to allocate 1.38 GiB for an array with shape (2884747, 128) and data type float32
```
Available memory at the time was only 1.61 GB; `vstack` needs to hold both the original list and the new array simultaneously, and the peak exceeded what was available.

### Analysis
- Across the full 20,000 training images, the total descriptor count reached **37,376,263** (≈1,869 per image on average). An earlier extrapolation from a 50-image sample had suggested only ~927/image — a reminder that small-sample extrapolation can be significantly off and shouldn't be trusted for capacity planning.
- Holding all raw descriptors (128-dim float32) at once would require roughly 9.5–19 GB, far beyond available memory, and one-shot concatenation operations like `vstack` add an additional peak on top of that.

### Solution
Abandoned the "extract everything, then concatenate" approach in favor of a **two-pass pipeline**:
- **Pass 1**: stream through all images extracting SIFT descriptors, using **reservoir sampling** to maintain a fixed-size random sample (200,000 descriptors) for vocabulary learning. Descriptors are discarded per image; nothing is concatenated in bulk.
- **Pass 2**: once the vocabulary is trained, re-stream through each image, extract descriptors, and immediately encode into a fixed-length BoVW histogram (far smaller than the raw descriptors). Raw descriptors are discarded right after encoding.

### Pass 1 results (notebook validation)
- All 20,000 images processed, 0 failures
- Total time: 1,223s ≈ 20.4 min
- Available memory stayed stable in the 2.5–3.3 GB range throughout, no spikes
- Sample pool filled to the target 200,000 descriptors

### Pass 2 results (notebook validation, encoding for k=128/256/512 simultaneously)
- Train (20,000 images): 1,782.2s ≈ 29.7 min, 0 failures
- Val (5,000 images): 529.9s ≈ 8.8 min, 0 failures
- Per-image time: ~0.089s (train) vs ~0.106s (val) — the difference is within normal range (val was briefly mistaken for anomalously slow, later confirmed to be a misreading of cumulative elapsed time, not an actual performance issue)

---

## Issue 2: Choosing the vocabulary size (k)

### Decision: compare k = 128 / 256 / 512, using MiniBatchKMeans (much faster than standard KMeans at this scale)

### Clustering time vs. inertia (on the 200k sampled descriptors)
| k | Time | Inertia (within-cluster sum of squares) |
|---|---|---|
| 128 | 5.4s | 17,253,734,400 |
| 256 | 2.0s | 16,275,645,440 |
| 512 | 3.9s | 15,349,090,304 |

**Important caveat**: inertia decreases monotonically as k increases almost by definition (more clusters → tighter clusters), so it cannot be used to choose k. The real criterion is downstream SVM accuracy on the validation set.

### Val-set accuracy with default SVM (rbf, C=1.0) across k
| k | Val accuracy | SVM train time | SVM predict time |
|---|---|---|---|
| 128 | 7.14% | 221.8s | 72.6s |
| 256 | 8.26% | 304.5s | 90.4s |
| 512 | **8.90%** (best) | 432.2s | 104.6s |

Accuracy improves with k but with diminishing returns (128→256: +1.12pp; 256→512: +0.64pp), while compute cost keeps rising. **k=512 was selected**, with SVM hyperparameter tuning to follow on top of it.

---

## Issue 3: SVM hyperparameter tuning (kernel + C) at k=512

### Grid
kernel: `linear` / `rbf`; C: `0.1 / 1 / 10 / 100` — 8 combinations total

### Full results (sorted by val accuracy)
| kernel | C | Val accuracy | Train time | Predict time |
|---|---|---|---|---|
| rbf | 10 | **10.66%** (tied best) | 62.4s | 99.9s |
| rbf | 100 | **10.66%** (tied best) | 62.2s | 100.5s |
| linear | 10 | 9.38% | 54.6s | 97.3s |
| linear | 100 | 9.14% | 85.5s | 105.5s |
| rbf | 1 | 8.90% | 432.2s | 104.6s |
| linear | 1 | 7.56% | 41.2s | 88.3s |
| rbf | 0.1 | 5.42% | 55.7s | 96.9s |
| linear | 0.1 | 4.16% | 49.6s | 88.1s |

### Observations
1. **rbf with C=10 and C=100 tie exactly** at 10.66%, meaning further increasing C yields no additional benefit while raising overfitting risk — C=10 sits at the sweet spot.
2. **Notable anomaly**: rbf at C=1 took 432.2s to train, but C=10/100 took only ~62s — training time doesn't scale monotonically with C. This reflects the convergence behavior of the underlying SVM solver (libsvm) for different optimization problems, not a code efficiency issue.
3. Linear kernel underperforms rbf across the board, indicating the BoVW feature space has meaningful non-linear structure that rbf's kernel mapping captures better.

### Final configuration
```
Vocabulary size k = 512
SVM kernel = rbf
SVM C = 10
Val accuracy = 10.66%
```

---

## Stage C: Full run of `src/train_traditional.py`

### Script structure
The validated notebook logic was reorganized into a clean, reproducible script (no exploratory code, only the final chosen configuration):
- `build_vocabulary_sample()` — Pass 1 reservoir sampling
- `encode_bovw()` / `encode_dataset()` — Pass 2 BoVW encoding
- `main()` — orchestrates the full pipeline and saves models + timing info

### Runtime notes
- The script must be run from the **project root** (`COMP9517_Project_Team_Package`), not from within `src/` — relative paths like `data/splits/train_paths.csv` will otherwise raise `FileNotFoundError`.
- On Windows, running via an IDE agent terminal may show a `conda activate base` error due to PowerShell's script execution policy — this is harmless and can be ignored. If invoking `python` directly appears to hang or do nothing (conda environment not properly activated in that terminal), use the full interpreter path (e.g. `D:\anaconda3\python.exe`) instead.

### Full run timing (fresh process, not reusing notebook state)
| Stage | Time |
|---|---|
| Pass 1 (reservoir sampling, 20,000 images) | 1,055.1s ≈ 17.6 min |
| KMeans training (k=512) | 7.2s |
| Pass 2 (BoVW encoding, 20,000 images) | 1,061.5s ≈ 17.7 min |
| SVM training (rbf, C=10, probability=True) | 501.6s ≈ 8.4 min |
| **Total** | **2,625.4s ≈ 43.8 min** |

Failed images: 0 (all 20,000 processed successfully)

### Note on SVM training time discrepancy
The formal script's SVM training (501.6s) is notably slower than the notebook tuning run (62.4s) because `probability=True` was enabled, which triggers sklearn's internal Platt-scaling calibration. This was originally intended to support `predict_proba()` for top-5 ranking. (**Later correction**: the prediction script ended up using `decision_function()` instead, so `probability=True` turned out not to be strictly necessary — the model was already trained by that point and wasn't retrained. Recorded here as an honest account of how the decision evolved.)

### Artifacts produced
```
results/models/kmeans_sample_descriptors.npy
results/models/traditional_kmeans_k512.pkl
results/models/traditional_svm.pkl
results/models/traditional_train_timing.json
results/features/train_feat_k512.npy
results/features/train_labels.npy
```

---

## Stage D: Writing and debugging `src/predict_traditional.py`

### Script design
Loads the trained kmeans/svm models and predicts on 1 clean test set + 15 robustness variants (5 degradation types × 3 severities, read from `degradations.py`'s `DEGRADATION_LEVELS`). Ranking uses `decision_function()` rather than `predict_proba()` (faster, and equivalent for top-k purposes). Degradation is applied via `degradations.py`'s `apply_degradation()`; images are loaded with PIL throughout to match that function's interface.

### Issue 4: `decision_function()` on 5,000 images at once → ArrayMemoryError

#### Symptom
```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 4.65 GiB for an array
with shape (5000, 124750) and data type float64
```

#### Root cause
Regardless of the `decision_function_shape` setting, sklearn's `SVC` computes One-vs-One pairwise scores internally under the hood. With 500 classes, the number of pairs is:
$$\frac{500 \times 499}{2} = 124{,}750$$
Calling `decision_function` on all 5,000 images at once requires allocating `5000 × 124750 × 8 bytes ≈ 4.65 GB`, which exceeds available memory. This is the same class of problem as the earlier `vstack` MemoryError — a one-shot operation on the full dataset causing a memory spike — just surfacing at inference time rather than training time.

#### Fix
Batch the `decision_function` calls (batch_size=200), reducing peak memory to ~200MB per batch. Batching doesn't change total compute, so the extra overhead is negligible.

#### Takeaway (usable in the report's methodology section)
For multi-class SVMs with a large number of classes, `decision_function` / `predict_proba` memory cost grows **quadratically** with the number of classes (O(n²) OvO pairs) — an easy-to-miss hidden trap. Inference should not be assumed to be inherently more memory-friendly than training.

### Final run: all 16 variants completed successfully
- Total time: 4,096.6s ≈ 68.3 min
- All 16 variants (1 clean + 15 robustness), 0 failures

### Deliverables (used by teammates for integration / cross-model analysis)
```
results/clean/traditional_predictions.csv
results/robustness/predictions/traditional_{degradation}_{severity}_predictions.csv (×15)
results/models/traditional_predict_timing.json
```

---

## Stage E: Final evaluation (`evaluate.py` + `summarize_robustness.py`)

### Clean test set — final metrics
| Metric | Value |
|---|---|
| Top-1 accuracy | 10.74% |
| Top-5 accuracy | 23.04% |
| Macro precision | 10.47% |
| Macro recall | 10.74% |
| Macro F1 | 9.82% |

Test accuracy (10.74%) tracks val accuracy (10.66%) closely (within 0.08pp), indicating the hyperparameters chosen on the validation set did **not overfit** to it — performance transfers cleanly to unseen test data, which is a good sign that the whole pipeline is reproducible and reliable.

### Robustness results (`results/robustness/combined_robustness_results.csv`)
| Degradation | Severity | Top-1 accuracy | Macro F1 |
|---|---|---|---|
| clean | 0 | 0.1074 | 0.0982 |
| noise | 0.05 | 0.0946 | 0.0861 |
| noise | 0.1 | 0.0796 | 0.0688 |
| noise | 0.2 | 0.0392 | 0.0312 |
| blur | 3 | 0.0542 | 0.0491 |
| blur | 5 | 0.0212 | 0.0167 |
| blur | 7 | 0.0090 | 0.0043 |
| brightness | 0.8 | 0.1006 | 0.0920 |
| brightness | 0.6 | 0.0870 | 0.0822 |
| brightness | 0.4 | 0.0412 | 0.0405 |
| contrast | 0.8 | 0.0996 | 0.0916 |
| contrast | 0.6 | 0.0834 | 0.0783 |
| contrast | 0.4 | 0.0428 | 0.0421 |
| jpeg | 70 | 0.0990 | 0.0893 |
| jpeg | 40 | 0.1026 | 0.0929 |
| jpeg | 20 | 0.0914 | 0.0829 |

### Key findings (for the report's discussion section)
1. **Blur is by far the most damaging degradation.** Accuracy collapses from 10.74% (clean) to 0.90% at blur_7 — essentially random-guess territory. This has a clean algorithmic explanation: SIFT relies on local gradient/edge information to localize keypoints, and blurring destroys exactly that structural detail. This is a well-known, fundamental weakness of SIFT-based methods, not an implementation artifact.
2. **JPEG compression has almost no effect.** Top-1 accuracy stays in the 9.0–10.3% band across compression levels 70/40/20, without a clear monotonic decline (jpeg_40 is even slightly higher than jpeg_70, likely sampling noise). JPEG mainly discards high-frequency detail and color blocks, while SIFT keypoints tend to sit on strong structural edges that survive compression reasonably well.
3. **Noise, brightness, and contrast all show non-linear degradation** — accuracy drops faster as severity increases rather than declining linearly — with noise degrading slightly faster than brightness/contrast at comparable severity levels.

---

## Status / next steps

- [x] Pass 1: reservoir-sample 200k descriptors
- [x] Compare k=128/256/512
- [x] Pass 2: encode all train/val images into BoVW histograms
- [x] SVM hyperparameter grid search — final config: k=512, rbf, C=10
- [x] Finalize and run `src/train_traditional.py` end-to-end
- [x] Write and run `src/predict_traditional.py` (clean + 15 robustness variants, 0 failures)
- [x] Compute final clean test-set metrics via `evaluate.py` (top1=10.74%, top5=23.04%)
- [x] Complete all 15 robustness evaluations and summary via `evaluate.py` + `summarize_robustness.py`
- [ ] (Optional) Generate confusion matrix / robustness curve plots via `plot_confusion_matrix.py` / `plot_robustness_curves.py` for the report
- [ ] Consolidate final training/testing time figures into the report's methodology/discussion sections

---

*Last updated: the full pipeline for this component (train → predict → evaluate → robustness analysis) is complete. Remaining work is limited to optional visualizations and writing the corresponding report sections.*
