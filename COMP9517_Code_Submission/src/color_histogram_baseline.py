"""
color_histogram_baseline.py

A lightweight supplementary baseline for comparison against SIFT + BoVW + SVM:
extract a simple HSV color histogram per image (no keypoints, no vocabulary),
train an SVM on it, and evaluate with exactly the same metrics used for the
main traditional model (via src/metrics.py) so the two are directly comparable.

This is NOT part of the official B/C/D deliverable pipeline (train_traditional.py
/ predict_traditional.py already cover that) — it's an extra experiment to make
the report's discussion section stronger by showing an actual comparison rather
than only theoretical pros/cons of different hand-crafted features.

Note: this baseline reuses the SVM hyperparameters (rbf, C=10) chosen for the
BoVW model rather than running a separate grid search, since the goal here is a
quick, fair comparison, not a fully independent optimization.

Usage:
    python src/color_histogram_baseline.py
"""

import os
import time
import json

import numpy as np
import pandas as pd
import cv2
from sklearn.svm import SVC

from metrics import evaluate_classification, save_metrics

import joblib

TRAIN_CSV = "data/splits/train_paths.csv"
TEST_CSV = "data/splits/test_paths.csv"
OUT_DIR = "results/extra"

BINS_PER_CHANNEL = 8  # 8x8x8 = 512 dims, same size as the BoVW histogram
SVM_KERNEL = "rbf"
SVM_C = 10
TOP_K = 5
PROGRESS_EVERY = 2000
BATCH_SIZE = 200  # for decision_function, same reasoning as predict_traditional.py


def color_histogram(image_path: str, bins: int = BINS_PER_CHANNEL) -> np.ndarray:
    """Compute a normalized HSV color histogram for one image."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(bins ** 3, dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def extract_features(df: pd.DataFrame, split_name: str) -> np.ndarray:
    n = len(df)
    feats = np.zeros((n, BINS_PER_CHANNEL ** 3), dtype=np.float32)
    start = time.time()
    for i, row in enumerate(df.itertuples()):
        feats[i] = color_histogram(row.image_path)
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"  [{split_name}] {i + 1}/{n}, elapsed={time.time() - start:.1f}s")
    print(f"[{split_name}] feature extraction done in {time.time() - start:.1f}s")
    return feats


def batched_predict(svm: SVC, feats: np.ndarray, batch_size: int = BATCH_SIZE):
    classes = svm.classes_
    n = feats.shape[0]
    pred_idx = np.zeros(n, dtype=np.int64)
    top5_idx = np.zeros((n, TOP_K), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores = svm.decision_function(feats[start:end])
        pred_idx[start:end] = classes[np.argmax(scores, axis=1)]
        order = np.argsort(-scores, axis=1)[:, :TOP_K]
        top5_idx[start:end] = classes[order]
    return pred_idx, top5_idx


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"train: {len(train_df)}, test: {len(test_df)}")

    timing = {}

    t0 = time.time()
    X_train = extract_features(train_df, "train")
    y_train = train_df["class_idx"].values
    timing["train_feature_extraction_sec"] = time.time() - t0

    t0 = time.time()
    X_test = extract_features(test_df, "test")
    y_test = test_df["class_idx"].values
    timing["test_feature_extraction_sec"] = time.time() - t0

    print(f"\nTraining SVM (kernel={SVM_KERNEL}, C={SVM_C}) ...")
    t0 = time.time()
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=9517)
    svm.fit(X_train, y_train)
    timing["svm_train_sec"] = time.time() - t0
    print(f"SVM trained in {timing['svm_train_sec']:.1f}s")

    # save the trained model so it can be reused without retraining
    joblib.dump(svm, os.path.join(OUT_DIR, "color_histogram_svm.pkl"))

    t0 = time.time()
    pred_idx, top5_idx = batched_predict(svm, X_test)
    timing["predict_sec"] = time.time() - t0

    metrics = evaluate_classification(
        y_true=y_test.tolist(),
        y_pred=pred_idx.tolist(),
        y_top5=top5_idx.tolist(),
        num_classes=500,
    )
    save_metrics(metrics, os.path.join(OUT_DIR, "color_histogram_metrics.json"))

    pred_df = pd.DataFrame({
        "image_path": test_df["image_path"],
        "true_idx": y_test,
        "pred_idx": pred_idx,
        "top5_idx": [" ".join(str(c) for c in row) for row in top5_idx],
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "color_histogram_predictions.csv"), index=False)

    timing["total_sec"] = sum(timing.values())
    with open(os.path.join(OUT_DIR, "color_histogram_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    print("\n=== Color Histogram + SVM baseline results ===")
    print(f"top1_accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"top5_accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()