"""
predict_traditional.py

Load the trained SIFT + BoVW + SVM model artifacts produced by
train_traditional.py, and run predictions on:
  1. The clean test set -> results/clean/traditional_predictions.csv
  2. 15 robustness-degraded versions of the test set (5 degradation types
     x 3 severities, using degradations.py) ->
     results/robustness/predictions/traditional_{deg}_{severity}_predictions.csv

Output CSV format (one row per test image):
    image_path,true_idx,pred_idx,top5_idx
where top5_idx is a space-separated list of the 5 most likely class
indices, ranked by SVM decision_function score (descending).

decision_function is used instead of predict_proba for ranking, since it
does not require the expensive Platt-scaling calibration that
probability=True triggers at train time, and gives an equivalent ranking
for top-k purposes.

Usage:
    python src/predict_traditional.py
"""

import os
import time
import json

import numpy as np
import pandas as pd
import cv2
import joblib
from PIL import Image

from degradations import apply_degradation, DEGRADATION_LEVELS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_CSV = "data/splits/test_paths.csv"
MODELS_DIR = "results/models"
CLEAN_OUT_DIR = "results/clean"
ROBUST_OUT_DIR = "results/robustness/predictions"

VOCAB_SIZE = 512
KMEANS_PATH = os.path.join(MODELS_DIR, f"traditional_kmeans_k{VOCAB_SIZE}.pkl")
SVM_PATH = os.path.join(MODELS_DIR, "traditional_svm.pkl")

TOP_K = 5
PROGRESS_EVERY = 2000


def load_models():
    print(f"Loading vocabulary from {KMEANS_PATH}")
    kmeans = joblib.load(KMEANS_PATH)
    print(f"Loading SVM from {SVM_PATH}")
    svm = joblib.load(SVM_PATH)
    return kmeans, svm


def extract_descriptors(pil_image: Image.Image, sift: cv2.SIFT):
    """Extract SIFT descriptors from a PIL image (RGB)."""
    arr = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    return desc


def encode_bovw(desc, kmeans, k: int) -> np.ndarray:
    """Encode SIFT descriptors into an L2-normalized BoVW histogram."""
    if desc is None:
        return np.zeros(k, dtype=np.float32)
    labels = kmeans.predict(desc)
    hist, _ = np.histogram(labels, bins=np.arange(k + 1))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


def encode_variant(df: pd.DataFrame, sift: cv2.SIFT, kmeans, k: int,
                    degradation: str = None, severity=None,
                    variant_name: str = "clean") -> np.ndarray:
    """
    Load each test image (optionally applying a degradation), extract SIFT
    descriptors, and encode into a BoVW feature matrix.
    """
    n = len(df)
    feats = np.zeros((n, k), dtype=np.float32)
    failed = 0

    start = time.time()
    for i, row in enumerate(df.itertuples()):
        try:
            img = Image.open(row.image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            failed += 1
            continue

        if degradation is not None:
            img = apply_degradation(img, degradation, severity)

        desc = extract_descriptors(img, sift)
        feats[i] = encode_bovw(desc, kmeans, k)

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"  [{variant_name}] {i + 1}/{n}, elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"[{variant_name}] encoding done in {elapsed:.1f}s, failed_images={failed}")
    return feats, elapsed


def predict_and_save(df: pd.DataFrame, feats: np.ndarray, svm, out_path: str,
                      batch_size: int = 200) -> None:
    """Run SVM prediction (top-1 and top-5 via decision_function) and
    write results to a CSV in the required format.

    Also computes a top1_confidence column via svm.predict_proba() (the model
    was trained with probability=True, so this works without retraining).
    Confidence is looked up for the specific class chosen by decision_function
    (via svm.classes_), not just the argmax of predict_proba, so pred_idx and
    top1_confidence always refer to the same class even in the rare case the
    two ranking methods would disagree.

    NOTE: sklearn's SVC.decision_function() internally computes a
    one-vs-one score matrix of shape (n_samples, n_classes*(n_classes-1)/2)
    regardless of decision_function_shape. With 500 classes this is
    124,750 columns; calling it on all 5000 test images at once requires
    ~4.65 GB and raises ArrayMemoryError on memory-constrained machines.
    We therefore call decision_function (and predict_proba) in small batches
    and concatenate the resulting top-1 / top-5 / confidence values (not the
    raw score matrices), which keeps peak memory low without changing the
    total computation.
    """
    classes = svm.classes_
    n = feats.shape[0]

    pred_idx_all = np.zeros(n, dtype=np.int64)
    top5_idx_all = np.zeros((n, TOP_K), dtype=np.int64)
    confidence_all = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_scores = svm.decision_function(feats[start:end])  # (batch, n_classes)
        batch_proba = svm.predict_proba(feats[start:end])       # (batch, n_classes)

        batch_pred_idx = np.argmax(batch_scores, axis=1)
        pred_idx_all[start:end] = classes[batch_pred_idx]

        top5_order = np.argsort(-batch_scores, axis=1)[:, :TOP_K]
        top5_idx_all[start:end] = classes[top5_order]

        # confidence = predicted probability of the class actually chosen above
        confidence_all[start:end] = batch_proba[np.arange(end - start), batch_pred_idx]

    rows = []
    for i, row in enumerate(df.itertuples()):
        top5_str = " ".join(str(c) for c in top5_idx_all[i])
        rows.append({
            "image_path": row.image_path,
            "true_idx": row.class_idx,
            "pred_idx": int(pred_idx_all[i]),
            "top5_idx": top5_str,
            "top1_confidence": float(confidence_all[i]),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


def main() -> None:
    os.makedirs(CLEAN_OUT_DIR, exist_ok=True)
    os.makedirs(ROBUST_OUT_DIR, exist_ok=True)

    sift = cv2.SIFT_create()
    kmeans, svm = load_models()

    df_test = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df_test)} test samples from {TEST_CSV}")

    timing = {}

    # ---- Clean test set ----
    print("\n=== Predicting on clean test set ===")
    feats, elapsed = encode_variant(df_test, sift, kmeans, VOCAB_SIZE, variant_name="clean")
    timing["clean_encode_sec"] = elapsed
    predict_and_save(df_test, feats, svm, os.path.join(CLEAN_OUT_DIR, "traditional_predictions.csv"))

    # ---- Robustness: 5 degradations x 3 severities = 15 variants ----
    for degradation, severities in DEGRADATION_LEVELS.items():
        for severity in severities:
            variant_name = f"{degradation}_{severity}"
            print(f"\n=== Predicting on robustness variant: {variant_name} ===")
            feats, elapsed = encode_variant(
                df_test, sift, kmeans, VOCAB_SIZE,
                degradation=degradation, severity=severity,
                variant_name=variant_name,
            )
            timing[f"{variant_name}_encode_sec"] = elapsed
            out_path = os.path.join(ROBUST_OUT_DIR, f"traditional_{variant_name}_predictions.csv")
            predict_and_save(df_test, feats, svm, out_path)

    # ---- Save timing info ----
    timing["total_sec"] = sum(timing.values())
    timing_path = os.path.join(MODELS_DIR, "traditional_predict_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\nTiming info saved to {timing_path}")
    print(json.dumps(timing, indent=2))

    print("\nAll predictions complete.")


if __name__ == "__main__":
    main()
