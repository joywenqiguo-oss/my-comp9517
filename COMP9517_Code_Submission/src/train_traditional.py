"""
train_traditional.py

Traditional baseline model: SIFT + Bag-of-Visual-Words (BoVW) + SVM.

Pipeline:
  1. Pass 1: Extract SIFT descriptors from all training images and build a
     fixed-size random sample (reservoir sampling) for vocabulary learning.
     This avoids holding all descriptors from all images in memory at once,
     which was found to cause MemoryError on machines with limited RAM.
  2. Train a MiniBatchKMeans vocabulary (visual words) on the sampled
     descriptors.
  3. Pass 2: Re-extract SIFT descriptors per training image and immediately
     encode each image into a fixed-length, L2-normalized BoVW histogram.
  4. Train an SVM (kernel='rbf', C=10) on the resulting histograms.
  5. Save the trained KMeans vocabulary and SVM model to results/models/,
     so predict_traditional.py can load them without retraining.

Final hyperparameters below were chosen after grid search across
vocabulary size (k=128/256/512) and SVM (kernel x C) on the validation
set. See reports/B_traditional_model_dev_log.md for full experiment logs.

Usage:
    python src/train_traditional.py
"""

import os
import time
import json
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Config (final values chosen from experimentation; see dev log)
# ---------------------------------------------------------------------------
TRAIN_CSV = "data/splits/train_paths.csv"
MODELS_DIR = "results/models"
FEATURES_DIR = "results/features"

VOCAB_SIZE = 512          # k in k-means (final chosen value)
RESERVOIR_SAMPLE_SIZE = 200_000  # descriptors sampled for vocabulary training
SVM_KERNEL = "rbf"
SVM_C = 10

RANDOM_SEED = 9517
PROGRESS_EVERY = 2000     # print progress every N images


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_vocabulary_sample(df: pd.DataFrame, sift: cv2.SIFT) -> np.ndarray:
    """
    Pass 1: stream through every training image, extract SIFT descriptors,
    and maintain a fixed-size reservoir sample of descriptors for
    vocabulary training. Descriptors are discarded per image after the
    reservoir update, so memory stays bounded regardless of dataset size.
    """
    reservoir = np.zeros((RESERVOIR_SAMPLE_SIZE, 128), dtype=np.float32)
    filled = 0
    seen_total = 0
    failed = 0

    start = time.time()
    for idx, row in enumerate(df.itertuples()):
        img = cv2.imread(row.image_path)
        if img is None:
            failed += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            failed += 1
            continue

        desc = desc.astype(np.float32)
        for d in desc:
            seen_total += 1
            if filled < RESERVOIR_SAMPLE_SIZE:
                reservoir[filled] = d
                filled += 1
            else:
                j = random.randint(0, seen_total - 1)
                if j < RESERVOIR_SAMPLE_SIZE:
                    reservoir[j] = d

        if (idx + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"  [vocab-sample] {idx + 1}/{len(df)} images, "
                  f"seen_descriptors={seen_total}, filled={filled}, "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"[vocab-sample] done in {elapsed:.1f}s, failed_images={failed}, "
          f"sample_size={filled}")
    return reservoir[:filled]


def encode_bovw(desc: Optional[np.ndarray], kmeans: MiniBatchKMeans, k: int) -> np.ndarray:
    """Encode a single image's SIFT descriptors into an L2-normalized
    BoVW histogram of length k."""
    if desc is None:
        return np.zeros(k, dtype=np.float32)
    labels = kmeans.predict(desc)
    hist, _ = np.histogram(labels, bins=np.arange(k + 1))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


def encode_dataset(df: pd.DataFrame, sift: cv2.SIFT, kmeans: MiniBatchKMeans,
                    k: int, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pass 2: re-extract SIFT descriptors per image and immediately encode
    into a BoVW histogram. Feature matrix is pre-allocated to avoid vstack
    memory spikes.
    """
    n = len(df)
    feats = np.zeros((n, k), dtype=np.float32)
    labels_arr = np.zeros(n, dtype=np.int64)
    failed = 0

    start = time.time()
    for i, row in enumerate(df.itertuples()):
        img = cv2.imread(row.image_path)
        labels_arr[i] = row.class_idx

        if img is None:
            failed += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = sift.detectAndCompute(gray, None)
        feats[i] = encode_bovw(desc, kmeans, k)

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"  [{split_name}-encode] {i + 1}/{n}, elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"[{split_name}-encode] done in {elapsed:.1f}s, failed_images={failed}")
    return feats, labels_arr


def main() -> None:
    set_seeds(RANDOM_SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    timing = {}
    sift = cv2.SIFT_create()

    df_train = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df_train)} training samples from {TRAIN_CSV}")

    # ---- Pass 1: build vocabulary sample ----
    print("\n=== Pass 1: building vocabulary sample (reservoir sampling) ===")
    t0 = time.time()
    sample_descriptors = build_vocabulary_sample(df_train, sift)
    timing["pass1_reservoir_sampling_sec"] = time.time() - t0
    np.save(os.path.join(MODELS_DIR, "kmeans_sample_descriptors.npy"), sample_descriptors)

    # ---- Train k-means vocabulary ----
    print(f"\n=== Training MiniBatchKMeans vocabulary (k={VOCAB_SIZE}) ===")
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=1000,
                              random_state=RANDOM_SEED, n_init=3)
    kmeans.fit(sample_descriptors)
    timing["kmeans_train_sec"] = time.time() - t0
    print(f"KMeans trained in {timing['kmeans_train_sec']:.1f}s, "
          f"inertia={kmeans.inertia_:.2f}")
    joblib.dump(kmeans, os.path.join(MODELS_DIR, f"traditional_kmeans_k{VOCAB_SIZE}.pkl"))

    # ---- Pass 2: encode training images into BoVW histograms ----
    print(f"\n=== Pass 2: encoding training images (k={VOCAB_SIZE}) ===")
    t0 = time.time()
    X_train, y_train = encode_dataset(df_train, sift, kmeans, VOCAB_SIZE, "train")
    timing["pass2_encode_train_sec"] = time.time() - t0
    np.save(os.path.join(FEATURES_DIR, f"train_feat_k{VOCAB_SIZE}.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, "train_labels.npy"), y_train)

    # ---- Train final SVM ----
    print(f"\n=== Training SVM (kernel={SVM_KERNEL}, C={SVM_C}) ===")
    t0 = time.time()
    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    timing["svm_train_sec"] = time.time() - t0
    print(f"SVM trained in {timing['svm_train_sec']:.1f}s")
    joblib.dump(clf, os.path.join(MODELS_DIR, "traditional_svm.pkl"))

    # ---- Save timing info for report ----
    timing["total_sec"] = sum(timing.values())
    timing_path = os.path.join(MODELS_DIR, "traditional_train_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\nTiming info saved to {timing_path}:")
    print(json.dumps(timing, indent=2))

    print("\nDone. Saved artifacts:")
    print(f"  - {os.path.join(MODELS_DIR, f'traditional_kmeans_k{VOCAB_SIZE}.pkl')}")
    print(f"  - {os.path.join(MODELS_DIR, 'traditional_svm.pkl')}")
    print(f"  - {os.path.join(FEATURES_DIR, f'train_feat_k{VOCAB_SIZE}.npy')}")
    print(f"  - {os.path.join(FEATURES_DIR, 'train_labels.npy')}")


if __name__ == "__main__":
    main()