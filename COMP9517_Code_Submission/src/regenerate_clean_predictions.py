"""
regenerate_clean_predictions.py

Quick, targeted re-run: regenerates ONLY results/clean/traditional_predictions.csv
with the new top1_confidence column added, as requested by E for error analysis
(high-confidence correct / high-confidence wrong sample filtering).

This does NOT retrain anything and does NOT touch the 15 robustness variants —
it reuses the already-trained model (traditional_kmeans_k512.pkl,
traditional_svm.pkl) and the same functions from predict_traditional.py, just
re-running inference on the clean test set (~5 min, dominated by SIFT
extraction on 5,000 images).

If the team later decides the robustness CSVs also need a top1_confidence
column, simply re-run the full predict_traditional.py (~68 min) instead —
predict_and_save() there has already been updated to include the column.

Usage:
    python src/regenerate_clean_predictions.py
"""

import os

from predict_traditional import (
    load_models,
    encode_variant,
    predict_and_save,
    VOCAB_SIZE,
)
import cv2
import pandas as pd

TEST_CSV = "data/splits/test_paths.csv"
CLEAN_OUT_DIR = "results/clean"


def main() -> None:
    os.makedirs(CLEAN_OUT_DIR, exist_ok=True)

    sift = cv2.SIFT_create()
    kmeans, svm = load_models()

    df_test = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df_test)} test samples from {TEST_CSV}")

    print("\n=== Re-encoding clean test set (same as before, just to feed prediction) ===")
    feats, elapsed = encode_variant(df_test, sift, kmeans, VOCAB_SIZE, variant_name="clean")
    print(f"Encoding done in {elapsed:.1f}s")

    out_path = os.path.join(CLEAN_OUT_DIR, "traditional_predictions.csv")
    predict_and_save(df_test, feats, svm, out_path)

    print(f"\nDone. {out_path} now includes the top1_confidence column.")


if __name__ == "__main__":
    main()
