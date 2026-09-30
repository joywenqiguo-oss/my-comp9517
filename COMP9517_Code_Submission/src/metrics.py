"""Evaluation metrics for COMP9517 classification experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def top1_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def topk_accuracy(y_true: Sequence[int], y_topk: Sequence[Sequence[int]], k: int = 5) -> float:
    correct = 0
    for true_label, topk in zip(y_true, y_topk):
        if int(true_label) in [int(x) for x in list(topk)[:k]]:
            correct += 1
    return float(correct / max(len(y_true), 1))


def macro_prf(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def per_class_accuracy(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> List[float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    accs: List[float] = []
    for c in range(num_classes):
        mask = y_true_arr == c
        if mask.sum() == 0:
            accs.append(float("nan"))
        else:
            accs.append(float((y_pred_arr[mask] == c).mean()))
    return accs


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_top5: Sequence[Sequence[int]] | None = None,
    num_classes: int | None = None,
) -> dict:
    if num_classes is None:
        num_classes = int(max(max(y_true), max(y_pred))) + 1
    metrics = {
        "top1_accuracy": top1_accuracy(y_true, y_pred),
        "overall_accuracy": top1_accuracy(y_true, y_pred),
    }
    if y_top5 is not None:
        metrics["top5_accuracy"] = topk_accuracy(y_true, y_top5, k=5)
    metrics.update(macro_prf(y_true, y_pred))
    metrics["per_class_accuracy"] = per_class_accuracy(y_true, y_pred, num_classes)
    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    ).tolist()
    return metrics


def save_metrics(metrics: dict, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
