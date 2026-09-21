"""Evaluation metrics for rain mask predictions."""

from __future__ import annotations

import numpy as np


def _to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.bool_)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union between a predicted and ground-truth binary
    mask. Returns 1.0 if both masks are empty (nothing to detect, nothing
    detected)."""
    pred = _to_binary(pred_mask)
    gt = _to_binary(gt_mask)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def precision_recall_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 score between a predicted and ground-truth
    binary mask."""
    pred = _to_binary(pred_mask)
    gt = _to_binary(gt_mask)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)
