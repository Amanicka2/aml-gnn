"""Shared evaluation helpers so every model is scored the exact same way."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(y_true, y_prob):
    """Pick the probability cutoff that maximizes F1. Only ever call this on VALIDATION data."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))  # last point has no threshold
    return float(thr[i]), float(f1[i])


def scores_at(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob) >= threshold
    tp = int((y_pred & (y_true == 1)).sum())
    fp = int((y_pred & (y_true == 0)).sum())
    fn = int((~y_pred & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tp": tp, "fp": fp, "fn": fn,
        "n_pos": int(y_true.sum()), "n": int(len(y_true)),
    }
