from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred, labels=None, average="weighted") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tps = np.diag(cm).astype(float)
    fps = cm.sum(axis=0).astype(float) - tps
    fns = cm.sum(axis=1).astype(float) - tps
    tns = cm.sum().astype(float) - (tps + fps + fns)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = tps / (tps + fns)
        per_class_specificity = tns / (tns + fps)
        per_class_far = fps / (fps + tns)
    per_class_recall = np.nan_to_num(per_class_recall)
    per_class_specificity = np.nan_to_num(per_class_specificity)
    per_class_far = np.nan_to_num(per_class_far)
    per_class_bci = 0.5 * (per_class_recall + per_class_specificity)

    # Binary and multiclass both use weighted metrics for manuscript comparability.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "macro_DR": float(per_class_recall.mean()),
        "macro_FAR": float(per_class_far.mean()),
        "macro_BCI": float(per_class_bci.mean()),
        "TP_sum": int(tps.sum()),
        "FP_sum": int(fps.sum()),
        "TN_sum": int(tns.sum()),
        "FN_sum": int(fns.sum()),
        "confusion_matrix": cm,
    }
