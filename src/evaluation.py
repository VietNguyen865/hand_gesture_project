"""Các chỉ số phân loại để so sánh những nhánh nhận dạng cử chỉ."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classification(y_true, y_pred, labels=None):
    """Trả về accuracy, chỉ số macro, chỉ số từng lớp và confusion matrix."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if truth.ndim != 1 or predictions.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(truth) == 0 or len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    if any(
        str(label).strip().casefold() in {"", "nan", "none", "<na>", "nat"}
        for label in truth.tolist() + predictions.tolist()
    ):
        raise ValueError("Labels cannot contain missing or empty values.")

    class_labels = list(labels) if labels is not None else sorted(set(truth.tolist()) | set(predictions.tolist()))
    if not class_labels or len(class_labels) != len(set(class_labels)):
        raise ValueError("labels must be a non-empty sequence of unique classes.")
    unexpected_labels = (set(truth.tolist()) | set(predictions.tolist())) - set(class_labels)
    if unexpected_labels:
        raise ValueError(f"labels does not include observed classes: {sorted(unexpected_labels)}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels": class_labels,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=class_labels).tolist(),
        "per_class": classification_report(
            truth,
            predictions,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        ),
    }
