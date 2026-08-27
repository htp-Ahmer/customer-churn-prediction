"""Model evaluation helpers used by the training script and dashboard."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model: Any, x_test, y_test) -> dict:
    """Return portfolio-friendly classification metrics and curve data."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions).tolist()
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix,
        "roc_curve": {
            "fpr": np.asarray(fpr).tolist(),
            "tpr": np.asarray(tpr).tolist(),
            "thresholds": np.asarray(thresholds).tolist(),
        },
    }