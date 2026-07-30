"""
Model evaluation module.

This module provides reusable functions for evaluating
classification models and generating confusion matrix report.
"""

import logging

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Evaluate Classification Model
# ==========================================================
def evaluate_classification_model(
    y_true,
    y_pred,
) -> tuple[
    float,
    np.ndarray,
    str,
    float,
    float,
    float,
]:
    """
    Evaluate a classification model.

    Args:
        y_true:
            Actual target values.

        y_pred:
            Predicted target values.

    Returns:
        accuracy,
        confusion_matrix,
        classification_report,
        precision,
        recall,
        f1_score
    """

    accuracy = accuracy_score(y_true, y_pred)

    matrix = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    logger.info("Model evaluation completed.")

    return (
        accuracy,
        matrix,
        report,
        precision,
        recall,
        f1,
    )


# ==========================================================
# Confusion Matrix Report
# ==========================================================
def create_confusion_matrix_report(
    matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Convert a confusion matrix into a tabular report.

    Args:
        matrix:
            2x2 confusion matrix.

    Returns:
        DataFrame containing TN, FP, FN and TP.
    """

    tn, fp, fn, tp = matrix.ravel()

    return pd.DataFrame({
        "Metric": [
            "True Negative (TN)",
            "False Positive (FP)",
            "False Negative (FN)",
            "True Positive (TP)",
        ],
        "Value": [
            tn,
            fp,
            fn,
            tp,
        ],
    })