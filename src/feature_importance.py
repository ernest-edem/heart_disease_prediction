"""
Feature importance module.

This module generates feature importance report for the
machine learning models used in the Heart Disease Prediction System.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Logistic Regression Feature Importance
# ==========================================================
def logistic_feature_importance(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate feature importance report for Logistic Regression.

    Args:
        model:
            Trained Logistic Regression model.

        X:
            Feature dataset.

    Returns:
        DataFrame sorted by coefficient.
    """

    logistic_report = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0].round(2)
    })

    logistic_report = logistic_report.sort_values(
        by="Coefficient",
        ascending=False,
        ignore_index=True,
    )

    logger.info(
        "Logistic Regression feature importance generated."
    )

    return logistic_report


# ==========================================================
# Tree-Based Feature Importance
# ==========================================================
def tree_feature_importance(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate feature importance report for a tree-based model.

    Works for:
        - Decision Tree
        - Random Forest
        - Extra Trees
        - Gradient Boosting
        - XGBoost
        - LightGBM
        - CatBoost

    Args:
        model:
            Trained tree-based model.

        X:
            Feature dataset.

    Returns:
        Sorted feature importance report.
    """

    report = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_.round(2)
    })

    report = report.sort_values(
        by="Importance",
        ascending=False,
        ignore_index=True,
    )

    logger.info(
        "Tree-based feature importance generated."
    )

    return report