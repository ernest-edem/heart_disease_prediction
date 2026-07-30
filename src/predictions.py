"""
Prediction module.

This module contains functions for making predictions using
trained machine learning models.
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Logistic Regression Prediction
# ==========================================================
def logistic_prediction(
    model,
    X_test,
):
    """
    Make predictions using the Logistic Regression model.

    Args:
        model:
            Trained Logistic Regression model.

        X_test:
            Scaled testing features.

    Returns:
        prediction,
        probability,
        prediction_report
    """

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)

    prediction_report = pd.DataFrame({
        "Prediction": prediction,
        "No HeartDisease Probability": probability[:, 0].round(4),
        "HeartDisease Probability": probability[:, 1].round(4),
    })

    logger.info("Logistic Regression predictions completed.")

    return prediction, probability, prediction_report


# ==========================================================
# Decision Tree Prediction
# ==========================================================
def tree_prediction(
    model,
    X_test,
):
    """
    Make predictions using the Decision Tree model.
    """

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)

    prediction_report = pd.DataFrame({
        "Prediction": prediction,
        "No HeartDisease Probability": probability[:, 0].round(4),
        "HeartDisease Probability": probability[:, 1].round(4),
    })

    logger.info("Decision Tree predictions completed.")

    return prediction, probability, prediction_report


# ==========================================================
# Random Forest Prediction
# ==========================================================
def forest_prediction(
    model,
    X_test,
):
    """
    Make predictions using the Random Forest model.
    """

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)

    prediction_report = pd.DataFrame({
        "Prediction": prediction,
        "No HeartDisease Probability": probability[:, 0].round(4),
        "HeartDisease Probability": probability[:, 1].round(4),
    })

    logger.info("Random Forest predictions completed.")


    return prediction, probability, prediction_report