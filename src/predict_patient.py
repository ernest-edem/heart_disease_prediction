"""
Patient prediction module.

This module predicts heart disease risk for one or more new
patients using all trained machine learning models.
"""

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ==========================================================
# Predict New Patient
# ==========================================================
def predict_patient(
    patient_data: pd.DataFrame,
    scaler: StandardScaler,
    logistic_model,
    decision_tree_model,
    random_forest_model,
) -> pd.DataFrame:
    """
    Predict heart disease risk for new patient(s).

    Args:
        patient_data:
            DataFrame containing one or more new patients.

        scaler:
            Trained StandardScaler.

        logistic_model:
            Trained Logistic Regression model.

        decision_tree_model:
            Trained Decision Tree model.

        random_forest_model:
            Trained Random Forest model.

    Returns:
        DataFrame summarizing predictions and probabilities.
    """

    # Scale only for Logistic Regression
    patient_scaled = scaler.transform(patient_data)

    # Logistic Regression
    logistic_prediction = logistic_model.predict(patient_scaled)[0]
    logistic_probability = (
        logistic_model.predict_proba(patient_scaled)[0][1]
    )

    # Decision Tree
    tree_prediction = decision_tree_model.predict(patient_data)[0]
    tree_probability = (
        decision_tree_model.predict_proba(patient_data)[0][1]
    )

    # Random Forest
    forest_prediction = random_forest_model.predict(patient_data)[0]
    forest_probability = (
        random_forest_model.predict_proba(patient_data)[0][1]
    )

    report = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
        ],
        "Prediction": [
            logistic_prediction,
            tree_prediction,
            forest_prediction,
        ],
        "Heart Disease Probability": [
            round(logistic_probability, 4),
            round(tree_probability, 4),
            round(forest_probability, 4),
        ],
    })

    logger.info(
        "Prediction completed for %d patient(s).",
        len(patient_data),
    )

    return report