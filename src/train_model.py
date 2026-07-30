"""
Model training module.

This module contains functions for training all machine
learning models used in the Heart Disease Prediction System.

Models:
- Logistic Regression
- Decision Tree
- Random Forest
"""

import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    MAX_ITER,
    RANDOM_STATE,
    N_ESTIMATORS,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Logistic Regression
# ==========================================================
def train_logistic_model(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Args:
        X_train:
            Scaled training features.

        y_train:
            Training labels.

    Returns:
        Trained LogisticRegression model.
    """

    model = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    logger.info("Logistic Regression model trained successfully.")

    return model


# ==========================================================
# Decision Tree
# ==========================================================
def train_decision_tree_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    Args:
        X_train:
            Training feature matrix.

        y_train:
            Training labels.

    Returns:
        Trained DecisionTreeClassifier.
    """

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    logger.info("Decision Tree model trained successfully.")

    return model


# ==========================================================
# Random Forest
# ==========================================================
def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train:
            Training feature matrix.

        y_train:
            Training labels.

    Returns:
        Trained RandomForestClassifier.
    """

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    logger.info("Random Forest model trained successfully.")

    return model