"""
Feature engineering module.

This module is responsible for:

- Feature selection
- Train/test splitting
- Feature scaling
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Feature Selection
# ==========================================================
def feature_selection(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target.

    Args:
        df:
            Processed dataset.

    Returns:
        Tuple containing:
            X -> Feature matrix
            y -> Target vector
    """

    if "HeartDisease" not in df.columns:
        raise ValueError(
            "'HeartDisease' column not found in dataset."
        )

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    logger.info(
        "Feature selection completed (%d features).",
        X.shape[1],
    )

    return X, y


# ==========================================================
# Train/Test Split
# ==========================================================
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset into training and testing sets.

    Args:
        X:
            Feature matrix.

        y:
            Target vector.

    Returns:
        X_train,
        X_test,
        y_train,
        y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Dataset split completed "
        "(Train=%d, Test=%d).",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test


# ==========================================================
# Feature Scaling
# ==========================================================
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Scale numerical features using StandardScaler.

    NOTE:
        Only models that depend on feature scaling
        (e.g. Logistic Regression, SVM, KNN)
        should use these scaled datasets.

    Args:
        X_train:
            Training feature matrix.

        X_test:
            Testing feature matrix.

    Returns:
        X_train_scaled,
        X_test_scaled,
        fitted StandardScaler
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    logger.info("Feature scaling completed.")

    return X_train_scaled, X_test_scaled, scaler