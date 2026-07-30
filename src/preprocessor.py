"""
Data preprocessing module.

This module contains all preprocessing functions used by the
Heart Disease Prediction System, including:

- Dataset inspection
- Dataset validation
- Binary encoding
- One-hot encoding
- Dataset statistics
- Saving the processed dataset
"""

from pathlib import Path

import logging
import pandas as pd
import numpy as np

from src.config import DATASET_DIR

logger = logging.getLogger(__name__)


# ==========================================================
# Dataset Information
# ==========================================================
def dataset_info(df: pd.DataFrame) -> dict:
    """
    Generate basic information about the dataset.

    Args:
        df:
            Input dataset.

    Returns:
        Dictionary containing dataset information.
    """

    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes,
        "missing_values": df.isnull().sum(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    logger.info(
        "Dataset loaded successfully (%d rows, %d columns).",
        df.shape[0],
        df.shape[1],
    )

    return info

#===========
def replace_zero_cholesterol(df: pd.DataFrame) ->pd.DataFrame:
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())
    return df


# ==========================================================
# Dataset Validation
# ==========================================================
def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that all required columns exist.

    Args:
        df:
            Dataset to validate.

    Raises:
        ValueError:
            If one or more required columns are missing.
    """

    required_columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
        "HeartDisease",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# ==========================================================
# Binary Encoding
# ==========================================================
def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary categorical variables.

    Args:
        df:
            Original dataset.

    Returns:
        Encoded dataset.
    """

    df = df.copy()

    df["Sex"] = df["Sex"].map({
        "M": 1,
        "F": 0
    })

    df["ExerciseAngina"] = df["ExerciseAngina"].map({
        "Y": 1,
        "N": 0
    })

    logger.info("Binary encoding completed.")

    return df


# ==========================================================
# One-Hot Encoding
# ==========================================================
def one_hot_encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical features.

    Args:
        df:
            Dataset after binary encoding.

    Returns:
        Encoded dataset.
    """

    categorical_columns = [
        "RestingECG",
        "ChestPainType",
        "ST_Slope",
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
    )

    logger.info("One-hot encoding completed.")

    return df


# ==========================================================
# Complete Preprocessing Pipeline
# ==========================================================
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Args:
        df:
            Raw dataset.

    Returns:
        Fully processed dataset.
    """

    validate_dataset(df)

    df = encode_binary_features(df)

    df = one_hot_encode_features(df)

    return df


# ==========================================================
# Save Encoded Dataset
# ==========================================================
def save_encoded_dataset(df: pd.DataFrame) -> None:
    """
    Save the encoded dataset.

    Args:
        df:
            Encoded dataset.
    """

    output_path = DATASET_DIR / "encoded_dataset.csv"

    df.to_csv(output_path, index=False)

    logger.info(
        "Encoded dataset saved to %s",
        output_path,
    )


# ==========================================================
# Dataset Statistics
# ==========================================================
def dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate dataset summary statistics.

    Args:
        df:
            Processed dataset.

    Returns:
        DataFrame containing summary statistics.
    """

    statistics = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Shape",
            "Total Missing Values",
            "Total Duplicate Rows",
            "Highest Age",
            "Lowest Age",
            "Average Age",
            "Number of Males",
            "Number of Females",
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            str(df.shape),
            df.isnull().sum().sum(),
            df.duplicated().sum(),
            df["Age"].max(),
            df["Age"].min(),
            round(df["Age"].mean(), 2),
            (df["Sex"] == 1).sum(),
            (df["Sex"] == 0).sum(),
        ]
    })

    logger.info("Dataset statistics generated.")

    return statistics