import logging

import pandas as pd

from src.config import DATASET_PATH


def load_dataset() -> pd.DataFrame:
    """
    Load the heart disease dataset.

    Returns:
        pd.DataFrame:
            Loaded dataset.

    Raises:
        FileNotFoundError:
            If the dataset file cannot be found.
    """

    logging.info("Loading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    logging.info(
        "Dataset loaded successfully (%d rows, %d columns).",
        dataset.shape[0],
        dataset.shape[1],
    )

    return dataset