"""
Model persistence module.

This module provides utility functions for saving and
loading trained machine learning models.
"""

from pathlib import Path
import logging

import joblib

logger = logging.getLogger(__name__)

# ==========================================================
# Project Directories
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Save Model
# ==========================================================
def save_model(
    model,
    filename: str,
) -> Path:
    """
    Save a trained model to disk.

    Args:
        model:
            Trained machine learning model or transformer.

        filename:
            Output filename.

    Returns:
        Path to the saved model.
    """

    filepath = MODEL_DIR / filename

    joblib.dump(model, filepath)

    logger.info("Model saved: %s", filepath)

    return filepath


# ==========================================================
# Load Model
# ==========================================================
def load_model(filename: str):
    """
    Load a trained model from disk.

    Args:
        filename:
            Model filename.

    Returns:
        Loaded machine learning model.
    """

    filepath = MODEL_DIR / filename

    model = joblib.load(filepath)

    logger.info("Model loaded: %s", filepath)

    return model