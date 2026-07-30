"""
Application configuration.

This module contains all configurable constants used throughout
the Heart Disease Prediction System.
Changing a value here updates the entire application.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"
IMAGE_DIR = BASE_DIR / "images"

# ==========================================================
# Dataset
# ==========================================================

DATASET_PATH = DATASET_DIR / "heart_disease.csv"

# ==========================================================
# Machine Learning Parameters
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_ITER = 1000

N_ESTIMATORS = 100

# ==========================================================
# Output Files
# ==========================================================

LOGISTIC_MODEL = MODEL_DIR / "logistic_regression.pkl"

TREE_MODEL = MODEL_DIR / "decision_tree.pkl"

RANDOM_FOREST_MODEL = MODEL_DIR / "random_forest.pkl"

SCALER_MODEL = MODEL_DIR / "scaler.pkl"