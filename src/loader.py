import pandas as pd
from pathlib import Path

# Load Dataset
def load_dataset():
    BASE_DIR = Path(__file__).resolve().parent.parent

    dataset_path = BASE_DIR / "dataset" / "heart_disease.csv"
    return pd.read_csv(dataset_path)