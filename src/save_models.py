import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Save Trained Models
def save_model(model, filename):
    return joblib.dump(model, MODEL_DIR / filename)

# Load Trained Model
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)