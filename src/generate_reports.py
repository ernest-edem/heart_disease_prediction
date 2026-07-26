import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)



def save_reports(report: pd.DataFrame, filename: str):
    """
    Save a pandas DataFrame as a CSV report.
    """
    filepath = REPORT_DIR / filename
    report.to_csv(filepath, index=False)

    print(f"✓ Report saved: {filepath}")
    return

