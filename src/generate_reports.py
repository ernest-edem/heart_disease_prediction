import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)


def save_reports(report: pd.DataFrame, filename: str):
    """
    Save a pandas DataFrame as a CSV report.
    """
    filepath = REPORT_DIR / filename
    report.to_csv(filepath, index=False)

    print(f"✓ Report saved: {filepath}")
    return

def save_text_report(text, filename):
    filepath = REPORT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"✓ Report saved: {filepath}")
    return

def save_chart(filename):
    filepath = CHART_DIR / filename

    plt.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Report saved: {filepath}")
    return
