"""
Report generation module.

This module provides utility functions for saving report,
text files, and charts generated throughout the project.
"""

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# ==========================================================
# Project Directories
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Save CSV Report
# ==========================================================
def save_reports(
    report: pd.DataFrame,
    filename: str,
) -> None:
    """
    Save a DataFrame as a CSV file.

    Args:
        report:
            Pandas DataFrame to save.

        filename:
            Output CSV filename.
    """

    filepath = REPORT_DIR / filename

    report.to_csv(
        filepath,
        index=False,
        encoding="utf-8",
    )

    logger.info("Report saved: %s", filepath)


# ==========================================================
# Save Text Report
# ==========================================================
def save_text_report(
    text: str,
    filename: str,
) -> None:
    """
    Save a text report.

    Args:
        text:
            Text content.

        filename:
            Output text filename.
    """

    filepath = REPORT_DIR / filename

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(text)

    logger.info("Text report saved: %s", filepath)


# ==========================================================
# Save Chart
# ==========================================================
def save_chart(
    filename: str,
) -> None:
    """
    Save the current matplotlib figure.

    Args:
        filename:
            Output image filename.
    """

    filepath = CHART_DIR / filename

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Chart saved: %s", filepath)