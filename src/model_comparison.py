"""
Model comparison module.

This module creates a summary table comparing the performance
of multiple machine learning models.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Model Comparison
# ==========================================================
def model_comparison(
    results: dict,
) -> pd.DataFrame:
    """
    Create a model comparison report.

    Args:
        results:
            Dictionary containing evaluation metrics
            for each trained model.

    Returns:
        DataFrame summarizing model performance.
    """

    comparison = pd.DataFrame(results).T.reset_index()

    comparison = comparison.round(2)

    comparison.rename(
        columns={"index": "Model"},
        inplace=True,
    )

    logger.info(
        "Model comparison report generated."
    )

    return comparison