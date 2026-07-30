"""
Visualization module.

This module contains reusable plotting functions for
exploratory data analysis and model evaluation.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src.generate_reports import save_chart

logger = logging.getLogger(__name__)


# ==========================================================
# Generic Bar Chart
# ==========================================================
def plot_bar_chart(
    labels,
    values,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot and save a bar chart.

    Args:
        labels:
            X-axis labels.

        values:
            Y-axis values.

        title:
            Chart title.

        xlabel:
            X-axis label.

        ylabel:
            Y-axis label.

        filename:
            Output image filename.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.tight_layout()

    save_chart(filename)

    plt.close()

    logger.info("%s saved successfully.", filename)


# ==========================================================
# Age Distribution
# ==========================================================
def age_distribution(df: pd.DataFrame) -> None:

    counts = df["Age"].value_counts().sort_index()

    plot_bar_chart(
        counts.index,
        counts.values,
        "Age Distribution",
        "Age",
        "Frequency",
        "age_distribution.png",
    )


# ==========================================================
# Cholesterol Distribution
# ==========================================================
def cholesterol_distribution(df: pd.DataFrame) -> None:

    counts = df["Cholesterol"].value_counts().sort_index()

    plot_bar_chart(
        counts.index,
        counts.values,
        "Cholesterol Distribution",
        "Cholesterol",
        "Frequency",
        "cholesterol_distribution.png",
    )


# ==========================================================
# Heart Disease Distribution
# ==========================================================
def heart_disease_distribution(
    df: pd.DataFrame,
) -> None:

    counts = (
        df["HeartDisease"]
        .map({0: "No", 1: "Yes"})
        .value_counts()
    )

    plot_bar_chart(
        counts.index,
        counts.values,
        "Heart Disease Distribution",
        "Diagnosis",
        "Frequency",
        "heart_disease_distribution.png",
    )


# ==========================================================
# Confusion Matrix Chart
# ==========================================================
def confusion_matrix_chart(
    matrix_report: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """
    Plot a confusion matrix summary as a bar chart.
    """

    plot_bar_chart(
        matrix_report["Metric"],
        matrix_report["Value"],
        title,
        "Metric",
        "Count",
        filename,
    )




# ==========================================================
# Generic Horizontal Bar Chart
# ==========================================================
def plot_barh_chart(
    labels,
    values,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot and save a bar chart.

    Args:
        labels:
            X-axis labels.

        values:
            Y-axis values.

        title:
            Chart title.

        xlabel:
            X-axis label.

        ylabel:
            Y-axis label.

        filename:
            Output image filename.
    """

    plt.figure(figsize=(8, 5))

    plt.barh(labels, values)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.tight_layout()

    save_chart(filename)

    plt.close()

    logger.info("%s saved successfully.", filename)

# ================================
# Logistic Regression Feature Importance
# ================================
def logistic_feature_chart(
    logistic_report: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """
    Plot a logistic regression feature importance horizontal bar chart.
    """

    plot_barh_chart(
        logistic_report["Feature"],
        logistic_report["Coefficient"],
        title,
        "Metric",
        "Count",
        filename,
    )


# ================================
# Decision Tree Feature Importance
# ================================
def tree_feature_chart(
    tree_feature_report: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """
    Plot a decision tree feature importance horizontal bar chart.
    """

    plot_barh_chart(
        tree_feature_report["Feature"],
        tree_feature_report["Importance"],
        title,
        "Metric",
        "Count",
        filename,
    )

# ================================
# Random Forest Feature Importance
# ================================
def rf_feature_chart(
    rf_feature_report: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """
    Plot a random forest feature importance horizontal bar chart.
    """

    plot_barh_chart(
        rf_feature_report["Feature"],
        rf_feature_report["Importance"],
        title,
        "Metric",
        "Count",
        filename,
    )