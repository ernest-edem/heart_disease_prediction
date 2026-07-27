import matplotlib.pyplot as plt
from src.generate_reports import save_chart


def age_distribution(df):
    age_count = df["Age"].value_counts()
    plt.figure(figsize=(8,5))
    plt.bar(age_count.index, age_count.values)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    save_chart("age_distribution")
    return

def cholesterol_distribution(df):
    plt.figure(figsize=(8, 5))
    chol_count = df["Cholesterol"].value_counts()
    plt.bar(chol_count.index, chol_count.values)
    plt.title("Cholesterol Distribution")
    plt.xlabel("Cholesterol")
    plt.ylabel("Frequency")
    save_chart("chol_distribution.png")
    return

def heart_disease_distribution(df):
    df["HeartDisease"] = df["HeartDisease"].map({
        0: "No",
        1: "Yes"
    })

    ht_count = df["HeartDisease"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(ht_count.index, ht_count.values)
    plt.title("Heart Disease Distribution")
    plt.xlabel("Heart Disease")
    plt.ylabel("Frequency")
    save_chart("heart_disease_distribution.png")
    return

def lg_confusion_matrix(lg_matrix_report):
    plt.figure(figsize=(8, 5))
    plt.bar(lg_matrix_report["Metrics"], lg_matrix_report["Values"])
    plt.title("Logistic Regression Matrix")
    plt.xlabel("Matrix")
    plt.ylabel("Values")
    save_chart("lg_confusion_matrix.png")
    return

def tree_confusion_matrix(tree_matrix_report):
    plt.figure(figsize=(8, 5))
    plt.bar(tree_matrix_report["Metrics"], tree_matrix_report["Values"])
    plt.title("Decision Tree Matrix")
    plt.xlabel("Matrix")
    plt.ylabel("Values")
    save_chart("decision_tree_matrix.png")
    return

def rf_confusion_matrix(rf_matrix_report):
    plt.figure(figsize=(8, 5))
    plt.bar(rf_matrix_report["Metrics"], rf_matrix_report["Values"])
    plt.title("Random Forest Matrix")
    plt.xlabel("Matrix")
    plt.ylabel("Values")
    save_chart("random_forest_matrix.png")
    return