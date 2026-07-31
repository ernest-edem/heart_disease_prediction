"""
Main entry point for the Heart Disease Prediction System.
"""

import pandas as pd

from src.loader import load_dataset

from src.preprocessor import (
    dataset_info,
    preprocess_dataset,
    save_encoded_dataset,
    dataset_statistics,
    replace_zero_cholesterol
)

from src.feature_engineering import (
    feature_selection,
    split_data,
    scale_features,
)

from src.generate_reports import (
    save_reports,
    save_text_report,
)

from src.train_model import (
    train_logistic_model,
    train_decision_tree_model,
    train_random_forest_model,
)

from src.save_models import (
    save_model,
    load_model,
)

from src.predictions import (
    logistic_prediction,
    tree_prediction,
    forest_prediction,
)

from src.model_evaluation import (
    evaluate_classification_model,
    create_confusion_matrix_report,
)

from src.visualization import (
    age_distribution,
    cholesterol_distribution,
    heart_disease_distribution,
    confusion_matrix_chart,
    logistic_feature_chart,
    tree_feature_chart,
    rf_feature_chart
)

from src.feature_importance import (
    logistic_feature_importance,
    tree_feature_importance,
)

from src.sample_data import create_patient

from src.model_comparison import model_comparison

from src.predict_patient import predict_patient


# ==========================================================
# Main Function
# ==========================================================
def main():
    """Execute the complete machine learning pipeline."""

    print("\n==============================================")
    print("Heart Disease Prediction Pipeline Started")
    print("==============================================")

    # ======================================================
    # Load Dataset
    # ======================================================
    print("\nLoading dataset...")

    df = load_dataset()

    dataset_info(df)

    # ======================================================
    # Data Preprocessing
    # ======================================================
    print("\nPreprocessing dataset...")

    replace_zero_cholesterol(df)

    df_new = preprocess_dataset(df)

    save_encoded_dataset(df_new)

    dataset_stats = dataset_statistics(df_new)

    save_reports(
        dataset_stats,
        "dataset_summary.csv",
    )

    # ======================================================
    # Feature Engineering
    # ======================================================
    print("\nPreparing features...")

    X, y = feature_selection(df_new)

    X_train, X_test, y_train, y_test = split_data(X, y)

    lg_X_train, lg_X_test, scaler = scale_features(
        X_train,
        X_test,
    )

    # ======================================================
    # Train Models
    # ======================================================
    print("\nTraining models...")

    logistic_model = train_logistic_model(
        lg_X_train,
        y_train,
    )

    tree_model = train_decision_tree_model(
        X_train,
        y_train,
    )

    rf_model = train_random_forest_model(
        X_train,
        y_train,
    )

    # ======================================================
    # Save Models
    # ======================================================
    print("\nSaving trained models...")

    save_model(
        logistic_model,
        "logistic_regression.pkl",
    )

    save_model(
        tree_model,
        "decision_tree.pkl",
    )

    save_model(
        rf_model,
        "random_forest.pkl",
    )

    save_model(
        scaler,
        "standard_scaler.pkl",
    )

    # ======================================================
    # Reload Models
    # ======================================================
    print("\nLoading trained models...")

    logistic_model = load_model(
        "logistic_regression.pkl"
    )

    tree_model = load_model(
        "decision_tree.pkl"
    )

    rf_model = load_model(
        "random_forest.pkl"
    )

    scaler = load_model(
        "standard_scaler.pkl"
    )

    # ======================================================
    # Predictions
    # ======================================================
    print("\nGenerating predictions...")

    (
        lg_prediction,
        lg_probability,
        lg_prediction_report,
    ) = logistic_prediction(
        logistic_model,
        lg_X_test,
    )

    save_reports(
        lg_prediction_report,
        "logistic_prediction_report.csv",
    )

    (
        tree_prediction_result,
        tree_probability,
        tree_prediction_report,
    ) = tree_prediction(
        tree_model,
        X_test,
    )

    save_reports(
        tree_prediction_report,
        "tree_prediction_report.csv",
    )

    (
        rf_prediction,
        rf_probability,
        rf_prediction_report,
    ) = forest_prediction(
        rf_model,
        X_test,
    )

    save_reports(
        rf_prediction_report,
        "random_forest_prediction_report.csv",
    )


    # ======================================================
    # Model Evaluation
    # ======================================================
    print("\nEvaluating models...")

    (
        lg_accuracy,
        lg_matrix,
        lg_class_report,
        lg_precision,
        lg_recall,
        lg_f1,
    ) = evaluate_classification_model(
        y_test,
        lg_prediction,
    )

    (
        tree_accuracy,
        tree_matrix,
        tree_class_report,
        tree_precision,
        tree_recall,
        tree_f1,
    ) = evaluate_classification_model(
        y_test,
        tree_prediction_result,
    )

    (
        rf_accuracy,
        rf_matrix,
        rf_class_report,
        rf_precision,
        rf_recall,
        rf_f1,
    ) = evaluate_classification_model(
        y_test,
        rf_prediction,
    )

    save_text_report(
        lg_class_report,
        "logistic_classification_report.txt",
    )

    save_text_report(
        tree_class_report,
        "decision_tree_classification_report.txt",
    )

    save_text_report(
        rf_class_report,
        "random_forest_classification_report.txt",
    )

    # ======================================================
    # Confusion Matrix Reports
    # ======================================================
    lg_matrix_report = create_confusion_matrix_report(
        lg_matrix
    )

    tree_matrix_report = create_confusion_matrix_report(
        tree_matrix
    )

    rf_matrix_report = create_confusion_matrix_report(
        rf_matrix
    )

    # ======================================================
    # Feature Importance
    # ======================================================
    print("\nGenerating feature importance report...")

    logistic_feature_report = logistic_feature_importance(
            logistic_model,
            X
    )

    save_reports(
        logistic_feature_report,
        "logistic_feature_importance.csv",
    )

    tree_feature_report = tree_feature_importance(
            tree_model,
            X,
    )
    save_reports(
        tree_feature_report,
        "decision_tree_feature_importance.csv",
    )

    rf_feature_report =  tree_feature_importance(
            rf_model,
            X,
    )

    save_reports(
       rf_feature_report,
        "random_forest_feature_importance.csv",
    )

    # ======================================================
    # Model Comparison
    # ======================================================
    print("\nComparing models...")

    results = {
        "Logistic Regression": {
            "Accuracy": lg_accuracy,
            "Precision": lg_precision,
            "Recall": lg_recall,
            "F1 Score": lg_f1,
        },
        "Decision Tree": {
            "Accuracy": tree_accuracy,
            "Precision": tree_precision,
            "Recall": tree_recall,
            "F1 Score": tree_f1,
        },
        "Random Forest": {
            "Accuracy": rf_accuracy,
            "Precision": rf_precision,
            "Recall": rf_recall,
            "F1 Score": rf_f1,
        },
    }

    comparison = model_comparison(results)

    save_reports(
        comparison,
        "model_comparison.csv",
    )

    # ======================================================
    # Visualizations
    # ======================================================
    print("\nGenerating charts...")

    age_distribution(df)

    cholesterol_distribution(df)

    heart_disease_distribution(df)


    confusion_matrix_chart(
        lg_matrix_report,
        "Logistic Regression Confusion Matrix",
        "logistic_confusion_matrix.png",
    )

    confusion_matrix_chart(
        tree_matrix_report,
        "Decision Tree Confusion Matrix",
        "decision_tree_confusion_matrix.png",
    )

    confusion_matrix_chart(
        rf_matrix_report,
        "Random Forest Confusion Matrix",
        "random_forest_confusion_matrix.png",
    )


    logistic_feature_chart(
        logistic_feature_report,
        "Logistic Regression Feature Importance",
        "logistic_feature_importance.png",
    )

    tree_feature_chart(
        tree_feature_report,
        "Decision Tree Feature Importance",
        "tree_feature_importance.png",
    )

    rf_feature_chart(
        rf_feature_report,
        "Random Forest Feature Importance",
        "rf_feature_importance.png",
    )

    # ======================================================
    # Predict New Patient
    # ======================================================
    print("\nPredicting new patient...")

    patient = create_patient(
        age=60,
        sex=1,
        resting_bp=150,
        cholesterol=260,
        fasting_bs=1,
        max_hr=115,
        exercise_angina=1,
        oldpeak=2.7,
        resting_ecg_normal=1,
        resting_ecg_st=0,
        chest_pain_ata=1,
        chest_pain_nap=0,
        chest_pain_ta=0,
        st_slope_flat=1,
        st_slope_up=0,
    )

    patient_report = predict_patient(
        patient,
        scaler,
        logistic_model,
        tree_model,
        rf_model,
    )

    save_reports(
        patient_report,
        "new_patient_prediction.csv",
    )

    print("\n==============================================")
    print("Pipeline completed successfully.")
    print("==============================================")


# ==========================================================
# Run Application
# ==========================================================
if __name__ == "__main__":
    main()