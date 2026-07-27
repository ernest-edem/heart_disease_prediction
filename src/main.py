import pandas as pd
from src.loader import load_dataset
from src.preprocessor import (
    dataset_info,
    encoding,
    hot_encoding,
    save_encoded_dataset,
    dataset_statistics
)
from src.generate_reports import (
    save_reports,
    save_text_report
)

from src.feature_engineering import (
    feature_selection,
    split_data,
    scaled_for_logistics
)

from src.train_model import (
    train_logistic_model,
    decision_tree_model,
    random_forest_model
)

from src.save_models import (
    save_model,
    load_model
)

from src.predictions import (
    logistic_prediction,
    tree_prediction,
    forest_prediction
)

from src.model_evaluation import (
    logistic_model_evaluation,
    tree_model_evaluation,
    rf_model_evaluation
)

from src.visualization import (
    age_distribution,
    cholesterol_distribution,
    heart_disease_distribution,
    lg_confusion_matrix,
    tree_confusion_matrix,
    rf_confusion_matrix
)




#Main Function
def main():

    #load dataset>>>>>>>>>>>
    df = load_dataset()

    # Dataset Info
    dataset_info(df)

    #Preprocessor>>>>>>>>>>>>>>
    df_new = encoding(df)

    df_new = hot_encoding(df_new)

    save_encoded_dataset(df_new)

    # Save Report>>>>>>>>>>>>>
    dataset_stats = dataset_statistics(df_new)

    #Dataset Summary
    save_reports(dataset_stats, "dataset_summary")

    print(f"\n{df_new.head()}")
    print(df_new.columns.tolist())

    print(f"\n{dataset_stats}")

    # Feature Selection>>>>>>>>>>>>>
    X, y = feature_selection(df_new)

    # Split Data>>>>>>>>>>
    # Not Scaled
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scaled for Logistics
    lg_X_train, lg_X_test, scaler = scaled_for_logistics(X_train, X_test)

    # Train Models>>>>>>>>>>>>>>>
    # Logistic Model
    logistic_model = train_logistic_model(lg_X_train, y_train)

    # Decision Tree Model
    tree_model = decision_tree_model(X_train, y_train)

    # Random Forest Model
    rf_model = random_forest_model(X_train, y_train)

    #Save and Load Trained Models>>>>>>>>>>>>
    # Save Models
    save_model(logistic_model, "logistic_regression.pkl")
    save_model(tree_model, "decision_tree.pkl")
    save_model(rf_model,"rain_forest.pkl")
    save_model(scaler, "scaler.pkl")

    # Load Models
    logistic_model = load_model("logistic_regression.pkl")
    tree_model = load_model("decision_tree.pkl")
    rf_model = load_model("rain_forest.pkl")
    scaler = load_model("scaler.pkl")


    # Prediction>>>>>>>>>>>
    # Logistics Regression Prediction
    lg_prediction = logistic_prediction(logistic_model, lg_X_test)

    lg_probability = logistic_model.predict_proba(lg_X_test)

    lg_prediction_report = pd.DataFrame({
        "Actual": y_test.values,
        "Prediction": lg_prediction,
        "Correct": y_test.values == lg_prediction,
        "No HeartDisease Probability": f"{lg_probability[0][0]:.2%}",
        "HeartDisease Probability": f"{lg_probability[0][1]:.2%}"
    })
    print(f"\n{lg_prediction_report}")

    # Logistic Regression Prediction Report
    save_reports(lg_prediction_report, "logistic_prediction_report.csv")


    # Decision Tree Prediction
    d_tree_prediction = tree_prediction(tree_model, X_test)

    tree_probability = tree_model.predict_proba(X_test)

    tree_prediction_report = pd.DataFrame({
        "Actual": y_test.values,
        "Prediction": d_tree_prediction,
        "Correct": y_test.values == d_tree_prediction,
        "No HeartDisease Probability": f"{tree_probability[0][0]:.2%}",
        "HeartDisease Probability": f"{tree_probability[0][1]:.2%}"
    })

    print(f"\n{tree_prediction_report}")

    # Decision Tree Prediction Report
    save_reports(tree_prediction_report, "tree_prediction_report.csv")


    # Random Forest Prediction
    rf_prediction = forest_prediction(rf_model, X_test)

    rf_probability = rf_model.predict_proba(X_test)

    rf_prediction_report = pd.DataFrame({
        "Actual": y_test.values,
        "Prediction": rf_prediction,
        "Correct": y_test.values == rf_prediction,
        "No HeartDisease Probability": f"{rf_probability[0][0]:.2%}",
        "HeartDisease Probability": f"{rf_probability[0][1]:.2%}"
    })

    print(f"\n{rf_prediction_report}")

    # Random Forest Prediction Report
    save_reports(rf_prediction_report, "rf_prediction_report.csv")

    # Model Evaluation >>>>>>>>>>>>>>>>>>
    # Logistic Model
    lg_accuracy, lg_matrix, lg_class_report, lg_precision, lg_recall, lg_f1 = logistic_model_evaluation(y_test, lg_prediction)

    lg_tn, lg_fp, lg_fn, lg_tp = lg_matrix.ravel()
    lg_matrix_report = pd.DataFrame({
        "Metrics": [
            "True Negative (TN)",
            "False Positive (FP)",
            "False Negative (FN)",
            "True Positive (TP)"
        ],
        "Values": [
            lg_tn,
            lg_fp,
            lg_fn,
            lg_tp
        ]
    })

    # Save Logistic Classification Report
    save_text_report(lg_class_report, "lg_class_report.txt")


    # Tree Model
    tree_accuracy, tree_matrix, tree_class_report, tree_precision, tree_recall, tree_f1 = tree_model_evaluation(y_test, lg_prediction)

    tree_tn, tree_fp, tree_fn, tree_tp = tree_matrix.ravel()
    tree_matrix_report = pd.DataFrame({
        "Metrics": [
            "True Negative (TN)",
            "False Positive (FP)",
            "False Negative (FN)",
            "True Positive (TP)"
        ],
        "Values": [
            tree_tn,
            tree_fp,
            tree_fn,
            tree_tp
        ]
    })

    # Save Tree Classification Report
    save_text_report(tree_class_report, "tree_class_report.txt")

    # Tree Feature Importance
    tree_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": tree_model.feature_importances_
    })

    tree_importance.sort_values(by="Importance", ascending=False)

    # Decision Tree Feature Report
    save_reports(tree_importance, "tree_feature_importance.csv")

    # Random Forest Model
    rf_accuracy, rf_matrix, rf_class_report, rf_precision, rf_recall, rf_f1 = rf_model_evaluation(y_test, lg_prediction)

    rf_tn, rf_fp, rf_fn, rf_tp = rf_matrix.ravel()
    rf_matrix_report = pd.DataFrame({
        "Metrics": [
            "True Negative (TN)",
            "False Positive (FP)",
            "False Negative (FN)",
            "True Positive (TP)"
        ],
        "Values": [
            rf_tn,
            rf_fp,
            rf_fn,
            rf_tp
        ]
    })

    # Save Random Forest Classification Report
    save_text_report(rf_class_report, "rf_class_report.txt")

    # Random Forest Feature Importance >>>>>>>>>>>>>>>
    rf_importance = pd.DataFrame({
        "Feature": X.columns.values,
        "Importance": rf_model.feature_importances_
    })

    rf_importance.sort_values(by="Importance", ascending=False)

    # Random Forest Feature Report
    save_reports(rf_importance, "rf_feature_importance.csv")

    # Model Comparison >>>>>>>>>>>>>>
    comparison_report = pd.DataFrame({
        "Models": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [f"{lg_accuracy:.2%}", f"{tree_accuracy:.2%}", f"{rf_accuracy:.2%}"],
        "Precision": [f"{lg_precision:.2%}", f"{tree_precision:.2%}", f"{rf_precision:.2%}" ],
        "Recall": [f"{lg_recall:.2%}", f"{tree_recall:.2%}", f"{rf_recall:.2%}"],
        "F1 Score": [f"{lg_f1:.2%}", f"{tree_f1:.2%}", f"{rf_f1:.2%}"]
    })

    print(f"\n{comparison_report}")

    save_reports(comparison_report, "model_comparison.csv")

    #Visualization >>>>>>>>>>>>>>>>>
    age_distribution(df) # Age Distribution
    cholesterol_distribution(df) # Cholesterol Distribution
    heart_disease_distribution(df) # Heart Disease Distribution
    lg_confusion_matrix(lg_matrix_report) # Logistic Regression Confusion Matrix
    tree_confusion_matrix(tree_matrix_report) # Decision Tree Matrix
    rf_confusion_matrix(rf_matrix_report) # Random Forest Matrix

    # =======================
    # Prediction for New Patient
    # ==========================
    new_patient = pd.DataFrame({
        "Age": [54],
        "Sex": [1],
        "RestingBP": [145],
        "Cholesterol": [245],
        "FastingBS": [1],
        "MaxHR": [122],
        "ExerciseAngina": [1],
        "Oldpeak": [2.4],

        # One-hot encoded columns
        "RestingECG_Normal": [1],
        "RestingECG_ST": [0],
        "ChestPainType_ATA": [1],
        "ChestPainType_NAP": [0],
        "ChestPainType_TA": [0],
        "ST_Slope_Flat": [1],
        "ST_Slope_Up": [0]
    })


    # Logistic Regression
    new_patient_scaled = scaler.transform(new_patient)
    lg_np_prediction = logistic_model.predict(new_patient_scaled)
    lg_np_probability = logistic_model.predict_proba(new_patient_scaled)

    print(f"\nNew Patient Prediction: {lg_np_prediction}")

    lg_np_probability_report = pd.DataFrame(
        lg_np_probability,
        columns = ["No HeartDisease", "HeartDisease"]
    )
    print(f"\n{lg_np_probability_report.round(2)}")

    # Decision Tree
    tree_np_prediction = tree_model.predict(new_patient)
    tree_np_probability = tree_model.predict_proba(new_patient)

    print(f"\nNew Patient Prediction: {tree_np_prediction}")

    tree_np_probability_report = pd.DataFrame(
        tree_np_probability,
        columns=["No HeartDisease", "HeartDisease"]
    )
    print(f"\n{tree_np_probability_report.round(2)}")

    # Random Forest
    rf_np_prediction = tree_model.predict(new_patient)
    rf_np_probability = tree_model.predict_proba(new_patient)

    print(f"\nNew Patient Prediction: {rf_np_prediction}")

    rf_np_probability_report = pd.DataFrame(
        rf_np_probability,
        columns=["No HeartDisease", "HeartDisease"]
    )
    print(f"\n{rf_np_probability_report.round(2)}")

    #

if __name__ == "__main__":
    main()