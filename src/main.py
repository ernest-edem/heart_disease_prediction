import pandas as pd
from src.loader import load_dataset
from src.preprocessor import (
    dataset_info,
    encoding,
    hot_encoding,
    save_encoded_dataset,
    dataset_statistics
)
from src.generate_reports import (save_reports)

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

    # Logistic Regression Prediction Report
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

    # Logistic Regression Prediction Report
    save_reports(rf_prediction_report, "rf_prediction_report.csv")




if __name__ == "__main__":
    main()