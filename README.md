# ❤️ Heart Disease Prediction using Machine Learning

A production-ready Machine Learning project that predicts the likelihood of heart disease using patient clinical data. The project implements multiple supervised learning algorithms, evaluates their performance using standard classification metrics, and provides predictions for new patients.

---

## 📌 Project Overview

Heart disease remains one of the leading causes of death worldwide. Early detection allows healthcare professionals to provide timely treatment and reduce mortality.

This project builds and compares several machine learning classification models capable of predicting whether a patient is at risk of heart disease based on clinical attributes.

The project follows a modular architecture to encourage maintainability, scalability, and production readiness.

---

## 🎯 Objectives

* Perform exploratory data analysis (EDA)
* Preprocess clinical data
* Engineer and select relevant features
* Train multiple classification models
* Evaluate model performance
* Compare algorithms
* Predict heart disease risk for new patients
* Save reports, visualizations, and trained models

---

# 📁 Project Structure

```text
HeartDiseasePrediction/
│
├── dataset/
│   ├── heart_disease.csv
│   └── README.md
│
├── charts/
│   ├── age_distribution.png
│   ├── cholesterol_distribution.png
│   ├── heart_disease_distribution.png
│   ├── confusion_matrix_lg.png
│   ├── confusion_matrix_dt.png
│   ├── confusion_matrix_rf.png
│   ├── feature_importance_dt.png
│   ├── feature_importance_rf.png
│   └── model_comparison.png
│
├── reports/
│   ├── dataset_summary.csv
│   ├── descriptive_statistics.csv
│   ├── logistic_regression_results.csv
│   ├── decision_tree_results.csv
│   ├── random_forest_results.csv
│   ├── model_comparison.csv
│   ├── classification_report_lg.txt
│   ├── classification_report_dt.txt
│   └── classification_report_rf.txt
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── prediction.py
│   ├── utils.py
│   └── config.py
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# 🧠 Machine Learning Models

The project compares multiple supervised learning algorithms.

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

Each model is evaluated using identical datasets for fair comparison.

---

# 📊 Evaluation Metrics

The following metrics are calculated for every model.

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

# 📈 Visualizations

The project automatically generates visualizations including:

* Dataset distribution
* Histograms
* Bar charts
* Heart disease distribution
* Feature importance
* Confusion matrices
* Model comparison charts

Generated figures are saved inside the **charts/** directory.

---

# 🗂 Reports

Generated reports include:

* Dataset Summary
* Descriptive Statistics
* Prediction Results
* Classification Reports
* Model Comparison
* Evaluation Metrics

Reports are stored inside the **reports/** folder.

---

# 💾 Saved Models

After training, the following artifacts can be saved:

* Logistic Regression model
* Decision Tree model
* Random Forest model
* Feature scaler

These files are stored in the **models/** directory for future inference without retraining.

---

# ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/ernest-edem/patient_risk_prediction.git
```

Move into the project directory.

```bash
cd patient_risk_prediction
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
```

Activate it.

```bash
.venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Execute:

```bash
python main.py
```

The program will automatically:

* Load the dataset
* Preprocess data
* Engineer features
* Split training/testing data
* Scale features
* Train models
* Evaluate models
* Generate reports
* Generate charts
* Predict new patient outcomes

---

# 📦 Dependencies

Major libraries used include:

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

---

# 📋 Example Prediction

Input:

```text
Age               : 54
Sex               : Male
Resting BP        : 145
Cholesterol       : 245
Fasting Blood Sugar : 1
Maximum Heart Rate : 122
Exercise Angina   : Yes
Old Peak          : 2.4
```

Output:

```text
Prediction : Heart Disease
Probability: 92.7%
```

---

# 🚀 Future Improvements

Future enhancements include:

* Hyperparameter tuning
* Cross-validation
* XGBoost
* LightGBM
* CatBoost
* SHAP Explainability
* Model deployment using FastAPI
* Docker containerization
* CI/CD pipeline
* Web dashboard
* Cloud deployment

---

# 📚 Dataset

The project uses the Heart Disease clinical dataset containing patient demographic and cardiovascular health information.

Typical features include:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

Target:

* HeartDisease (0 = No Disease, 1 = Heart Disease)

---

# 👨‍💻 Author

**Ernest Edem**

Machine Learning Engineer | Software Engineer | AI Enthusiast

GitHub:
https://github.com/ernest-edem

---

# 📄 License

This project is released under the MIT License.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
