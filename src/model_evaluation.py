from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Logistic Model Evaluation
def logistic_model_evaluation(y_test, lg_prediction):
    lg_accuracy = accuracy_score(y_test, lg_prediction)
    lg_matrix = confusion_matrix(y_test, lg_prediction)
    lg_class_report = classification_report(y_test, lg_prediction)
    lg_precision = precision_score(y_test, lg_prediction)
    lg_recall = recall_score(y_test, lg_prediction)
    lg_f1 = f1_score(y_test, lg_prediction)
    return  lg_accuracy, lg_matrix, lg_class_report, lg_precision, lg_recall, lg_f1

# Tree Model Evaluation
def tree_model_evaluation(y_test, lg_prediction):
    tree_accuracy = accuracy_score(y_test, lg_prediction)
    tree_matrix = confusion_matrix(y_test, lg_prediction)
    tree_class_report = classification_report(y_test, lg_prediction)
    tree_precision = precision_score(y_test, lg_prediction)
    tree_recall = recall_score(y_test, lg_prediction)
    tree_f1 = f1_score(y_test, lg_prediction)
    return  tree_accuracy, tree_matrix, tree_class_report, tree_precision, tree_recall, tree_f1

#  Random Forest Model Evaluation
def rf_model_evaluation(y_test, lg_prediction):
    rf_accuracy = accuracy_score(y_test, lg_prediction)
    rf_matrix = confusion_matrix(y_test, lg_prediction)
    rf_class_report = classification_report(y_test, lg_prediction)
    rf_precision = precision_score(y_test, lg_prediction)
    rf_recall = recall_score(y_test, lg_prediction)
    rf_f1 = f1_score(y_test, lg_prediction)
    return  rf_accuracy, rf_matrix, rf_class_report, rf_precision, rf_recall, rf_f1