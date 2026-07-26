

# Prediction - Logistic Regression
def logistic_prediction(logistic_model, X):
    return logistic_model.predict(X)

# Prediction - Decision Tree
def tree_prediction(tree_model, X):
    return tree_model.predict(X)

# Prediction - Random Forest
def forest_prediction(rf_model, X):
    return rf_model.predict(X)