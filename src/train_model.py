from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Train Logistic Model
def train_logistic_model(lg_X_train, y_train):
    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    logistic_model.fit(lg_X_train, y_train)
    return logistic_model

# Train Decision Tree Model
def decision_tree_model(X_train, y_train):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    return tree_model

# Train Random Forest Model
def random_forest_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

