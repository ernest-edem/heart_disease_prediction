from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature Selection
def feature_selection(df_new):
    X = df_new.drop(columns=["HeartDisease"])
    y = df_new["HeartDisease"]
    return X, y

# Split Dataset
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

# Scaled Data for Logistics
def scaled_for_logistics(X_train, X_test):
    scaler = StandardScaler()

    lg_X_train = scaler.fit_transform(X_train)
    lg_X_test = scaler.transform(X_test)

    return lg_X_train, lg_X_test, scaler
