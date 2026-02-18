from src.data.load_data import load_data
from src.data.split_data import split_data, fit_re, transform_re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

FILE_PATH = "data/raw_data/raw.csv"
THRESHOLD = 0.4

def train():

    data = load_data(FILE_PATH)

    #we wrote this function
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    # categorical encoding — fit on training data only to avoid data leakage
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train = pd.get_dummies(X_train, columns=categorical_columns, dtype=float)
    X_val = pd.get_dummies(X_val, columns=categorical_columns, dtype=float).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=float).reindex(columns=X_train.columns, fill_value=0)

    # 4. Scale + SMOTE on training data
    X_train_res, y_train_res, scaler = fit_re(X_train, y_train)

    # 5. Scale val/test using the same scaler
    X_val_scaled, X_test_scaled = transform_re(X_val, X_test, scaler)

    # 6. Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_res, y_train_res)

    # 7. Evaluate
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)
    print("Validation Results:")
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))

    # 8. Save
    artifact = {"model": model, "scaler": scaler, "threshold": THRESHOLD, "columns": X_train.columns.tolist()}
    joblib.dump(artifact, "models/logistic_regression_model.pkl")
    print("Model saved to models/logistic_regression_model.pkl")

if __name__ == "__main__":
    train()
