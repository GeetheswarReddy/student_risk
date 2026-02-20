from src.data.load_data import load_data
from src.data.split_data import split_data, transform_re
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd


def evaluate():
    # Load the saved model and artifacts
    artifact = joblib.load("models/xgboost_model.pkl")
    model = artifact["model"]
    scaler = artifact["scaler"]
    threshold = artifact["threshold"]
    columns = artifact["columns"]

    # Load and prepare the data
    data = load_data("data/raw_data/raw.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    # Categorical encoding
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    X_val = pd.get_dummies(X_val, columns=categorical_columns, dtype=float).reindex(columns=columns, fill_value=0)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=float).reindex(columns=columns, fill_value=0)

    #evaluae on test set

    y_pred_proba_xgb = model.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_pred_proba_xgb >= 0.4).astype(int)
    print("XGBoost Test Results:")
    print(classification_report(y_test, y_pred_xgb))
    print(confusion_matrix(y_test, y_pred_xgb))
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)
    print("ROC-AUC:", roc_auc)

if __name__ == "__main__":
    evaluate()