import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

def split_data(data, test_size=0.3, random_state=42):
    try:
        X = data.drop(columns="risk_flag")
        y = data["risk_flag"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state,stratify=y_temp)
        return X_train, X_val,X_test, y_train, y_val, y_test
    
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None
    
def fit_re(X_train, y_train):
    pipeline=ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42))
    ])
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def transform_re(X_val, X_test):
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_val_scaled, X_test_scaled