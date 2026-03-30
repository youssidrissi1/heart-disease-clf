import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Replace '?' with NaN and drop — actual cleaning step
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)
    return df

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Optional feature engineering: interactions and normalizations.
    Motivated by domain knowledge (cardiology).
    """
    X_eng = X.copy()
    
    # Feature 1: Age-Cholesterol interaction (older + high chol = higher risk)
    X_eng["age_chol_interaction"] = X["age"] * X["chol"] / 1000
    
    # Feature 2: Heart rate to BP ratio
    X_eng["hr_bp_ratio"] = (X["thalach"] + 1) / (X["trestbps"] + 1)
    
    # Feature 3: ST-Age interaction (ST depression more significant in younger patients)
    X_eng["oldpeak_age_interaction"] = X["oldpeak"] * X["age"] / 50
    
    # Feature 4: Normalized cholesterol
    X_eng["chol_normalized"] = (X["chol"] - X["chol"].mean()) / (X["chol"].std() + 1e-8)
    
    return X_eng

def split_data(df: pd.DataFrame, seed: int = 42, use_feature_engineering: bool = False):
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Optional feature engineering
    if use_feature_engineering:
        X = engineer_features(X)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
