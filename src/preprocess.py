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

def split_data(df: pd.DataFrame, seed: int = 42):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
