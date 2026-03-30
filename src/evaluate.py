import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from preprocess import load_and_clean, split_data

def evaluate(data_path, model_path, seed=42):
    df = load_and_clean(data_path)
    _, _, X_test, _, _, y_test = split_data(df, seed=seed)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.savefig("logs/confusion_matrix.png")
    print("Saved: logs/confusion_matrix.png")

    # Error analysis: false negatives (missed disease)
    X_test = X_test.copy()
    X_test["true"] = y_test.values
    X_test["pred"] = preds
    fn = X_test[(X_test["true"] == 1) & (X_test["pred"] == 0)]
    print(f"\nFalse Negatives ({len(fn)}):")
    print(fn.head(3).to_string())

if __name__ == "__main__":
    evaluate(
        data_path="data/heart.csv",
        model_path="models/xgb_seed42_d4.json"
    )
