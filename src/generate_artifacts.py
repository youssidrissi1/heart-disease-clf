"""
Generate evaluation artifacts: ROC, SHAP, summary
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score
import xgboost as xgb
import shap
from preprocess import load_and_clean, split_data

df = load_and_clean("data/heart.csv")
_, _, X_test, _, _, y_test = split_data(df, seed=42)

model = xgb.XGBClassifier()
model.load_model("models/xgb_seed42_d4.json")

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
print("[*] Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Test Set')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("logs/roc_curve.png", dpi=150)
print("✓ Saved: logs/roc_curve.png")
plt.close()

# SHAP
print("[*] Generating SHAP feature importance...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("logs/shap_feature_importance.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: logs/shap_feature_importance.png")
    plt.close()
except Exception as e:
    print(f"⚠ SHAP error (continuing): {e}")

# Evaluation summary
print("[*] Generating evaluation summary...")
f1 = f1_score(y_test, preds)
summary = {
    "model": "xgb_seed42_d4",
    "test_size": len(X_test),
    "f1_score": round(float(f1), 4),
    "auc": round(float(roc_auc), 4),
    "false_negatives": int(((y_test == 1) & (preds == 0)).sum()),
    "false_positives": int(((y_test == 0) & (preds == 1)).sum()),
}
with open("logs/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: logs/evaluation_summary.json")
print("\n✓ DONE: All evaluation artifacts generated")
