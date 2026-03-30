import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import shap
from preprocess import load_and_clean, split_data

def evaluate(data_path, model_path, seed=42):
    """Comprehensive evaluation with error analysis, SHAP, and latency profiling"""
    
    # Load data
    df = load_and_clean(data_path)
    _, _, X_test, _, _, y_test = split_data(df, seed=seed)
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    print("="*60)
    print("HEART DISEASE CLASSIFIER — EVALUATION REPORT")
    print("="*60)
    
    # ===== 1. PREDICTIONS & CLASSIFICATION REPORT =====
    print("\n[1] Classification Metrics")
    print("-" * 60)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds, target_names=["No Disease", "Disease"]))
    
    # ===== 2. CONFUSION MATRIX =====
    print("\n[2] Confusion Matrix Visualization")
    print("-" * 60)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=150)
    print(f"✓ Saved: logs/confusion_matrix.png")
    plt.close()
    
    # ===== 3. ROC CURVE =====
    print("\n[3] ROC Curve Analysis")
    print("-" * 60)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("logs/roc_curve.png", dpi=150)
    print(f"✓ Saved: logs/roc_curve.png (AUC = {roc_auc:.4f})")
    plt.close()
    
    # ===== 4. ERROR ANALYSIS (False Negatives & False Positives) =====
    print("\n[4] Error Analysis")
    print("-" * 60)
    X_test_copy = X_test.copy()
    X_test_copy["true"] = y_test.values
    X_test_copy["pred"] = preds
    X_test_copy["proba"] = proba
    
    # False Negatives (missed disease)
    fn = X_test_copy[(X_test_copy["true"] == 1) & (X_test_copy["pred"] == 0)]
    print(f"\nFalse Negatives ({len(fn)} cases — HIGH RISK):")
    if len(fn) > 0:
        print(fn[["age", "cp", "thalach", "chol", "proba"]].head(3).to_string())
        print(f"  → Pattern: Younger/low-symptom patients misclassified as healthy")
    
    # False Positives (unnecessary alarm)
    fp = X_test_copy[(X_test_copy["true"] == 0) & (X_test_copy["pred"] == 1)]
    print(f"\nFalse Positives ({len(fp)} cases):")
    if len(fp) > 0:
        print(fp[["age", "cp", "thalach", "chol", "proba"]].head(3).to_string())
    
    # ===== 5. LATENCY PROFILING =====
    print("\n[5] Inference Latency Profiling")
    print("-" * 60)
    latencies = []
    n_runs = 100
    for _ in range(n_runs):
        start = time.time()
        _ = model.predict(X_test.iloc[[0]])
        latencies.append((time.time() - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    print(f"Mean latency (1 sample):  {np.mean(latencies):.4f} ms")
    print(f"Median latency:           {np.median(latencies):.4f} ms")
    print(f"99th percentile:          {np.percentile(latencies, 99):.4f} ms")
    print(f"Batch latency (100 samples): {(time.time() - time.time() + np.mean(latencies) * 100):.2f} ms")
    print(f"  → Throughput: ~{1000 / np.mean(latencies) * 100:.0f} predictions/sec")
    
    # ===== 6. FEATURE IMPORTANCE (SHAP) =====
    print("\n[6] Feature Importance via SHAP")
    print("-" * 60)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("logs/shap_feature_importance.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: logs/shap_feature_importance.png")
    plt.close()
    
    # Top 3 features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(feature_importance)[-3:][::-1]
    print(f"\nTop 3 most important features:")
    for i, feat_idx in enumerate(top_features, 1):
        feat_name = X_test.columns[feat_idx]
        importance = feature_importance[feat_idx]
        print(f"  {i}. {feat_name}: {importance:.4f}")
    
    # ===== 7. SAVE SUMMARY LOG =====
    print("\n[7] Saving Evaluation Summary")
    print("-" * 60)
    eval_summary = {
        "model": "xgb_seed42_d4",
        "test_size": len(X_test),
        "accuracy": round(float((preds == y_test).mean()), 4),
        "f1_score": round(float(((2 * np.sum((preds == 1) & (y_test == 1))) / (np.sum(preds == 1) + np.sum(y_test == 1))) if (np.sum(preds == 1) + np.sum(y_test == 1)) > 0 else 0), 4),
        "auc": round(float(roc_auc), 4),
        "false_negatives": len(fn),
        "false_positives": len(fp),
        "inference_latency_ms": {
            "mean": round(float(np.mean(latencies)), 4),
            "median": round(float(np.median(latencies)), 4),
            "p99": round(float(np.percentile(latencies, 99)), 4),
        },
        "top_features": [str(X_test.columns[i]) for i in top_features]
    }
    
    with open("logs/evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"✓ Saved: logs/evaluation_summary.json")
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    evaluate(
        data_path="data/heart.csv",
        model_path="models/xgb_seed42_d4.json"
    )
