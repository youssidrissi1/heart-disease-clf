"""
Cross-validation and hyperparameter tuning with multiple models.
Demonstrates systematic model comparison for production deployment.
"""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from preprocess import load_and_clean, split_data

def run_cross_validation(data_path="data/heart.csv", seed=42):
    """5-fold stratified cross-validation with multiple models"""
    
    print("="*70)
    print("CROSS-VALIDATION: MODEL COMPARISON (5-Fold Stratified)")
    print("="*70)
    
    # Load data
    df = load_and_clean(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Define models
    models = {
        "XGBoost (seed42_d4)": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed
        ),
        "XGBoost (seed42_d3)": xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed
        ),
        "XGBoost (seed42_d5)": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed
        ),
        "Random Forest (n=100)": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1
        ),
        "Logistic Regression (C=1)": LogisticRegression(
            max_iter=1000, random_state=seed, n_jobs=-1
        ),
    }
    
    # Scorers
    scorers = {
        "F1": make_scorer(f1_score),
        "AUC": make_scorer(roc_auc_score),
    }
    
    # Run CV
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for model_name, model in models.items():
        print(f"\n[{model_name}]")
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scorers, n_jobs=1)
        
        f1_scores = cv_results["test_F1"]
        auc_scores = cv_results["test_AUC"]
        
        results[model_name] = {
            "f1_mean": round(float(f1_scores.mean()), 4),
            "f1_std": round(float(f1_scores.std()), 4),
            "auc_mean": round(float(auc_scores.mean()), 4),
            "auc_std": round(float(auc_scores.std()), 4),
            "f1_fold": [round(float(s), 4) for s in f1_scores],
            "auc_fold": [round(float(s), 4) for s in auc_scores],
        }
        
        print(f"  F1:  {results[model_name]['f1_mean']:.4f} ± {results[model_name]['f1_std']:.4f}")
        print(f"  AUC: {results[model_name]['auc_mean']:.4f} ± {results[model_name]['auc_std']:.4f}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<30} {'F1 (mean±std)':<20} {'AUC (mean±std)':<20}")
    print("-"*70)
    for model_name, res in results.items():
        f1_str = f"{res['f1_mean']:.4f}±{res['f1_std']:.4f}"
        auc_str = f"{res['auc_mean']:.4f}±{res['auc_std']:.4f}"
        print(f"{model_name:<30} {f1_str:<20} {auc_str:<20}")
    
    # Save results
    with open("logs/cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: logs/cv_results.json")
    
    # Recommendation
    print("\n" + "="*70)
    best_model = max(results.items(), key=lambda x: x[1]["f1_mean"])
    print(f"🎯 RECOMMENDED MODEL: {best_model[0]}")
    print(f"   F1-Score: {best_model[1]['f1_mean']:.4f}")
    print("="*70)

if __name__ == "__main__":
    run_cross_validation()
