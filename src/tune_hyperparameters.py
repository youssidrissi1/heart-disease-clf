"""
Comprehensive hyperparameter tuning via GridSearchCV.
Systematic exploration of XGBoost parameter space.
"""
import json
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
from preprocess import load_and_clean, split_data

def tune_hyperparameters(data_path="data/heart.csv", seed=42):
    """Grid search over XGBoost hyperparameters"""
    
    print("="*70)
    print("HYPERPARAMETER TUNING: GridSearchCV (XGBoost)")
    print("="*70)
    
    # Load data
    df = load_and_clean(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=seed)
    
    # Combine train + val for CV (70+15=85% of data)
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    
    # Parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_lambda': [0.1, 1.0, 10.0],  # L2 regularization
    }
    
    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Base model
    xgb_base = xgb.XGBClassifier(
        n_estimators=200,
        random_state=seed,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    
    # GridSearchCV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("\n[Tuning in progress...]")
    grid_search.fit(X_combined, y_combined)
    
    # Results
    print("\n" + "="*70)
    print("BEST PARAMETERS")
    print("="*70)
    print(f"Best F1-Score (CV): {grid_search.best_score_:.4f}")
    print(f"Best params:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Top 5 models
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['param_max_depth', 'param_learning_rate', 'param_subsample', 
         'param_colsample_bytree', 'param_reg_lambda', 'mean_test_score', 'std_test_score']
    ]
    
    print("\n" + "="*70)
    print("TOP 5 MODELS")
    print("="*70)
    for idx, row in top_5.iterrows():
        print(f"\nRank {top_5.index.get_loc(idx) + 1}:")
        print(f"  F1 (CV): {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Params: max_depth={row['param_max_depth']}, lr={row['param_learning_rate']}, "
              f"subsample={row['param_subsample']}, colsample={row['param_colsample_bytree']}, "
              f"lambda={row['param_reg_lambda']}")
    
    # Train final model with best params
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    best_model = xgb.XGBClassifier(
        n_estimators=200,
        **grid_search.best_params_,
        random_state=seed,
        eval_metric='logloss',
        use_label_encoder=False
    )
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Test evaluation
    from sklearn.metrics import f1_score, roc_auc_score
    test_preds = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\nTest Set Performance:")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Save results
    tuning_results = {
        "best_params": grid_search.best_params_,
        "best_cv_f1": round(float(grid_search.best_score_), 4),
        "test_f1": round(float(test_f1), 4),
        "test_auc": round(float(test_auc), 4),
        "improvement_vs_baseline": round(float(test_f1 - 0.7692), 4),
    }
    
    with open("logs/hyperparameter_tuning.json", "w") as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"\n✓ Results saved: logs/hyperparameter_tuning.json")
    print("="*70)
    
    return best_model

if __name__ == "__main__":
    import pandas as pd
    tune_hyperparameters()
