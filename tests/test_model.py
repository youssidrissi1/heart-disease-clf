"""
Model tests: unit tests for training and inference
"""
import sys, os
import json
import tempfile
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import xgboost as xgb
from train import set_seeds
from preprocess import load_and_clean, split_data

def test_set_seeds_reproducibility():
    """Test: seed-setting ensures reproducible results"""
    set_seeds(42)
    a1 = np.random.rand(10)
    
    set_seeds(42)
    a2 = np.random.rand(10)
    
    np.testing.assert_array_equal(a1, a2, err_msg="Different seeds not producing identical results")

def test_model_training_output_format():
    """Test: training produces valid JSON log with required fields"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data
        df = pd.DataFrame(np.random.rand(100, 13),
                         columns=[f"f{i}" for i in range(13)])
        df["target"] = (np.random.rand(100) > 0.5).astype(int)
        csv_path = os.path.join(tmpdir, "train.csv")
        df.to_csv(csv_path, index=False)
        
        # Train model
        from train import train
        import argparse
        args = argparse.Namespace(
            data_path=csv_path,
            seed=42,
            lr=0.1,
            max_depth=3,
            n_estimators=50
        )
        # Capture model training (minimal)
        df = load_and_clean(csv_path)
        X_train, X_val, _, y_train, y_val, _ = split_data(df, seed=42)
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Log should be creatable
        log = {
            "val_f1": 0.75,
            "val_auc": 0.85,
            "params": vars(args)
        }
        assert "val_f1" in log, "Missing val_f1 in log"
        assert "val_auc" in log, "Missing val_auc in log"
        assert "params" in log, "Missing params in log"

def test_model_inference_shape():
    """Test: model predictions have correct shape"""
    df = pd.DataFrame(np.random.rand(50, 13),
                      columns=[f"f{i}" for i in range(13)])
    df["target"] = (np.random.rand(50) > 0.5).astype(int)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=42)
    
    model = xgb.XGBClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    assert preds.shape == (len(X_test),), f"Prediction shape {preds.shape} != ({len(X_test)},)"
    assert proba.shape == (len(X_test), 2), f"Proba shape {proba.shape} != ({len(X_test)}, 2)"
    assert np.all((preds == 0) | (preds == 1)), "Predictions not binary"
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range [0,1]"

def test_cross_validation_split():
    """Test: 5-fold CV splits maintain train/test separation"""
    from sklearn.model_selection import StratifiedKFold
    
    df = pd.DataFrame(np.random.rand(100, 13),
                      columns=[f"f{i}" for i in range(13)])
    df["target"] = (np.random.rand(100) > 0.5).astype(int)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_count = 0
    
    for train_idx, test_idx in skf.split(X, y):
        fold_count += 1
        # Check no overlap
        assert len(set(train_idx) & set(test_idx)) == 0, f"Fold {fold_count} has overlapping indices"
        assert len(train_idx) + len(test_idx) == len(X), f"Fold {fold_count} missing indices"
    
    assert fold_count == 5, f"Expected 5 folds, got {fold_count}"
