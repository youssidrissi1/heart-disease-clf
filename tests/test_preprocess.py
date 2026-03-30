import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from preprocess import load_and_clean, split_data

def test_no_nulls_after_cleaning(tmp_path):
    """Test: missing values marked as '?' are removed"""
    csv = tmp_path / "test.csv"
    csv.write_text(
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n"
        "63,1,3,145,233,1,0,150,0,2.3,0,?,1,1\n"
        "37,1,2,130,250,0,1,187,0,3.5,0,0,2,0\n"
    )
    df = load_and_clean(str(csv))
    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning"
    assert len(df) == 1, "Row with '?' should be dropped"

def test_target_is_binary(tmp_path):
    """Test: multi-class target correctly binarized"""
    csv = tmp_path / "test.csv"
    csv.write_text(
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n"
        "63,1,3,145,233,1,0,150,0,2.3,0,0,1,2\n"
        "37,1,2,130,250,0,1,187,0,3.5,0,0,2,0\n"
        "45,0,1,120,220,0,0,160,0,1.5,1,0,1,3\n"
    )
    df = load_and_clean(str(csv))
    assert set(df["target"].unique()).issubset({0, 1}), "Target not binary (0 or 1)"
    assert len(df[df["target"] == 0]) == 1, "Class 0 count incorrect"
    assert len(df[df["target"] == 1]) == 2, "Class 1 count incorrect (targets >0)"

def test_split_sizes():
    """Test: train/val/test split maintains correct proportions"""
    df = pd.DataFrame(np.random.rand(100, 13),
                      columns=[f"f{i}" for i in range(13)])
    df["target"] = (np.random.rand(100) > 0.5).astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    assert len(X_train) + len(X_val) + len(X_test) == 100, "Total samples incorrect"
    assert len(X_train) == 70, f"Train size should be 70, got {len(X_train)}"
    assert len(X_val) == 15, f"Val size should be 15, got {len(X_val)}"
    assert len(X_test) == 15, f"Test size should be 15, got {len(X_test)}"

def test_stratification(tmp_path):
    """Test: stratified split preserves class distribution"""
    csv = tmp_path / "test.csv"
    # Create imbalanced dataset
    csv.write_text(
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n"
        + "\n".join([f"{60+i},1,1,120,200,0,0,150,0,1,0,0,1,1" for i in range(80)])
        + "\n" + "\n".join([f"{40+i},0,0,110,180,0,0,140,0,0.5,0,0,0,0" for i in range(20)])
    )
    df = load_and_clean(str(csv))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=42)
    
    # Check class distribution is similar across splits
    train_pos_ratio = y_train.mean()
    val_pos_ratio = y_val.mean()
    test_pos_ratio = y_test.mean()
    
    assert abs(train_pos_ratio - val_pos_ratio) < 0.15, "Val distribution skewed"
    assert abs(train_pos_ratio - test_pos_ratio) < 0.15, "Test distribution skewed"

def test_feature_types(tmp_path):
    """Test: all features converted to float"""
    csv = tmp_path / "test.csv"
    csv.write_text(
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n"
        "63,1,3,145,233,1,0,150,0,2.3,0,0,1,1\n"
    )
    df = load_and_clean(str(csv))
    for col in df.columns:
        assert df[col].dtype in [np.float32, np.float64], f"Column {col} is not float type"

def test_seedable_splits():
    """Test: same seed produces identical splits"""
    df = pd.DataFrame(np.random.rand(50, 13),
                      columns=[f"f{i}" for i in range(13)])
    df["target"] = (np.random.rand(50) > 0.5).astype(int)
    
    X_train_1, X_val_1, X_test_1, *_ = split_data(df, seed=42)
    X_train_2, X_val_2, X_test_2, *_ = split_data(df, seed=42)
    
    pd.testing.assert_frame_equal(X_train_1, X_train_2, check_names=True)
    pd.testing.assert_frame_equal(X_val_1, X_val_2, check_names=True)
    pd.testing.assert_frame_equal(X_test_1, X_test_2, check_names=True)
