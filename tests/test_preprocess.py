import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from preprocess import load_and_clean, split_data

def test_no_nulls_after_cleaning(tmp_path):
    # Write a tiny synthetic CSV with a '?' to simulate dirty data
    csv = tmp_path / "test.csv"
    csv.write_text(
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n"
        "63,1,3,145,233,1,0,150,0,2.3,0,?,1,1\n"
        "37,1,2,130,250,0,1,187,0,3.5,0,0,2,0\n"
    )
    df = load_and_clean(str(csv))
    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning"

def test_target_is_binary(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text(
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n"
        "63,1,3,145,233,1,0,150,0,2.3,0,0,1,2\n"
        "37,1,2,130,250,0,1,187,0,3.5,0,0,2,0\n"
    )
    df = load_and_clean(str(csv))
    assert set(df["target"].unique()).issubset({0, 1}), "Target not binary"

def test_split_sizes():
    df = pd.DataFrame(np.random.rand(100, 13),
                      columns=[f"f{i}" for i in range(13)])
    df["target"] = (np.random.rand(100) > 0.5).astype(int)
    X_train, X_val, X_test, *_ = split_data(df)
    assert len(X_train) + len(X_val) + len(X_test) == 100
