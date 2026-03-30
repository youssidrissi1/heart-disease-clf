import argparse, json, os, random
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from preprocess import load_and_clean, split_data

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def train(args):
    set_seeds(args.seed)

    df = load_and_clean(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=args.seed)

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.8,
        eval_metric="logloss",
        random_state=args.seed,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    f1  = f1_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_proba)

    log = {
        "val_f1": round(f1, 4),
        "val_auc": round(auc, 4),
        "params": vars(args),
    }
    print(json.dumps(log, indent=2))

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("logs/train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    model.save_model(f"models/xgb_seed{args.seed}_d{args.max_depth}.json")
    print(f"Model saved: models/xgb_seed{args.seed}_d{args.max_depth}.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    default="data/heart.csv")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--max_depth",    type=int,   default=4)
    p.add_argument("--n_estimators", type=int,   default=200)
    train(p.parse_args())
