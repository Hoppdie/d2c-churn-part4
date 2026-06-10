"""
Train and save the churn model for the FastAPI service.
Run this script before starting the API if model.pkl is not present.

Usage:
    python train_model.py --data_dir /path/to/csv/files/
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, average_precision_score, confusion_matrix)


def main(data_dir: str):
    print(f"Loading data from {data_dir}")
    df = pd.read_csv(f"{data_dir}/rfm_modeling_snapshot.csv")
    print(f"Loaded {len(df):,} rows")

    # Identify columns
    target_col = "churn_next_60d"
    id_cols = ["customer_id", "snapshot_date", "split"]
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    # Encode categoricals
    cat_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dummy_na=True)
    feature_cols_encoded = [c for c in df_encoded.columns if c not in id_cols + [target_col]]

    # Fill missing
    for col in feature_cols_encoded:
        if df_encoded[col].isnull().any():
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

    # Split
    train = df_encoded[df_encoded["split"] == "train"]
    val   = df_encoded[df_encoded["split"] == "validation"]
    test  = df_encoded[df_encoded["split"] == "test"]

    X_train, y_train = train[feature_cols_encoded], train[target_col]
    X_val,   y_val   = val[feature_cols_encoded],   val[target_col]
    X_test,  y_test  = test[feature_cols_encoded],  test[target_col]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Train XGBoost
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # Evaluate on test
    probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.40
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": "XGBoost",
        "threshold": threshold,
        "test_roc_auc": float(roc_auc_score(y_test, probs)),
        "test_f1": float(f1_score(y_test, preds)),
        "test_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        "test_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

    print("\n=== Test Results ===")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churned"]))
    print(f"ROC-AUC: {metrics['test_roc_auc']:.4f}")

    # Save
    joblib.dump(model, "model.pkl")
    joblib.dump(feature_cols_encoded, "feature_cols.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: model.pkl, feature_cols.pkl, metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to CSV directory")
    args = parser.parse_args()
    main(args.data_dir)
