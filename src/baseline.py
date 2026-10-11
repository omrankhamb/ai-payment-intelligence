"""Phase 1: EDA, Strict Time-Based Split, and XGBoost Baselines.

Trains:
- Baseline A: Weak baseline with 8 hand-crafted features (target AUC ~0.769).
- Baseline B: Tabular XGBoost with all ~430 features (target AUC ~0.907).
Evaluates with:
- ROC-AUC
- Average Precision (PR-AUC)
- Precision @ 10% Recall
Logs to MLflow and outputs artifacts/split_boundaries.json and artifacts/baseline_metrics.json.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import xgboost as xgb

from src.features import compute_time_split_indices


def compute_precision_at_recall(
    y_true: np.ndarray, y_pred_prob: np.ndarray, target_recall: float = 0.10
) -> float:
    """Compute precision at target recall (e.g. 10% recall).

    Operational fraud operations metric: evaluates precision at high confidence threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
    # Filter where recall >= target_recall
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) == 0:
        return 0.0
    # The highest threshold yielding >= target_recall gives the maximum precision at that operating point
    return float(np.max(precisions[valid_indices]))


def run_eda(df: pd.DataFrame) -> Dict[str, float]:
    """Perform quick EDA checks."""
    total_tx = len(df)
    fraud_count = int(df["isFraud"].sum())
    fraud_rate = float(df["isFraud"].mean())
    missing_rate_overall = float(df.isna().mean().mean())
    has_device_rate = float((~df["DeviceInfo"].isna() | ~df["DeviceType"].isna()).mean())

    print("\n--- [Phase 1 EDA Summary] ---")
    print(f"Total Transactions: {total_tx:,}")
    print(f"Total Fraud: {fraud_count:,} ({fraud_rate:.4%}) - Expectation: ~3.5%")
    print(f"Transactions with Device Info: {has_device_rate:.4%} - Expectation: ~20%")
    print(f"Average Missing Field Ratio: {missing_rate_overall:.4%}")
    print("-----------------------------\n")

    return {
        "total_transactions": total_tx,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "has_device_rate": has_device_rate,
        "missing_rate_overall": missing_rate_overall,
    }


def prepare_baseline_a_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer ~8 features for Baseline A:

    1. TransactionAmt
    2. log_TransactionAmt
    3. ProductCD_code
    4. card1
    5. card4_code
    6. addr1
    7. transaction_hour
    8. has_device
    """
    feat = pd.DataFrame(index=df.index)
    feat["TransactionAmt"] = df["TransactionAmt"].fillna(0)
    feat["log_TransactionAmt"] = np.log1p(df["TransactionAmt"].clip(lower=0).fillna(0))
    feat["ProductCD_code"] = df["ProductCD"].astype("category").cat.codes
    feat["card1"] = df["card1"].fillna(-1)
    feat["card4_code"] = df["card4"].astype("category").cat.codes
    feat["addr1"] = df["addr1"].fillna(-1)
    # TransactionDT is seconds; calculate hour of day
    feat["transaction_hour"] = (df["TransactionDT"] // 3600) % 24
    feat["has_device"] = (~df["DeviceInfo"].isna() | ~df["DeviceType"].isna()).astype(int)
    return feat


def prepare_baseline_b_features(
    df: pd.DataFrame, train_idx: np.ndarray
) -> pd.DataFrame:
    """Process ~430 features for Baseline B.

    Applies frequency encoding for categoricals fitted strictly on training data.
    """
    drop_cols = ["TransactionID", "isFraud", "TransactionDT"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feat = df[feature_cols].copy()

    for col in feat.select_dtypes(include=["object", "category"]).columns:
        # Compute frequency mapping strictly on train slice to prevent leakage
        freq = df.iloc[train_idx][col].value_counts(normalize=True).to_dict()
        feat[col] = feat[col].map(freq).fillna(0.0).astype(np.float32)

    for col in feat.select_dtypes(include=["number"]).columns:
        feat[col] = feat[col].astype(np.float32)

    return feat


def evaluate_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Calculate ROC-AUC, Average Precision, and Precision@10% Recall."""
    auc = float(roc_auc_score(y_true, y_pred_prob))
    ap = float(average_precision_score(y_true, y_pred_prob))
    p_at_10_recall = compute_precision_at_recall(y_true, y_pred_prob, target_recall=0.10)
    return {
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision_at_10_recall": round(p_at_10_recall, 4),
    }


def run_pipeline(
    data_dir: str = "data",
    artifacts_dir: str = "artifacts",
    mlflow_uri: str = "sqlite:///artifacts/mlflow.db",
):
    """Execute complete Phase 1 pipeline."""
    os.makedirs(artifacts_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("FraudGraph_Phase1_Baselines")

    data_path = Path(data_dir)
    tx_file = data_path / "train_transaction.csv"
    id_file = data_path / "train_identity.csv"

    if not tx_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {tx_file}. Please download IEEE-CIS fraud detection data or specify --data_dir."
        )

    print(f"Loading data from {data_path}...")
    df_tx = pd.read_csv(tx_file)
    if id_file.exists():
        df_id = pd.read_csv(id_file)
        df = df_tx.merge(df_id, on="TransactionID", how="left")
    else:
        print("Warning: Identity table not found; using transaction table only.")
        df = df_tx

    # 1. Quick EDA
    eda_stats = run_eda(df)

    # 2. Strict Time-Based Split (70% train, 10% val, 20% test)
    print("Computing strict time-based split (70% train / 10% val / 20% test)...")
    train_idx, val_idx, test_idx, boundaries = compute_time_split_indices(df)

    split_file = Path(artifacts_dir) / "split_boundaries.json"
    with open(split_file, "w") as f:
        json.dump(boundaries, f, indent=2)
    print(f"Saved split boundaries to {split_file}: {boundaries}")

    y = df["isFraud"].to_numpy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    results = {}

    # 3. Train Baseline A (8 features)
    print("\n--- Training Baseline A: XGBoost (8 engineered features) ---")
    feat_a = prepare_baseline_a_features(df)
    X_train_a = feat_a.iloc[train_idx]
    X_val_a = feat_a.iloc[val_idx]
    X_test_a = feat_a.iloc[test_idx]

    model_a = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=float((len(y_train) - sum(y_train)) / sum(y_train)),
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=15,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="Baseline_A_XGBoost_8feats"):
        model_a.fit(
            X_train_a,
            y_train,
            eval_set=[(X_val_a, y_val)],
            verbose=False,
        )
        val_probs_a = model_a.predict_proba(X_val_a)[:, 1]
        test_probs_a = model_a.predict_proba(X_test_a)[:, 1]

        val_metrics_a = evaluate_predictions(y_val, val_probs_a)
        test_metrics_a = evaluate_predictions(y_test, test_probs_a)

        mlflow.log_params({"features": 8, "model": "xgboost"})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics_a.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics_a.items()})

        results["Baseline_A_Val"] = val_metrics_a
        results["Baseline_A_Test"] = test_metrics_a
        print(f"Baseline A Val Metrics:  {val_metrics_a}")
        print(f"Baseline A Test Metrics: {test_metrics_a} (Target AUC ~0.769)")

    # 4. Train Baseline B (all ~430 features)
    print("\n--- Training Baseline B: XGBoost (All ~430 features) ---")
    feat_b = prepare_baseline_b_features(df, train_idx)
    X_train_b = feat_b.iloc[train_idx]
    X_val_b = feat_b.iloc[val_idx]
    X_test_b = feat_b.iloc[test_idx]

    model_b = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float((len(y_train) - sum(y_train)) / sum(y_train)),
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="Baseline_B_XGBoost_AllFeats"):
        model_b.fit(
            X_train_b,
            y_train,
            eval_set=[(X_val_b, y_val)],
            verbose=False,
        )
        val_probs_b = model_b.predict_proba(X_val_b)[:, 1]
        test_probs_b = model_b.predict_proba(X_test_b)[:, 1]

        val_metrics_b = evaluate_predictions(y_val, val_probs_b)
        test_metrics_b = evaluate_predictions(y_test, test_probs_b)

        mlflow.log_params({"features": len(feat_b.columns), "model": "xgboost"})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics_b.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics_b.items()})

        results["Baseline_B_Val"] = val_metrics_b
        results["Baseline_B_Test"] = test_metrics_b
        print(f"Baseline B Val Metrics:  {val_metrics_b}")
        print(f"Baseline B Test Metrics: {test_metrics_b} (Target AUC ~0.907)")

    # Save summary
    summary_file = Path(artifacts_dir) / "baseline_metrics.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics summary to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Baselines Pipeline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument(
        "--mlflow_uri", type=str, default="sqlite:///artifacts/mlflow.db"
    )
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.artifacts_dir, args.mlflow_uri)
