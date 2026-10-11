"""Phase 2: Heterogeneous Graph Construction & Velocity Edge Linking.

Builds PyTorch Geometric HeteroData with:
- Node types:
  - 'transaction' (carrying 437 standardized features)
  - 'card' (embedding index)
  - 'merchant' (embedding index)
  - 'device' (embedding index)
- Forward and reverse edges:
  - ('transaction', 'uses_card', 'card') & reverse
  - ('transaction', 'at_merchant', 'merchant') & reverse
  - ('transaction', 'on_device', 'device') & reverse
  - ('transaction', 'velocity_cluster', 'transaction') & reverse
- Strict temporal forward velocity edges (< 300s on same card).
- Verifies graph statistics and velocity fraud rate lift (target >= 1.6x).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None

from src.features import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TOTAL_FEATURE_DIM,
    get_card_key,
    get_device_key,
    get_merchant_key,
)


def build_hetero_graph(
    df: pd.DataFrame,
    split_boundaries: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Construct HeteroData graph and entity mappings.

    Ensures no feature leakage by fitting scalers only on training transactions.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    print("Sorting transactions strictly by TransactionDT...")
    df = df.sort_values(by="TransactionDT").reset_index(drop=True)
    n_tx = len(df)
    t_val_start = split_boundaries["t_val_start"]
    train_mask = (df["TransactionDT"] < t_val_start).to_numpy()

    print(f"Total transactions: {n_tx:,} (Train: {train_mask.sum():,})")

    # 1. Feature Preprocessing (Fit only on Train)
    print("Preparing 437-dimensional transaction feature matrix...")
    X_num = np.zeros((n_tx, len(NUMERICAL_COLS)), dtype=np.float32)
    medians = {}
    stds = {}
    means = {}

    for idx, col in enumerate(NUMERICAL_COLS):
        if col in df.columns:
            train_vals = df.loc[train_mask, col].dropna()
            med = float(train_vals.median()) if len(train_vals) > 0 else 0.0
            medians[col] = med
            filled = df[col].fillna(med).to_numpy()
            if col == "TransactionAmt":
                filled = np.log1p(np.clip(filled, 0, None))
            X_num[:, idx] = filled
        else:
            medians[col] = 0.0

    # Fit StandardScaler on train slice
    scaler = StandardScaler()
    scaler.fit(X_num[train_mask])
    X_num_scaled = scaler.transform(X_num).astype(np.float32)

    means_dict = {col: float(m) for col, m in zip(NUMERICAL_COLS, scaler.mean_)}
    stds_dict = {col: float(s) for col, s in zip(NUMERICAL_COLS, scaler.scale_)}

    # Categorical vocabulary
    cat_vocab = {}
    X_cat = np.zeros((n_tx, len(CATEGORICAL_COLS)), dtype=np.float32)
    for idx, col in enumerate(CATEGORICAL_COLS):
        if col in df.columns:
            unique_train_vals = df.loc[train_mask, col].dropna().unique().tolist()
            val_to_code = {val: i + 1 for i, val in enumerate(unique_train_vals)}
            cat_vocab[col] = val_to_code
            encoded = df[col].map(val_to_code).fillna(0).to_numpy()
            X_cat[:, idx] = encoded
        else:
            cat_vocab[col] = {}

    # Concatenate features and pad or slice to exact 437
    X_tx = np.hstack([X_num_scaled, X_cat])
    if X_tx.shape[1] < TOTAL_FEATURE_DIM:
        pad = np.zeros((n_tx, TOTAL_FEATURE_DIM - X_tx.shape[1]), dtype=np.float32)
        X_tx = np.hstack([X_tx, pad])
    else:
        X_tx = X_tx[:, :TOTAL_FEATURE_DIM]

    feature_stats = {
        "medians": medians,
        "means": means_dict,
        "stds": stds_dict,
        "cat_vocab": {k: len(v) for k, v in cat_vocab.items()},
    }

    # 2. Entity Mapping & Edge Extraction
    print("Extracting canonical entity keys and bipartite edges...")
    card_map: Dict[str, int] = {}
    merch_map: Dict[str, int] = {}
    dev_map: Dict[str, int] = {}

    tx_to_card: List[Tuple[int, int]] = []
    tx_to_merch: List[Tuple[int, int]] = []
    tx_to_dev: List[Tuple[int, int]] = []

    # Map cards to transaction indices for velocity linking
    card_to_txs: Dict[int, List[Tuple[int, int]]] = {}

    for i, row in df.iterrows():
        # Card
        c_key = get_card_key(row)
        if c_key:
            c_id = card_map.setdefault(c_key, len(card_map))
            tx_to_card.append((i, c_id))
            card_to_txs.setdefault(c_id, []).append((i, int(row["TransactionDT"])))

        # Merchant: (ProductCD, addr1)
        m_key = get_merchant_key(row)
        if m_key:
            m_id = merch_map.setdefault(m_key, len(merch_map))
            tx_to_merch.append((i, m_id))

        # Device: (DeviceType, DeviceInfo)
        d_key = get_device_key(row)
        if d_key:
            d_id = dev_map.setdefault(d_key, len(dev_map))
            tx_to_dev.append((i, d_id))

    # 3. Velocity Cluster Edges (same card, dt < 300s, forward in time)
    print("Constructing velocity burst edges (same card within 5 minutes)...")
    velocity_edges: List[Tuple[int, int]] = []
    for c_id, tx_list in card_to_txs.items():
        if len(tx_list) < 2:
            continue
        # tx_list is already time-ordered because df is sorted
        for idx1 in range(len(tx_list)):
            t1_idx, t1_time = tx_list[idx1]
            for idx2 in range(idx1 + 1, len(tx_list)):
                t2_idx, t2_time = tx_list[idx2]
                delta = t2_time - t1_time
                if delta <= 300:
                    velocity_edges.append((t1_idx, t2_idx))
                else:
                    break

    # 4. Check Velocity Signal Lift
    velocity_tx_ids = set([src for src, _ in velocity_edges] + [dst for _, dst in velocity_edges])
    base_fraud_rate = float(df["isFraud"].mean())
    if len(velocity_tx_ids) > 0:
        vel_fraud_rate = float(df.iloc[list(velocity_tx_ids)]["isFraud"].mean())
        vel_lift = vel_fraud_rate / base_fraud_rate if base_fraud_rate > 0 else 1.0
    else:
        vel_fraud_rate = base_fraud_rate
        vel_lift = 1.0

    print("\n--- [Graph Verification Stats] ---")
    print(f"Transactions:     {n_tx:,}")
    print(f"Distinct Cards:   {len(card_map):,}")
    print(f"Distinct Merchs:  {len(merch_map):,}")
    print(f"Distinct Devices: {len(dev_map):,}")
    print(f"uses_card edges:     {len(tx_to_card):,}")
    print(f"at_merchant edges:   {len(tx_to_merch):,}")
    print(f"on_device edges:     {len(tx_to_dev):,}")
    print(f"velocity edges:      {len(velocity_edges):,}")
    print(f"Base Fraud Rate:     {base_fraud_rate:.4%}")
    print(f"Velocity Fraud Rate: {vel_fraud_rate:.4%}")
    print(f"Velocity Lift:       {vel_lift:.2f}x (Target: >= 1.60x)")
    print("----------------------------------\n")

    entity_maps = {
        "card_to_id": card_map,
        "merchant_to_id": merch_map,
        "device_to_id": dev_map,
        "stats": {
            "num_transactions": n_tx,
            "num_cards": len(card_map),
            "num_merchants": len(merch_map),
            "num_devices": len(dev_map),
            "velocity_lift": round(vel_lift, 4),
        },
    }

    # 5. Build PyG HeteroData or Dict Representation
    data = HeteroData() if HeteroData is not None else {}

    def to_edge_index(edges_list):
        if len(edges_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        arr = np.array(edges_list, dtype=np.int64).T
        return torch.from_numpy(arr)

    y_tensor = torch.from_numpy(df["isFraud"].to_numpy(dtype=np.float32))
    x_tensor = torch.from_numpy(X_tx)

    e_uses_card = to_edge_index(tx_to_card)
    e_at_merch = to_edge_index(tx_to_merch)
    e_on_dev = to_edge_index(tx_to_dev)
    e_vel = to_edge_index(velocity_edges)

    if HeteroData is not None:
        data["transaction"].x = x_tensor
        data["transaction"].y = y_tensor
        data["transaction"].train_mask = torch.from_numpy(train_mask)
        data["card"].num_nodes = len(card_map)
        data["merchant"].num_nodes = len(merch_map)
        data["device"].num_nodes = len(dev_map)

        # Forward edges
        data["transaction", "uses_card", "card"].edge_index = e_uses_card
        data["transaction", "at_merchant", "merchant"].edge_index = e_at_merch
        data["transaction", "on_device", "device"].edge_index = e_on_dev
        data["transaction", "velocity_cluster", "transaction"].edge_index = e_vel

        # Reverse edges (8 edge types total)
        data["card", "rev_uses_card", "transaction"].edge_index = e_uses_card.flip(0)
        data["merchant", "rev_at_merchant", "transaction"].edge_index = e_at_merch.flip(0)
        data["device", "rev_on_device", "transaction"].edge_index = e_on_dev.flip(0)
        data["transaction", "rev_velocity_cluster", "transaction"].edge_index = e_vel.flip(0)
    else:
        data = {
            "x_tx": x_tensor,
            "y": y_tensor,
            "edges": {
                "uses_card": e_uses_card,
                "at_merchant": e_at_merch,
                "on_device": e_on_dev,
                "velocity": e_vel,
            },
            "num_nodes": {
                "transaction": n_tx,
                "card": len(card_map),
                "merchant": len(merch_map),
                "device": len(dev_map),
            },
        }

    # Save artifacts
    print("Saving graph.pt and metadata to artifacts...")
    torch.save(data, artifacts_path / "graph.pt")

    with open(artifacts_path / "entity_maps.json", "w") as f:
        json.dump(entity_maps, f)

    with open(artifacts_path / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    return data, entity_maps, feature_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Heterogeneous FraudGraph")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    split_file = Path(args.artifacts_dir) / "split_boundaries.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split boundaries missing at {split_file}. Run Phase 1 first.")

    with open(split_file) as f:
        split_bounds = json.load(f)

    tx_p = Path(args.data_dir) / "train_transaction.csv"
    id_p = Path(args.data_dir) / "train_identity.csv"
    df_raw = pd.read_csv(tx_p)
    if id_p.exists():
        df_raw = df_raw.merge(pd.read_csv(id_p), on="TransactionID", how="left")

    build_hetero_graph(df_raw, split_bounds, args.artifacts_dir)
