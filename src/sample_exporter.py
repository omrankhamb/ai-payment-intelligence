"""Phase 5: Serving Sample Exporter & Artifact Bootstrapper.

Generates:
1. data/test_sample_full.csv: realistic test transactions (both fraud and legitimate)
   with raw payload fields and 437-dimensional precomputed feature vectors.
2. Initializes lightweight artifacts/ (split_boundaries, entity_maps, feature_stats,
   explainer_stats, model weights, mean embeddings) for immediate local microservice testing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

from src.features import NUMERICAL_COLS, TOTAL_FEATURE_DIM


def generate_synthetic_serving_sample(
    num_samples: int = 100,
    output_path: str = "data/test_sample_full.csv",
) -> pd.DataFrame:
    """Generate realistic test transactions with balanced fraud and legitimate cases."""
    os.makedirs(Path(output_path).parent, exist_ok=True)
    np.random.seed(42)

    records: List[Dict] = []
    base_time = 15000000

    products = ["W", "C", "R", "H", "S"]
    card4_types = ["visa", "mastercard", "discover", "american express"]
    card6_types = ["debit", "credit"]

    for i in range(num_samples):
        is_fraud = 1 if i % 4 == 0 else 0  # 25% fraud in sample for strong evaluation balance
        amt = float(np.random.choice([15.5, 49.99, 120.0, 250.0, 899.99, 2500.0]) if not is_fraud else np.random.choice([499.0, 999.0, 1500.0, 4999.0]))
        card1 = int(np.random.choice([10001, 10002, 10003, 10004, 20005, 30006]))
        addr1 = int(np.random.choice([126, 299, 315, 325, 441]))
        dt = base_time + (i * 120)  # every 2 minutes

        # 437 features
        vec = np.random.randn(TOTAL_FEATURE_DIM).astype(np.float32) * 0.5
        vec[0] = np.log1p(amt)  # scaled TransactionAmt
        if is_fraud:
            vec += 0.8  # shift distribution for fraud

        row = {
            "TransactionID": 3500000 + i,
            "isFraud": is_fraud,
            "TransactionDT": dt,
            "TransactionAmt": amt,
            "ProductCD": np.random.choice(products),
            "card1": card1,
            "card2": 321,
            "card3": 150,
            "card4": np.random.choice(card4_types),
            "card5": 102,
            "card6": np.random.choice(card6_types),
            "addr1": addr1,
            "addr2": 87,
            "P_emaildomain": "gmail.com" if not is_fraud else "anonymous.org",
            "R_emaildomain": "gmail.com" if is_fraud else None,
            "DeviceType": "mobile" if is_fraud else "desktop",
            "DeviceInfo": "iOS Device" if is_fraud else "Windows",
            "features_json": json.dumps(vec.tolist()),
        }
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} test sample records at {output_path}")
    return df


def bootstrap_mock_artifacts(artifacts_dir: str = "artifacts"):
    """Create local artifact package to enable offline verification and microservice serving."""
    art_path = Path(artifacts_dir)
    art_path.mkdir(parents=True, exist_ok=True)

    # 1. Split Boundaries
    split_bounds = {
        "t_train_start": 86400,
        "t_val_start": 10000000,
        "t_test_start": 12500000,
        "t_max": 15811131,
        "n_train": 413378,
        "n_val": 59054,
        "n_test": 118108,
    }
    with open(art_path / "split_boundaries.json", "w") as f:
        json.dump(split_bounds, f, indent=2)

    # 2. Entity Maps
    cards = {f"card_{10000+i}_321_150_visa_102_debit": i for i in range(500)}
    merchants = {f"merch_{p}_{a}": idx for idx, (p, a) in enumerate([("W", 126), ("W", 315), ("C", 299), ("R", 325)])}
    devices = {f"dev_{t}_{d}": idx for idx, (t, d) in enumerate([("mobile", "iOS Device"), ("desktop", "Windows")])}

    entity_maps = {
        "card_to_id": cards,
        "merchant_to_id": merchants,
        "device_to_id": devices,
        "stats": {
            "num_transactions": 590540,
            "num_cards": len(cards),
            "num_merchants": len(merchants),
            "num_devices": len(devices),
            "velocity_lift": 1.68,
        },
    }
    with open(art_path / "entity_maps.json", "w") as f:
        json.dump(entity_maps, f, indent=2)

    # 3. Feature Stats
    feature_stats = {
        "medians": {col: 0.0 for col in NUMERICAL_COLS},
        "means": {col: 0.0 for col in NUMERICAL_COLS},
        "stds": {col: 1.0 for col in NUMERICAL_COLS},
        "cat_vocab": {"ProductCD": 5, "card4": 4, "card6": 2},
    }
    with open(art_path / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    # 4. Explainer Stats
    explainer_stats = {
        "base_fraud_rate": 0.035,
        "card_stats": {
            "card_10001_321_150_visa_102_debit": {
                "count": 45,
                "fraud_count": 8,
                "fraud_rate": 0.1778,
                "median_amt": 85.0,
            },
            "card_20005_321_150_visa_102_debit": {
                "count": 30,
                "fraud_count": 0,
                "fraud_rate": 0.0,
                "median_amt": 45.0,
            },
        },
        "merchant_stats": {
            "merch_C_299": {"count": 120, "fraud_count": 28, "fraud_rate": 0.2333},
        },
        "device_stats": {
            "dev_mobile_iOS Device": {
                "count": 85,
                "fraud_count": 14,
                "fraud_rate": 0.1647,
                "unique_cards": 7,
            },
        },
        "amount_stats": {"median": 68.5, "p95": 320.0, "p99": 780.0},
    }
    with open(art_path / "explainer_stats.json", "w") as f:
        json.dump(explainer_stats, f, indent=2)

    # 5. Model weights & Embeddings
    torch.manual_seed(42)
    hidden_dim = 128
    mean_embeddings = {
        "card": torch.randn(hidden_dim),
        "merchant": torch.randn(hidden_dim),
        "device": torch.randn(hidden_dim),
    }
    torch.save(mean_embeddings, art_path / "mean_embeddings.pt")

    entity_embeddings = {
        "card": torch.randn(len(cards), hidden_dim),
        "merchant": torch.randn(len(merchants), hidden_dim),
        "device": torch.randn(len(devices), hidden_dim),
    }
    torch.save(entity_embeddings, art_path / "entity_embeddings.pt")

    # Weights for inference scoring engine
    linear_tx = torch.randn(hidden_dim, TOTAL_FEATURE_DIM) * 0.05
    weight_head = torch.randn(1, 32) * 0.1
    weights = {
        "tx_encoder.0.weight": linear_tx,
        "tx_encoder.0.bias": torch.zeros(hidden_dim),
        "head.0.weight": torch.randn(32, 64) * 0.1,
        "head.0.bias": torch.zeros(32),
        "head.3.weight": weight_head,
        "head.3.bias": torch.tensor([0.2]),
    }
    torch.save(weights, art_path / "model.pt")

    print(f"Bootstrapped serving artifacts at {art_path}")


if __name__ == "__main__":
    generate_synthetic_serving_sample()
    bootstrap_mock_artifacts()
