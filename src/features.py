"""FraudGraph Feature Engineering & Entity Key Resolution.

Shared contract between training (Notebooks 01-05) and serving (src/serving/*).
Ensures zero train/serve skew for entity keys, category vocabularies, and velocity windows.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Canonical feature ordering: 437 features
# Index 0: TransactionAmt (log1p transformed & scaled)
# Index 1-20: Categorical mappings
# Index 21-436: V-features, C-features, D-features, ID-features
NUMERICAL_COLS = (
    ["TransactionAmt"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 340)]
    + [f"id_0{i}" if i < 10 else f"id_{i}" for i in range(1, 12)]
)

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
] + [f"id_{i}" for i in range(12, 39)]

TOTAL_FEATURE_DIM = 437


def get_card_key(row: Dict[str, Any] | pd.Series) -> Optional[str]:
    """Generate canonical card entity key.

    Combines card1 through card6 attributes. If card1 is missing, returns None.
    """
    card1 = row.get("card1")
    if pd.isna(card1) or card1 is None or card1 == "":
        return None
    card2 = row.get("card2", "nan")
    card3 = row.get("card3", "nan")
    card4 = row.get("card4", "nan")
    card5 = row.get("card5", "nan")
    card6 = row.get("card6", "nan")
    return f"card_{card1}_{card2}_{card3}_{card4}_{card5}_{card6}"


def get_merchant_key(row: Dict[str, Any] | pd.Series) -> Optional[str]:
    """Generate canonical merchant entity key.

    Rule: Pair of (ProductCD, addr1). Rows with no addr1 get no merchant edge.
    DO NOT aggregate missing addresses into a giant fake merchant node.
    """
    product_cd = row.get("ProductCD")
    addr1 = row.get("addr1")
    if pd.isna(addr1) or addr1 is None or pd.isna(product_cd) or product_cd is None:
        return None
    try:
        addr1_int = int(float(addr1))
    except (ValueError, TypeError):
        return None
    return f"merch_{product_cd}_{addr1_int}"


def get_device_key(row: Dict[str, Any] | pd.Series) -> Optional[str]:
    """Generate canonical device entity key from identity attributes.

    Rule: Combines DeviceInfo and DeviceType, or id_30/id_31 OS/browser if present.
    """
    device_info = row.get("DeviceInfo")
    device_type = row.get("DeviceType")

    if pd.isna(device_info) or device_info is None:
        device_info = ""
    if pd.isna(device_type) or device_type is None:
        device_type = ""

    device_info = str(device_info).strip()
    device_type = str(device_type).strip()

    if not device_info and not device_type:
        return None

    return f"dev_{device_type}_{device_info}"


def compute_time_split_indices(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Strict time-based split.

    Sorts by time_col.
    First 70% is Train, next 10% is Validation, last 20% is Test.
    Never randomizes.
    """
    sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(sorted_df)

    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    train_indices = sorted_df.index[:n_train].to_numpy()
    val_indices = sorted_df.index[n_train:n_val].to_numpy()
    test_indices = sorted_df.index[n_val:].to_numpy()

    boundaries = {
        "t_train_start": int(sorted_df.iloc[0][time_col]),
        "t_val_start": int(sorted_df.iloc[n_train][time_col]),
        "t_test_start": int(sorted_df.iloc[n_val][time_col]),
        "t_max": int(sorted_df.iloc[-1][time_col]),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
    }

    return train_indices, val_indices, test_indices, boundaries
