"""Phase 4: Operational Fraud Explainer.

Provides plain-text, interpretable risk reasons and weights for fraud decisions.
Rules:
1. Lift-based evidence: High volume without statistical lift is not flagged as risky.
2. Label timing integrity: Uses only confirmed labels from the train-period ledger.
3. Fast execution (< 1 ms latency budget).
4. Protective reasons worded constructively for legitimate history.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.features import get_card_key, get_device_key, get_merchant_key


class FraudExplainer:
    """Interpretable graph and behavioral fraud explainer."""

    def __init__(self, stats: Dict[str, Any]):
        self.base_fraud_rate = stats.get("base_fraud_rate", 0.035)
        self.card_stats = stats.get("card_stats", {})
        self.merchant_stats = stats.get("merchant_stats", {})
        self.device_stats = stats.get("device_stats", {})
        self.amount_stats = stats.get("amount_stats", {"median": 68.5, "p95": 300.0, "p99": 750.0})

    @classmethod
    def load(cls, stats_path: str = "artifacts/explainer_stats.json") -> "FraudExplainer":
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats)

    def explain(
        self,
        payload: Dict[str, Any],
        velocity_info: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate ranked, weighted explanation reasons in < 1 ms.

        Returns list of {factor, weight, edge_type, description}.
        """
        reasons: List[Dict[str, Any]] = []

        # 1. Velocity Cluster Burst Reason
        if velocity_info:
            burst_5m = velocity_info.get("tx_count_5m", 1)
            confirmed_fraud_prior = velocity_info.get("prior_fraud_count", 0)
            if burst_5m >= 3:
                weight = min(0.60, 0.20 + 0.10 * burst_5m)
                desc = f"{burst_5m} rapid transactions on this card within 5 minutes"
                if confirmed_fraud_prior > 0:
                    weight += 0.20
                    desc += f" with {confirmed_fraud_prior} prior confirmed fraud charges"
                reasons.append({
                    "factor": "velocity_burst",
                    "weight": round(weight, 3),
                    "edge_type": "velocity_cluster",
                    "description": desc,
                })

        # 2. Card History & Lift Reason
        card_key = get_card_key(payload)
        card_rec = self.card_stats.get(card_key) if card_key else None
        if card_rec:
            c_tx_count = card_rec.get("count", 0)
            c_fraud_count = card_rec.get("fraud_count", 0)
            c_rate = card_rec.get("fraud_rate", 0.0)

            if c_fraud_count >= 2 and c_rate > self.base_fraud_rate * 2.0:
                lift = c_rate / self.base_fraud_rate
                reasons.append({
                    "factor": "high_risk_card_history",
                    "weight": min(0.55, 0.25 + 0.05 * lift),
                    "edge_type": "uses_card",
                    "description": f"Card has {c_fraud_count} confirmed prior fraud disputes ({lift:.1f}x baseline risk lift)",
                })
            elif c_tx_count >= 15 and c_fraud_count == 0:
                # Protective reason
                reasons.append({
                    "factor": "trusted_card_history",
                    "weight": -0.35,
                    "edge_type": "uses_card",
                    "description": f"Card has established history of {c_tx_count} verified legitimate transactions with 0 disputes",
                })

        # 3. Device Risk & Shared Card Reason
        dev_key = get_device_key(payload)
        dev_rec = self.device_stats.get(dev_key) if dev_key else None
        if dev_rec:
            d_cards = dev_rec.get("unique_cards", 1)
            d_fraud_count = dev_rec.get("fraud_count", 0)
            d_rate = dev_rec.get("fraud_rate", 0.0)

            if d_cards >= 3 and d_fraud_count >= 1:
                reasons.append({
                    "factor": "suspicious_shared_device",
                    "weight": 0.40,
                    "edge_type": "on_device",
                    "description": f"Device shared across {d_cards} distinct cards with prior fraud history",
                })
            elif d_rate > self.base_fraud_rate * 2.5 and dev_rec.get("count", 0) >= 5:
                lift = d_rate / self.base_fraud_rate
                reasons.append({
                    "factor": "high_risk_device",
                    "weight": 0.35,
                    "edge_type": "on_device",
                    "description": f"Device associated with high fraud incidence ({lift:.1f}x baseline rate)",
                })

        # 4. Merchant Risk Reason
        merch_key = get_merchant_key(payload)
        merch_rec = self.merchant_stats.get(merch_key) if merch_key else None
        if merch_rec:
            m_count = merch_rec.get("count", 0)
            m_rate = merch_rec.get("fraud_rate", 0.0)
            if m_count >= 10 and m_rate > self.base_fraud_rate * 3.0:
                lift = m_rate / self.base_fraud_rate
                reasons.append({
                    "factor": "high_risk_merchant_cluster",
                    "weight": min(0.40, 0.15 + 0.05 * lift),
                    "edge_type": "at_merchant",
                    "description": f"Merchant location exhibits elevated chargeback rate ({lift:.1f}x baseline risk)",
                })

        # 5. Amount Anomalies
        amt = payload.get("TransactionAmt", 0.0)
        try:
            amt = float(amt)
        except (ValueError, TypeError):
            amt = 0.0

        card_med_amt = card_rec.get("median_amt") if card_rec else None
        if card_med_amt and card_med_amt > 0 and amt > card_med_amt * 4.0:
            reasons.append({
                "factor": "abnormal_amount_for_card",
                "weight": 0.30,
                "edge_type": "transaction_feature",
                "description": f"Payment amount (${amt:.2f}) is over 4x card historical median (${card_med_amt:.2f})",
            })
        elif amt > self.amount_stats["p99"]:
            reasons.append({
                "factor": "outlier_transaction_amount",
                "weight": 0.20,
                "edge_type": "transaction_feature",
                "description": f"Payment amount (${amt:.2f}) exceeds the 99th percentile of platform traffic",
            })

        if not reasons:
            reasons.append({
                "factor": "baseline_relational_eval",
                "weight": 0.05,
                "edge_type": "transaction_feature",
                "description": "No localized relational risk spikes detected; evaluation driven by multi-hop graph embeddings",
            })

        # Sort by magnitude of risk weight (highest absolute weight first)
        reasons.sort(key=lambda r: abs(r["weight"]), reverse=True)
        return reasons[:top_k]


def build_explainer_stats(
    df_train: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Fit explainer statistics strictly on the training partition.

    Never compute on validation or test sets to prevent label timing leakage.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    base_rate = float(df_train["isFraud"].mean())

    # Card stats
    card_records: Dict[str, Dict[str, Any]] = {}
    for _, row in df_train.iterrows():
        ckey = get_card_key(row)
        if not ckey:
            continue
        entry = card_records.setdefault(ckey, {"count": 0, "fraud_count": 0, "amounts": []})
        entry["count"] += 1
        entry["fraud_count"] += int(row["isFraud"])
        amt = row.get("TransactionAmt")
        if pd.notna(amt):
            entry["amounts"].append(float(amt))

    card_stats = {}
    for ckey, d in card_records.items():
        card_stats[ckey] = {
            "count": d["count"],
            "fraud_count": d["fraud_count"],
            "fraud_rate": round(d["fraud_count"] / d["count"], 4),
            "median_amt": round(float(np.median(d["amounts"])), 2) if d["amounts"] else 0.0,
        }

    # Merchant stats
    merch_records: Dict[str, Dict[str, Any]] = {}
    for _, row in df_train.iterrows():
        mkey = get_merchant_key(row)
        if not mkey:
            continue
        entry = merch_records.setdefault(mkey, {"count": 0, "fraud_count": 0})
        entry["count"] += 1
        entry["fraud_count"] += int(row["isFraud"])

    merch_stats = {
        mkey: {
            "count": d["count"],
            "fraud_count": d["fraud_count"],
            "fraud_rate": round(d["fraud_count"] / d["count"], 4),
        }
        for mkey, d in merch_records.items()
    }

    # Device stats
    dev_records: Dict[str, Dict[str, Any]] = {}
    for _, row in df_train.iterrows():
        dkey = get_device_key(row)
        if not dkey:
            continue
        entry = dev_records.setdefault(dkey, {"count": 0, "fraud_count": 0, "cards": set()})
        entry["count"] += 1
        entry["fraud_count"] += int(row["isFraud"])
        ckey = get_card_key(row)
        if ckey:
            entry["cards"].add(ckey)

    dev_stats = {
        dkey: {
            "count": d["count"],
            "fraud_count": d["fraud_count"],
            "fraud_rate": round(d["fraud_count"] / d["count"], 4),
            "unique_cards": len(d["cards"]),
        }
        for dkey, d in dev_records.items()
    }

    amt_series = df_train["TransactionAmt"].dropna()
    amount_stats = {
        "median": float(amt_series.median()),
        "p95": float(amt_series.quantile(0.95)),
        "p99": float(amt_series.quantile(0.99)),
    }

    stats = {
        "base_fraud_rate": round(base_rate, 4),
        "card_stats": card_stats,
        "merchant_stats": merch_stats,
        "device_stats": dev_stats,
        "amount_stats": amount_stats,
    }

    out_file = artifacts_path / "explainer_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Fitted explainer stats on {len(df_train):,} train records. Saved to {out_file}")

    return stats
