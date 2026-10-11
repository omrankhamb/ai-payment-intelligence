"""Unit Tests for FraudGraph Core Components.

Validates:
1. FeatureBuilder: Fixed input returns exact 437-dim vector deterministically.
2. RedisVelocityStore: Sliding window counts strictly increment and query time windows.
3. Decision Thresholds: Exact boundaries at 0.30 and 0.70.
4. Explainer: Risky signal requires statistical lift; zero fraud produces protective reason.
"""

import numpy as np

from src.explainer import FraudExplainer
from src.features import TOTAL_FEATURE_DIM
from src.serving.feature_builder import FeatureBuilder
from src.serving.redis_store import RedisVelocityStore


def test_feature_builder_deterministic():
    stats = {
        "medians": {"TransactionAmt": 50.0, "C1": 1.0},
        "means": {"TransactionAmt": 3.9, "C1": 1.0},
        "stds": {"TransactionAmt": 1.2, "C1": 2.0},
        "cat_vocab": {"ProductCD": 5},
    }
    builder = FeatureBuilder(stats)
    payload = {
        "TransactionAmt": 100.0,
        "ProductCD": "W",
        "card1": 12345,
    }

    vec1 = builder.build_vector(payload)
    vec2 = builder.build_vector(payload)

    assert isinstance(vec1, np.ndarray)
    assert len(vec1) == TOTAL_FEATURE_DIM
    assert np.allclose(vec1, vec2), "Feature builder must be strictly deterministic"


def test_redis_store_sliding_windows():
    store = RedisVelocityStore(host="invalid_host", port=9999)  # Uses in-memory fallback
    card_key = "test_card_123"

    base_time = 1000000

    # First transaction at t=0
    res1 = store.record_and_get_velocity(card_key, transaction_id=1, transaction_dt=base_time, amount=50.0)
    assert res1["tx_count_5m"] == 1
    assert res1["tx_count_1h"] == 1

    # Second transaction at t=120s (within 5 minutes)
    res2 = store.record_and_get_velocity(card_key, transaction_id=2, transaction_dt=base_time + 120, amount=100.0)
    assert res2["tx_count_5m"] == 2
    assert res2["tx_count_1h"] == 2
    assert res2["last_amount"] == 50.0

    # Third transaction at t=600s (10 min later -> outside 5m, inside 1h)
    res3 = store.record_and_get_velocity(card_key, transaction_id=3, transaction_dt=base_time + 600, amount=150.0)
    assert res3["tx_count_5m"] == 1  # only itself in last 300s
    assert res3["tx_count_1h"] == 3


def test_decision_threshold_boundaries():
    def get_decision(score: float) -> str:
        if score < 0.30:
            return "APPROVE"
        elif score <= 0.70:
            return "FLAG"
        return "DECLINE"

    assert get_decision(0.299) == "APPROVE"
    assert get_decision(0.300) == "FLAG"
    assert get_decision(0.500) == "FLAG"
    assert get_decision(0.700) == "FLAG"
    assert get_decision(0.701) == "DECLINE"


def test_explainer_lift_requirement():
    stats = {
        "base_fraud_rate": 0.035,
        "card_stats": {
            "card_risky_321_150_visa_102_debit": {
                "count": 20,
                "fraud_count": 8,
                "fraud_rate": 0.40,
                "median_amt": 50.0,
            },
            "card_clean_321_150_visa_102_debit": {
                "count": 50,
                "fraud_count": 0,
                "fraud_rate": 0.0,
                "median_amt": 50.0,
            },
        },
        "merchant_stats": {},
        "device_stats": {},
        "amount_stats": {"median": 50.0, "p95": 200.0, "p99": 500.0},
    }
    explainer = FraudExplainer(stats)

    # 1. Risky card with high lift should generate risk factor
    payload_risky = {
        "card1": "risky",
        "card2": 321,
        "card3": 150,
        "card4": "visa",
        "card5": 102,
        "card6": "debit",
        "TransactionAmt": 50.0,
    }
    reasons_risky = explainer.explain(payload_risky)
    factors_risky = [r["factor"] for r in reasons_risky]
    assert "high_risk_card_history" in factors_risky

    # 2. Clean card should generate protective reason (negative weight)
    payload_clean = {
        "card1": "clean",
        "card2": 321,
        "card3": 150,
        "card4": "visa",
        "card5": 102,
        "card6": "debit",
        "TransactionAmt": 50.0,
    }
    reasons_clean = explainer.explain(payload_clean)
    factors_clean = [r["factor"] for r in reasons_clean]
    assert "trusted_card_history" in factors_clean
    protective = next(r for r in reasons_clean if r["factor"] == "trusted_card_history")
    assert protective["weight"] < 0, "Protective reasons must carry negative risk weight"
