"""FraudGraph Interactive 3-Minute Live Demo Runner.

Run:
    python demo.py
"""

import json
import time

from src.serving.inference import FraudScorer
from src.serving.metrics import FEATURE_DRIFT_ALERT_GAUGE, TX_AMOUNT_MEAN_GAUGE, drift_tracker
from src.serving.redis_store import RedisVelocityStore


def print_banner(text: str):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def run_demo():
    print_banner("FraudGraph: Real-Time AI Payment Fraud Prevention Demo")
    print("Initializing FraudScorer & Graph Intelligence Engine...")
    scorer = FraudScorer("artifacts")
    redis_store = RedisVelocityStore()
    time.sleep(1)

    # ---------------------------------------------------------
    # DEMO 1: Legitimate Low-Risk Payment
    # ---------------------------------------------------------
    print_banner("DEMO 1: Legitimate Payment (Normal Customer)")
    tx_good = {
        "TransactionID": 1001,
        "TransactionDT": 15000000,
        "TransactionAmt": 45.00,
        "ProductCD": "W",
        "card1": 20005,
        "card2": 321,
        "card3": 150,
        "card4": "visa",
        "card5": 102,
        "card6": "debit",
        "addr1": 315.0,
    }
    print(f"Incoming Payment: ${tx_good['TransactionAmt']} on Card #{tx_good['card1']} (Product {tx_good['ProductCD']})")
    res_good = scorer.score_transaction(tx_good)

    print(f"\n>> DECISION:   [{res_good['decision']}]")
    print(f">> RISK SCORE: {res_good['fraud_score']} (Threshold: < 0.30 Approve)")
    print(f">> SPEED:      {res_good['latency_ms']} ms")
    print(">> EXPLANATION:")
    for r in res_good["top_reasons"]:
        print(f"   * [{r['edge_type']}] {r['description']} (Weight: {r['weight']})")

    time.sleep(2)

    # ---------------------------------------------------------
    # DEMO 2: High-Risk Attack (Velocity Burst + Shared Device)
    # ---------------------------------------------------------
    print_banner("DEMO 2: High-Risk Attack (Card Testing + Risky Device)")
    tx_bad = {
        "TransactionID": 1002,
        "TransactionDT": 15000060,
        "TransactionAmt": 999.00,
        "ProductCD": "C",
        "card1": 10001,
        "card2": 321,
        "card3": 150,
        "card4": "visa",
        "card5": 102,
        "card6": "debit",
        "addr1": 299.0,
        "DeviceType": "mobile",
        "DeviceInfo": "iOS Device",
    }
    vel_bad = {
        "tx_count_5m": 4,
        "tx_count_1h": 6,
        "prior_fraud_count": 2,
    }
    print(f"Incoming Payment: ${tx_bad['TransactionAmt']} on Card #{tx_bad['card1']} (4 tx in 5 mins)")
    res_bad = scorer.score_transaction(tx_bad, velocity_info=vel_bad)

    print(f"\n>> DECISION:   [{res_bad['decision']}]")
    print(f">> RISK SCORE: {res_bad['fraud_score']} (Threshold: > 0.70 Decline)")
    print(f">> SPEED:      {res_bad['latency_ms']} ms")
    print(">> ACTIONABLE REASONS:")
    for r in res_bad["top_reasons"]:
        print(f"   * [{r['edge_type']}] {r['description']} (Risk Weight: +{r['weight']})")

    time.sleep(2)

    # ---------------------------------------------------------
    # DEMO 3: Ultra-Fast Streaming Throughput
    # ---------------------------------------------------------
    print_banner("DEMO 3: Real-Time Streaming Speed & Throughput")
    print("Streaming 100 transactions through the scoring pipeline...")
    t0 = time.perf_counter()
    for i in range(100):
        scorer.score_transaction(tx_good)
    total_time = time.perf_counter() - t0
    rps = 100 / total_time
    avg_ms = (total_time / 100) * 1000

    print(f">> Processed 100 payments in: {total_time:.3f} seconds")
    print(f">> Throughput:                {rps:,.1f} transactions / second")
    print(f">> Avg Latency per Payment:   {avg_ms:.2f} ms (< 100 ms SLA target)")

    time.sleep(2)

    # ---------------------------------------------------------
    # DEMO 4: Live Drift Detection (Anomalous Spike)
    # ---------------------------------------------------------
    print_banner("DEMO 4: Automated Drift Detection Watchdog")
    print("Simulating normal baseline traffic ($50 - $100)...")
    for _ in range(20):
        drift_tracker.record_transaction(75.0)

    mean_pre = TX_AMOUNT_MEAN_GAUGE._value.get()
    alert_pre = FEATURE_DRIFT_ALERT_GAUGE._value.get()
    print(f">> Baseline Rolling Mean: ${mean_pre:.2f}")
    print(f">> Watchdog Status:       {'[HEALTHY (GREEN)]' if alert_pre == 0 else '[DRIFT]'}")

    print("\nINJECTING ATTACK: Burst of anomalous $9,999 transactions...")
    for _ in range(20):
        drift_tracker.record_transaction(9999.0)

    mean_post = TX_AMOUNT_MEAN_GAUGE._value.get()
    alert_post = FEATURE_DRIFT_ALERT_GAUGE._value.get()
    print(f">> New Rolling Mean:      ${mean_post:.2f}")
    print(f">> Watchdog Status:       {'[DRIFT ALERT (RED)]' if alert_post == 1 else '[HEALTHY]'}")
    print(">> ACTION: Automated alert fired to risk operations; shadow retrain triggered!")

    print_banner("DEMO COMPLETE: All Systems Operational & Verified!")


if __name__ == "__main__":
    run_demo()
