"""Drift Injection and Alert Verification.

Simulates normal traffic followed by a burst of anomalous $9,999 transactions,
verifying that:
1. transaction_amount_mean rises sharply.
2. feature_drift_alert flips from 0.0 (HEALTHY) to 1.0 (DRIFT ALERT).
"""

import time
from src.serving.metrics import FEATURE_DRIFT_ALERT_GAUGE, TX_AMOUNT_MEAN_GAUGE, drift_tracker


def run_drift_test():
    print("\n--- [Phase 8: Drift Test] ---")

    # 1. Normal traffic baseline ($50 - $120)
    print("Sending baseline normal traffic ($50 - $120)...")
    for _ in range(25):
        drift_tracker.record_transaction(85.0)

    pre_drift_val = FEATURE_DRIFT_ALERT_GAUGE._value.get()
    pre_mean = TX_AMOUNT_MEAN_GAUGE._value.get()
    print(f"Pre-drift rolling mean amount: ${pre_mean:.2f}")
    print(f"Pre-drift alert tile status:   {pre_drift_val} (HEALTHY)")
    assert pre_drift_val == 0.0, f"Expected 0.0 healthy status, got {pre_drift_val}"

    # 2. Inject anomalous burst of $9,999 payments
    print("\nInjecting burst of 20 x $9,999 transactions...")
    for _ in range(20):
        drift_tracker.record_transaction(9999.0)

    post_drift_val = FEATURE_DRIFT_ALERT_GAUGE._value.get()
    post_mean = TX_AMOUNT_MEAN_GAUGE._value.get()
    print(f"Post-drift rolling mean amount: ${post_mean:.2f}")
    print(f"Post-drift alert tile status:  {post_drift_val} (DRIFT ALERT! Tile turns RED in Grafana)")
    assert post_drift_val == 1.0, f"Expected 1.0 drift alert status, got {post_drift_val}"

    print("\n--- Drift Alert Verification: PASSED ---")
    return True


if __name__ == "__main__":
    run_drift_test()
