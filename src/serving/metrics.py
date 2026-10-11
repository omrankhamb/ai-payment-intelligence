"""Prometheus Metrics & Drift Detection Tracker for Serving.

Tracks:
1. fraud_score: Histogram of fraud probability distribution.
2. decision_total: Counter of decisions (APPROVE, FLAG, DECLINE).
3. inference_latency_seconds: Histogram of end-to-end request latency.
4. transaction_amount_mean: 5-minute rolling mean of payment amounts.
5. feature_drift_alert: Gauge (0 or 1), fires alert if rolling amount mean
   deviates by > 2.0 standard deviations from the training baseline mean.
"""

from collections import deque
import time
from typing import Deque, Tuple
from prometheus_client import Counter, Gauge, Histogram

# 1. Fraud score distribution
FRAUD_SCORE_HIST = Histogram(
    "fraud_score",
    "Distribution of predicted payment fraud risk scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# 2. Decision counter
DECISION_TOTAL = Counter(
    "decision_total",
    "Total transaction decisions issued",
    ["decision"],
)

# 3. Latency histogram
LATENCY_HIST = Histogram(
    "inference_latency_seconds",
    "End-to-end inference and decision latency in seconds",
    buckets=[0.002, 0.005, 0.010, 0.015, 0.025, 0.050, 0.100, 0.250],
)

# 4. Rolling mean amount gauge
TX_AMOUNT_MEAN_GAUGE = Gauge(
    "transaction_amount_mean",
    "5-minute rolling mean of transaction payment amounts in USD",
)

# 5. Drift alert gauge (0 = Healthy, 1 = Drift Alert)
FEATURE_DRIFT_ALERT_GAUGE = Gauge(
    "feature_drift_alert",
    "Active feature drift indicator (1 if rolling mean moves > 2 std from training baseline)",
)


class DriftTracker:
    """Maintains 5-minute rolling window of transaction amounts and monitors for drift."""

    def __init__(
        self,
        train_mean: float = 135.0,
        train_std: float = 239.0,
        window_seconds: int = 300,
    ):
        self.train_mean = train_mean
        self.train_std = train_std
        self.window_seconds = window_seconds
        # Stores (timestamp, amount)
        self.window: Deque[Tuple[float, float]] = deque()

    def record_transaction(self, amount: float, current_time: float = None):
        if current_time is None:
            current_time = time.time()

        self.window.append((current_time, float(amount)))

        # Evict events outside 5-minute sliding window
        cutoff = current_time - self.window_seconds
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

        # Compute rolling mean
        if self.window:
            rolling_mean = sum(amt for _, amt in self.window) / len(self.window)
            TX_AMOUNT_MEAN_GAUGE.set(rolling_mean)

            # Check for drift (> 2 training standard deviations)
            drift_threshold = self.train_mean + (2.0 * self.train_std)
            if rolling_mean > drift_threshold:
                FEATURE_DRIFT_ALERT_GAUGE.set(1.0)
            else:
                FEATURE_DRIFT_ALERT_GAUGE.set(0.0)


# Global drift tracker instance
drift_tracker = DriftTracker()
