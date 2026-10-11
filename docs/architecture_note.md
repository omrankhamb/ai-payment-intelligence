# FraudGraph Architecture Note

## 1. System Topology & Request Lifecycle

```
[ Checkout / Payment Gateway ]
             │
             ▼
    [ FastAPI Endpoint ] ── (POST /score)
             │
      ┌──────┴───────────────────────────────────┐
      ▼                                          ▼
[ Feature Builder ]                    [ Redis Velocity Store ]
Imputes medians &                      Sliding windows (5m, 1h, 6h, 24h)
scales 437-dim vector                  Atomic increment-then-read
      │                                          │
      └──────────────────┬───────────────────────┘
                         ▼
             [ FraudScorer Core ]
             1-Hop Local Subgraph
             ├── Card Embedding (Cold-start mean fallback)
             ├── Merchant Embedding
             └── Device Embedding
                         │
                         ▼
             [ Hybrid Gated GNN ]
             Dynamic gating: GATv2 (Attention) + SAGE (Mean)
                         │
                         ▼
             [ Fraud Score: p in [0, 1] ]
             < 0.30: APPROVE | 0.30-0.70: FLAG | > 0.70: DECLINE
                         │
                         ▼
             [ Operational Explainer ]
             Faithful risk lift & protective explanations (< 1ms)
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[ Response JSON ]             [ Prometheus Scraper ]
(Contract 3 payload)          (Latency, Drift Alert, Score Hist)
```

---

## 2. Component Rationale & Resource Footprint

| Component | Why it Exists | Operational Cost / Trade-off |
|---|---|---|
| **FastAPI** | High-concurrency async HTTP serving with Pydantic validation. | Ultra-low overhead (< 2ms framework latency). |
| **Redis Store** | Maintains rolling velocity bursts without scanning historical database. | In-memory key expiration (48h TTL) limits memory footprint to < 500 MB for active cards. |
| **Hybrid Gated GNN** | Captures both dense identity relations (mean aggregation) and acute velocity bursts (attention). | ~1.2M parameters. Forward pass on CPU executes in ~2-5 ms. |
| **Operational Explainer** | Fraud analysts cannot act on black-box probabilities without actionable dispute grounds. | Sub-millisecond hash map lookups fitted on training partition. |
| **Kafka (KRaft)** | Decouples synchronous transaction ingress from asynchronous auditing and reporting. | Zero Zookeeper dependencies; partitioned by card ID to guarantee ordering. |
| **Prometheus & Grafana** | Instant visibility into latency degradation and upstream feature drift. | Lightweight metrics scraping at 5-second intervals. |

---

## 3. Strict Anti-Leakage Protocol

1. **Chronological Splitting:** The IEEE-CIS dataset was divided chronologically:
   - Train: First 70% of time interval
   - Validation: Next 10%
   - Test: Final 20%
2. **Train-Only Subgraph:** Message passing during model training is strictly isolated from nodes belonging to validation or test periods.
3. **Explainer Priors:** Entity dispute counts and fraud lift tables are computed solely from confirmed train-period chargebacks.
