# FraudGraph Technical Review & Governance

## 1. Provenance & Engineering Audit

| Component | AI-Assisted Drafted | Hand-Reviewed / Hardened | Key Design Rationale |
|---|---|---|---|
| **Data Contracts** (`docs/contracts.md`) | Initial schema formatting | Hardened constraints & thresholds | Standardized 437-element fixed vector to eliminate train/serve skew |
| **Leakage Controls** (`src/features.py`, `src/graph.py`) | Function definitions | Strictly audited chronological split logic | Scalers and category frequencies fitted strictly on $T < T_{\text{val}}$ |
| **GNN Architecture** (`src/model.py`) | PyG module scaffold | Custom `HybridGatedConv` fusion logic | Learnable temperature gating balances attention on bursts vs mean on identity |
| **Redis Store** (`src/serving/redis_store.py`) | Zset query template | Added in-memory fallback & pipeline atomic commits | Fallback ensures API does not crash when Redis cluster is restarting |
| **Operational Explainer** (`src/explainer.py`) | Pattern rules | Added statistical lift thresholding & protective reasons | Prevents false alerts on high-volume legitimate merchants |

---

## 2. Known Limitations & Trade-Offs

1. **Cold-Start Entity Embeddings:**
   - Novel credit cards or devices unseen during training receive the population mean embedding vector $\mathbf{\bar{e}}$.
   - *Mitigation:* The network still leverages the 437 tabular features and amount anomalies, but graph relational lift is reduced until history accumulates.

2. **6-Field vs 430-Field Feature Representation Gap:**
   - When requests contain only core payment attributes (Amount, ProductCD, card1, addr1), unprovided features are imputed with training medians.
   - *Mitigation:* Phase 11 recommends integrating an online feature store (e.g. Feast or Redis Feature Store) to bridge this gap at request time.

3. **Label Latency:**
   - Real-world fraud chargebacks take 30-90 days to settle. Explainer priors and retrained graphs must account for this lag.

---

## 3. Failure Drill Response Matrix

| Failure Mode | Handled By | Outcome |
|---|---|---|
| **Unknown Card / Merchant** | `FraudScorer.get_entity_embedding()` | Gracefully applies cold-start mean embedding $\mathbf{\bar{e}}$. |
| **Redis Cache Down** | `RedisVelocityStore` fallback memory cache | Logs warning, continues scoring using in-memory window accumulator. |
| **Kafka Cluster Outage** | Decoupled synchronous API architecture | `POST /score` continues serving traffic unaffected. |
| **Corrupted / Invalid Input** | Pydantic Schema Validation | Returns HTTP 422 Unprocessable Content immediately. |
| **Feature Drift (e.g., $9,999 burst)** | `DriftTracker` & Prometheus Gauge | Flips `feature_drift_alert` to 1.0, alerting on-call risk engineering. |
