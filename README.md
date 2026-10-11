# FraudGraph: Real-Time Payment Fraud Prevention with Hybrid GNN

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org)
[![Prometheus](https://img.shields.io/badge/Metrics-Prometheus-red.svg)](https://prometheus.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

FraudGraph is an enterprise-grade, real-time payment fraud intelligence system combining **Graph Neural Networks (Hybrid Gated GNN: GraphSAGE + GATv2)** and high-performance gradient-boosted baselines (XGBoost) with a sub-5ms microservices serving architecture.

---

## 🚀 High-Level Architecture

```
                                  [ Payment Stream ]
                                          │
                                          ▼
                                   [ Kafka Topic ]
                                   (transactions)
                                          │
                                          ▼
  [ Client / API ] ──► [ FastAPI: POST /score ] ◄──► [ Redis Velocity Cache ]
                              │                       (5m/1h/6h/24h card burst)
                              ▼
                     [ FraudScorer Engine ]
                 ┌────────────┴────────────┐
                 ▼                         ▼
         [ Feature Builder ]       [ 1-Hop Subgraph ]
         (437-dim vector)          (Card / Merchant / Device)
                 │                         │
                 └────────────┬────────────┘
                              ▼
                   [ Hybrid Gated GNN ]
                   Dynamic Gating: GATv2 + SAGE
                              │
                              ▼
                     [ Operational Explainer ]
                     (Plain-text reasons in < 1ms)
                              │
                              ▼
             [ Response / Prometheus Metrics / Kafka ]
               (fraud-scores topic & /metrics endpoint)
```

---

## 📊 Benchmark Results

Evaluated strictly on the **chronological test partition** (IEEE-CIS Fraud Detection, final 20% by `TransactionDT`):

| Model / Architecture | Features | ROC-AUC | PR-AUC (Avg Precision) | Precision @ 10% Recall |
|---|---|---|---|---|
| **Baseline A (XGBoost Weak)** | 8 Engineered | 0.769 | 0.324 | 0.782 |
| **Baseline B (XGBoost Full)** | All ~430 | **0.907** | **0.612** | **0.968** |
| **PureSAGE (Control GNN)** | GraphSAGE | 0.812 | 0.448 | 0.892 |
| **HybridGatedGNN (SAGE + GATv2)** | 437 + Hetero Graph | **0.854** | **0.528** | **0.955** |

### ⚡ Dynamic Gate Behavior (Ablation Finding):
- **Velocity Edges (`velocity_cluster`):** Gate average **0.80** $\to$ Strong reliance on multi-head attention (GATv2) to detect burst clusters.
- **Structural Identity Edges (`uses_card`):** Gate average **0.37** $\to$ Preference for mean aggregation (GraphSAGE) over dense entity history.

---

## ⚡ Measured Inference Latency (Benchmark Run)

Load test run on local host (`Intel64 AMD64 CPU`, $N=100$ requests):

| Metric | Measured | Target SLA | Status |
|---|---|---|---|
| **Throughput** | **1,728 req/s** | $> 100\text{ req/s}$ | ✅ PASS |
| **Mean Latency** | **0.45 ms** | $< 25.0\text{ ms}$ | ✅ PASS |
| **p50 Latency** | **0.22 ms** | $< 10.0\text{ ms}$ | ✅ PASS |
| **p95 Latency** | **0.72 ms** | $< 50.0\text{ ms}$ | ✅ PASS |
| **p99 Latency** | **4.97 ms** | **$< 100.0\text{ ms}$** | ✅ PASS |

---

## 📋 Phase Roadmap Completion

- [x] **Phase 0: Scope, contracts & repo setup** (`docs/contracts.md`, repo initialization)
- [x] **Phase 1: Data, split & baselines** (`notebooks/01_eda_baseline.ipynb`, `src/baseline.py`)
- [x] **Phase 2: Build the graph** (`notebooks/02_graph_construction.ipynb`, `src/graph.py`, `src/features.py`)
- [x] **Phase 3: Hybrid GNN & ablation** (`notebooks/03_graphsage_train.ipynb`, `src/model.py`)
- [x] **Phase 4: Operational explainer** (`notebooks/04_explainability.ipynb`, `src/explainer.py`)
- [x] **Phase 5: Export sample data for serving** (`notebooks/05_serving_sample.ipynb`, `src/sample_exporter.py`)
- [x] **Phase 6: Serving: API, Redis, scorer** (`src/serving/api.py`, `inference.py`, `redis_store.py`)
- [x] **Phase 7: Streaming with Kafka** (`src/serving/producer.py`, `src/serving/consumer.py`)
- [x] **Phase 8: Monitoring & Drift detection** (`monitoring/prometheus.yml`, Grafana dashboards, `src/drift_simulation.py`)
- [x] **Phase 9: Testing & Benchmarking** (`tests/run_tests.py`, `src/loadtest.py`)
- [x] **Phase 10: Documentation & Demo** (`docs/architecture_note.md`, `docs/demo_guide.md`, `docs/technical_review.md`)
- [ ] **Phase 11: Production Extensions** (UPI payments, managed online feature store)

---

## 🚦 Decision Thresholds

| Fraud Score | Decision | Action |
|---|---|---|
| $< 0.30$ | `APPROVE` | Frictionless checkout |
| $0.30 - 0.70$ | `FLAG` | Challenge with 3DS step-up or manual review |
| $> 0.70$ | `DECLINE` | Automated block to prevent chargeback loss |

---

## 🛠️ Quick Start

### 1. Environment & Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-serve.txt
```

### 2. Run Test Suite & Leakage Audit
```bash
python tests/run_tests.py
```

### 3. Run Load Benchmark
```bash
python -m src.loadtest --n 100
```

### 4. Start Serving API
```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### 5. Stream Replay Simulation
```bash
python -m src.serving.producer --n 100 --rate 25
python -m src.serving.consumer --max_messages 100
```

### 6. Drift Alert Simulation
```bash
python -m src.drift_simulation
```

---

## 🛡️ Failure Scenario Protections

- **Unknown Entity / Cold Start:** Fallback to population mean embeddings $\mathbf{\bar{e}}$.
- **Redis Cache Outage:** Seamless fallback to internal memory window accumulator.
- **Kafka Broker Down:** Synchronous `POST /score` remains fully available.
- **Upstream Feature Drift:** Real-time alert fires when rolling mean amount deviates $> 2\sigma$.
