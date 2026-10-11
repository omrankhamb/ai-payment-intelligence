# FraudGraph: Real-Time Payment Fraud Detection with Hybrid GNN

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

FraudGraph is an enterprise-grade, real-time payment fraud detection system combining **Graph Neural Networks (Hybrid Gated GNN with SAGE & GATv2)** and gradient-boosted baselines (XGBoost) with a high-throughput microservices serving architecture.

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
                              │                       (1h/6h/24h card burst)
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
                     (SAGE + GATv2)
                              │
                              ▼
                     [ Rule/Lift Explainer ]
                     (Plain-text reasons)
                              │
                              ▼
             [ Response / Prometheus Metrics / Kafka ]
               (fraud-scores topic & /metrics endpoint)
```

---

## 🎯 Target Benchmarks

| Metric | Target | Rationale |
|---|---|---|
| **Precision @ 10% Recall** | $\ge 0.955$ | Operational fraud team efficiency (high confidence on flags) |
| **ROC-AUC (Test)** | $\ge 0.854$ (GNN) / $\ge 0.907$ (XGBoost) | Time-split validation on unseen transactions |
| **Inference Latency (p99)** | $< 100\text{ ms}$ (Target: $\sim 15.6\text{ ms}$) | Synchronous gateway checkout threshold |
| **Velocity Lift** | $\ge 1.6\times$ base rate | Verifies card-testing burst graph signal |

---

## 📋 Implementation Roadmap

- [x] **Phase 0: Scope, contracts & repo setup**
- [ ] **Phase 1: Data, split, XGBoost baselines** (`notebooks/01_eda_baseline.ipynb`)
- [ ] **Phase 2: Build the graph** (`notebooks/02_graph_construction.ipynb`, `src/features.py`)
- [ ] **Phase 3: Hybrid GNN & ablation** (`notebooks/03_graphsage_train.ipynb`, `src/model.py`)
- [ ] **Phase 4: Explainer** (`notebooks/04_explainability.ipynb`, `src/explainer.py`)
- [ ] **Phase 5: Export sample data for serving** (`notebooks/05_serving_sample.ipynb`)
- [ ] **Phase 6: Serving: API, Redis, scorer** (`src/serving/api.py`, `inference.py`, `docker-compose.yml`)
- [ ] **Phase 7: Streaming with Kafka** (`src/serving/producer.py`, `consumer.py`)
- [ ] **Phase 8: Monitoring & Drift detection** (`monitoring/prometheus.yml`, Grafana dashboards)
- [ ] **Phase 9: Testing & Benchmarking** (`tests/`, `src/loadtest.py`)
- [ ] **Phase 10: Documentation & Demo**
- [ ] **Phase 11: Production Extensions**

---

## 🚦 Decision Thresholds

| Fraud Score | Decision | Action |
|---|---|---|
| $< 0.30$ | `APPROVE` | Frictionless processing |
| $0.30 - 0.70$ | `FLAG` | Challenge with 3DS or send to manual review queue |
| $> 0.70$ | `DECLINE` | Immediate decline to mitigate loss |

---

## 🛠️ Quick Start (Local Serving)

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-serve.txt
```

### 2. Infrastructure
```bash
docker compose up -d
```

### 3. Run FastAPI Service
```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```
