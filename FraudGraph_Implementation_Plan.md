# FraudGraph: Step-by-Step Implementation Plan

Real-time payment fraud detection with a graph neural network (GNN).
Stack: Python, PyTorch Geometric, XGBoost, MLflow, FastAPI, Kafka, Redis, Prometheus, Grafana, Docker Compose.

**How to use this plan:** do the phases in order. Every phase has steps, an output, and a "done when" check. Do not start the next phase until the check passes.

**Rule for numbers:** every score, speed or accuracy number in your report must come from a run you can repeat. The numbers in this plan (AUC 0.854, p99 15.6 ms, and so on) are what the FraudGraph README reports. Treat them as targets to compare against, not as facts to copy.

---

## Plan at a glance

| Phase | Goal | Where it runs | Rough time |
|---|---|---|---|
| 0 | Scope, contracts, repo setup | Laptop | 0.5 day |
| 1 | Data, split, XGBoost baselines | Kaggle (CPU) | 1 day |
| 2 | Build the graph | Kaggle (CPU) | 1 day |
| 3 | Train and compare the GNN | Kaggle (T4 GPU) | 2 days |
| 4 | Explainer (reasons) | Kaggle (CPU) | 1 day |
| 5 | Export sample data for serving | Kaggle (CPU) | 0.5 day |
| 6 | Serving: API, Redis, scorer | Laptop | 2 days |
| 7 | Streaming with Kafka | Laptop | 1 day |
| 8 | Monitoring: Prometheus, Grafana, drift | Laptop | 1 day |
| 9 | Testing and measuring | Laptop | 1 day |
| 10 | Docs, demo, review | Laptop | 1 day |
| 11 | Extensions (new payment method, feature store) | Optional | After v1 |

Total for version 1: about 12 working days for one person. Two people can overlap Phases 6 to 8 with Phases 3 to 5.

---

## Phase 0: Scope, contracts and repo setup

**Goal:** agree on what you are building before you write code.

1. **Write the problem in 3 lines.**
   - User: a fraud operations team.
   - Input: one payment.
   - Output: fraud score, decision (APPROVE, FLAG, DECLINE) and reasons.
2. **Set success targets.**
   - Quality: precision at 10% recall (the metric a fraud team works with) and AUC, both on a time-based test set.
   - Speed: decision in under 100 ms.
3. **Write the three contracts** in a `docs/contracts.md` file.
   - Request payload: `TransactionAmt, ProductCD, card1, addr1, TransactionDT, TransactionID` (plus optional device fields and an optional full feature vector).
   - Feature vector: 437 numbers in a fixed order.
   - Response: `fraud_score`, `decision`, `top_reasons[]` where each reason has `factor`, `weight`, `edge_type`.
4. **Set decision thresholds.** Under 0.3 approve, over 0.7 decline, between flag.
5. **Create the repo.** Folders: `notebooks/`, `src/`, `src/serving/`, `monitoring/`, `docs/`, `assets/`. Add `.gitignore` (ignore `artifacts/`, `data/`, `.venv/`), `README.md` stub, `requirements-serve.txt`.
6. **Set a branch and commit rule.** Small commits, one stage per branch or PR, review before merge.
7. **Install tools.** Docker Desktop, Python 3.11+, Git. Create a Kaggle account and open the IEEE-CIS Fraud Detection dataset.

**Done when:** contracts are written, the repo is on GitHub, and `docker --version` and `python --version` both work.

---

## Phase 1: Data, split and baselines (`01_eda_baseline.ipynb`)

**Goal:** know your data and set the bar that the GNN has to beat.

1. **Load the data** on Kaggle (transaction and identity tables). Join on `TransactionID`.
2. **Do quick EDA.** Fraud rate (about 3.5%), missing values, amount spread, how much of the data has device info (about 20%).
3. **Make the time-based split.** Sort by `TransactionDT`. First 70% is train, next 10% is validation, last 20% is test. Never use a random split. Fraud data is time-ordered and random folds leak the future.
4. **Rule:** all choices (features, settings, stopping point) use the validation slice only. The test slice is scored once per model.
5. **Train baseline A:** XGBoost with about 8 hand-built features.
6. **Train baseline B:** XGBoost with all ~430 features.
7. **Log everything in MLflow** (sqlite file). Save metrics: AUC, average precision, precision at 10% recall.
8. **Save the split boundaries** (the time cut points) in `artifacts/`. Every later notebook must reuse them.

**Output:** baseline metrics table, saved split, MLflow runs.
**Done when:** XGBoost with all features is close to the README (AUC about 0.907). Weak baseline A should be much lower (about 0.769).

---

## Phase 2: Build the graph (`02_graph_construction.ipynb`)

**Goal:** turn rows into a connected graph.

1. **Choose the node types.**
   - Transaction nodes carry the 437 features (scaled numbers plus category codes).
   - Card, merchant and device nodes carry no features. They get a learned identity vector (embedding) later.
2. **Define the keys in one file, `src/features.py`.** The same key rules must be used in training and in serving. If they differ, serving will break silently.
   - Card: from the card columns.
   - Merchant: pair of `(ProductCD, addr1)`. Rows with no address get no merchant edge. Do not put them into one fake giant merchant.
   - Device: from the identity columns.
3. **Add edges.**
   - `uses_card`: transaction to card.
   - `at_merchant`: transaction to merchant.
   - `on_device`: transaction to device.
   - `velocity_cluster`: transaction to transaction. Link two payments on the same card within 5 minutes. Point the edge from earlier to later so information only flows forward in time.
4. **Add reverse edges** for all four types, so there are 8 edge types for message passing.
5. **Check the graph.** Count nodes and edges and compare with the README (590,540 transactions, 13,553 cards, 527 merchants, 1,787 devices; edges 590,540 / 524,834 / 118,666 / 114,439).
6. **Test the velocity idea.** Fraud rate inside velocity clusters should be higher than the base rate (README: about 1.6x). If not, re-check how you built the edges.
7. **Save** `graph.pt`, the entity maps (id to index), and the scaler and category vocabulary.

**Output:** `graph.pt`, entity maps, feature stats.
**Done when:** counts match, and the velocity check shows lift.

---

## Phase 3: Hybrid GNN and ablation (`03_graphsage_train.ipynb`, `src/model.py`)

**Goal:** train the model and prove that the hybrid design helps.

1. **Write `src/model.py`** with three parts.
   - `PureSAGE`: GraphSAGE layers only. This is the control.
   - `HybridGatedConv`: runs SAGEConv (mean) and GATv2Conv (4-head attention) side by side, then blends them:
     `gate = sigmoid(W * x_dst / temperature)` and `out = gate * GAT(x) + (1 - gate) * SAGE(x)`. Make the temperature learnable.
   - `HybridGatedGNN`: transaction encoder (437 to 128), two hetero layers (128 to 128 to 64) with sum across edge types and ReLU, then a head (64 to 32 to 1). Entity nodes use `nn.Embedding(128)`.
2. **Write focal loss** (alpha 0.9, gamma 2.0) for the 3.5% fraud class. The README chose it over weighted BCE because weighted BCE made validation scores jump around.
3. **Set up training.**
   - NeighborLoader with fan-out [15, 10], batch size 512.
   - Adam at 1e-3, gradient clipping, ReduceLROnPlateau on validation average precision.
   - Early stopping after 8 flat evaluations, judged on validation only.
4. **Prevent leakage. This is the most important step.**
   - During training, only let message passing use the **train-period subgraph**. Otherwise test-period nodes can reach the model through reverse edges.
   - Pick checkpoints on validation, never on test.
   - At inference, using the full graph is fine. That matches production, where the graph holds all history up to now.
5. **Run the ablation.** Same graph, same settings, same seeds. Only difference: PureSAGE versus Hybrid. Compare AUC, average precision, precision at 10% recall.
6. **Read the gate.** Average the gate value per edge type. The README saw about 0.80 on velocity edges (attention) and about 0.37 on card edges (mean). If you see this kind of split, you have an explanation for why the hybrid works.
7. **Score the test set once** for each final model.
8. **Optional experiment:** feed GNN embeddings into XGBoost. The README found it did not help under the strict protocol. If yours also fails, keep it as a reported negative result.
9. **Export** model weights, entity embedding tables, and the mean embedding (used for unknown cards at serving time).

**Output:** `model.pt`, embeddings, ablation table, gate statistics.
**Done when:** Hybrid beats PureSAGE on all three metrics, and test numbers are in the same range as the README (AUC about 0.854, precision at 10% recall about 0.955).

---

## Phase 4: Explainer (`04_explainability.ipynb`, `src/explainer.py`)

**Goal:** every score comes with a reason a person can read.

1. **Build per-entity stats from the train period only.** For each card, merchant and device: count of payments, count of fraud, fraud rate.
2. **Compute lift** versus the base fraud rate. Only call something risky if lift is real. A busy card at the average fraud rate is just volume, not evidence.
3. **Write reason types.**
   - Card-testing burst (velocity cluster membership, for example "3 tx on this card in 5 minutes, 2 confirmed fraud").
   - Device shared by many cards with prior fraud.
   - Merchant with high fraud lift.
   - Amount far outside normal for the card.
   - Protective reasons, worded as protective.
4. **Rank and weight** the top reasons and return at most a few.
5. **Respect label timing.** Fraud labels take weeks to confirm. A payment from 3 minutes ago is a fact, its label is not. Use only the train-period ledger for labels.
6. **Check faithfulness.** On confirmed test fraud, measure how many get a real reason (README: about 80%). The rest should say honestly that no single relational cause was found.
7. **Check speed.** Reason generation should take under 1 ms.

**Output:** stats file for the explainer, `src/explainer.py`.
**Done when:** reasons appear for most fraud cases and the call is under 1 ms.

---

## Phase 5: Serving sample (`05_serving_sample.ipynb`)

**Goal:** have realistic test traffic for the serving side.

1. **Export a sample of test transactions** with full feature vectors to `data/test_sample_full.csv`.
2. **Include both fraud and normal payments** so you can check that scores separate them.
3. **Download all artifacts** (weights, embeddings, entity maps, stats, scaler) into `artifacts/` on your laptop.

**Done when:** `artifacts/` and `data/test_sample_full.csv` exist locally.

---

## Phase 6: Serving: API, Redis and scorer

**Goal:** one payment in, one decision out, fast.

Work in this order. Test each piece before the next.

1. **`docker-compose.yml`:** Kafka (KRaft mode), Redis, Prometheus, Grafana. Run `docker compose up -d` and confirm all four are up.
2. **`src/serving/redis_store.py`.** Per card store:
   - a sorted set of timestamps (for 1h, 6h, 24h counts),
   - a hash with current velocity numbers and last amount,
   - a capped list of the last 20 transaction ids,
   - expiry of 48 hours.
   Use **increment, then read**, so serving matches how training counted.
3. **`src/serving/feature_builder.py`.** Turn the payload into the 437-number vector.
   - If the caller gives a full vector, use it as-is.
   - Otherwise fill the known fields and use training medians for the rest.
4. **`src/serving/inference.py` (`FraudScorer`).**
   - Load weights, embeddings, entity maps.
   - Look up card, merchant, device. Unknown ones get the mean embedding (cold start).
   - Build a one-transaction subgraph and run the two layers.
   - Apply sigmoid, then the thresholds (0.3 and 0.7).
   - Call the explainer.
5. **`src/serving/metrics.py`.** Define the five Prometheus metrics (see Phase 8).
6. **`src/serving/api.py`.** FastAPI with `POST /score`, `GET /health`, `GET /metrics`. Validate input and return the contract response.
7. **Smoke test.** Run `uvicorn src.serving.api:app --port 8000` and send the curl example from the README.
8. **Check train/serve match.** Score rows from `test_sample_full.csv` with full vectors. Scores should match what the notebook gives for the same rows. If not, fix the feature order, scaling or key rules in `src/features.py` first.

**Done when:** `/score` returns score, decision and reasons, and train/serve scores agree on the sample.

---

## Phase 7: Streaming with Kafka

**Goal:** the same scorer works on a stream.

1. **Create topics:** `transactions` (3 partitions) and `fraud-scores`.
2. **`src/serving/producer.py`:** replay sample transactions into `transactions` at a chosen rate.
3. **`src/serving/consumer.py`:** consumer group reads, calls `FraudScorer`, publishes to `fraud-scores`.
4. **Partition by card id** so one card's payments stay in order.
5. **Handle bad cases:** commit offsets only after a score is published, skip and log bad messages, and make the consumer safe to restart.
6. **Test replay.** Stop the consumer, produce 100 events, start it again, and confirm all 100 are scored.

**Done when:** `python -m src.serving.consumer` and `python -m src.serving.producer --n 100` produce 100 scores in `fraud-scores`.

---

## Phase 8: Monitoring

**Goal:** see the system working, and see it go wrong.

1. **Five metrics:**
   - `fraud_score` histogram,
   - `decision_total` counter (approve, flag, decline),
   - `inference_latency_seconds` histogram,
   - `transaction_amount_mean` (5-minute rolling mean),
   - `feature_drift_alert` (becomes 1 when the rolling mean moves more than 2 training standard deviations from the training mean).
2. **`monitoring/prometheus.yml`:** scrape `/metrics` every 5 seconds.
3. **Grafana:** provision the datasource and dashboard JSON so it loads on first start. Panels: score distribution, decisions, p50/p95/p99 with a 100 ms line, amount mean versus training mean, drift tile (HEALTHY or DRIFT).
4. **Drift test.** Send a burst of $9,999 payments and watch the tile flip within one scrape. Record this as your demo clip.

**Done when:** Grafana at `localhost:3000` shows live data and the drift test turns the tile red.

---

## Phase 9: Testing and measuring

**Goal:** prove it works with numbers you can repeat.

1. **Unit tests.**
   - Feature builder: fixed input gives fixed vector.
   - Redis store: window counts after known timestamps.
   - Threshold logic: edge cases at 0.3 and 0.7.
   - Explainer: risky signal needs real lift.
2. **API tests:** valid payload, missing fields, wrong types, unknown card.
3. **Leakage audit (repeat before every report).**
   - Does any code select a model on the test set?
   - Can test-period nodes reach training through reverse edges?
   - Are fraud labels in the explainer only from the train period?
4. **Speed test.** `python -m src.loadtest --n 100`. Record throughput and p50/p95/p99. Note your machine. README reference: p99 about 15.6 ms on an M-series laptop, CPU only.
5. **Quality test.** Re-run the metric table. Keep the negative results in it.
6. **Failure drills** (see the table below).

**Done when:** tests pass, load test numbers are saved to a file, and the audit list is all "no".

---

## Phase 10: Docs, demo and review

1. **README:** problem, demo, results table, architecture diagram, run steps, known limits.
2. **One-page architecture note:** why each part exists and what it costs.
3. **Demo (3 to 5 minutes):** send a payment, show the reasons, show Grafana under steady traffic, inject the $9,999 burst, show the drift tile turn red.
4. **Technical review file:** what was AI-generated, what you reviewed or changed by hand, and known limits.
5. **Team check:** each member explains the data flow, the main trade-offs, and three failure cases without notes.

**Done when:** a new person can clone the repo, follow the README, and reproduce the demo.

---

## Failure scenarios to plan for

| What fails | What should happen | Status in README |
|---|---|---|
| Unknown card, merchant or device | Use the mean embedding (cold start) | Handled |
| Payload has few fields | Fill the rest with training medians. Scores are compressed. | Handled, documented |
| Kafka is down | `POST /score` still works. Streaming waits. | Your design, test it |
| Redis is down | Velocity counts missing. Add a fallback (score without velocity, log it, raise an alert). | Not built yet |
| Duplicate event | Use transaction id to skip repeats | Not built yet |
| Late or out-of-order event | Order by event time, not arrival time | Not built yet |
| Model files missing | Fail at startup with a clear message, not at first request | Add a startup check |
| Traffic shifts suddenly | Drift alert fires | Built |

---

## Phase 11: Extensions (after version 1)

### A. Add a new payment method (for example UPI or a wallet)

1. **Map the fields.** List what the new method gives you: amount, time, account or handle id, device, merchant. Map each to the existing payload fields.
2. **Add a new node type** if the method has its own identity (for example a UPI id). Add edges such as `uses_upi` and its reverse.
3. **Update `src/features.py`** with the new key rules. Use the same file for training and serving.
4. **Get labeled data** for the new method. A new method has new fraud patterns, so you must retrain. Do not reuse old weights as-is.
5. **Retrain and compare** with the same time-based split and leakage rules.
6. **Roll out safely.** Run the new model in shadow mode first (score but do not act), compare, then switch.

### B. Close the 6-versus-430 field gap
Add a feature store (for example Redis or a managed store) so full vectors are available at request time. This is the biggest quality gain for live scoring.

### C. Better validation
Replace the single time split with walk-forward validation (train on months 1 to 3, test on month 4, then slide forward).

### D. Stricter vocabulary
Build the category vocabulary on the train period only. Map unseen values to an "unknown" bucket.

### E. Production steps
- Graph service that grows from the stream instead of a fixed `graph.pt`.
- Drift alert triggers retraining with a shadow check.
- Blocklist and allowlist before the model, a short-lived score cache, and A/B or shadow model rollout.

---

## Risks and how to handle them

| Risk | Why it matters | What to do |
|---|---|---|
| Leakage | Scores look great but fail in real use | Time split, train-only subgraph, validation for all choices, audit every report |
| Key mismatch between training and serving | Wrong embeddings, silent bad scores | One shared `src/features.py`, a train/serve score match test |
| Live input has few fields | Live scores weaker than offline | State it clearly, plan the feature store |
| Class imbalance (3.5% fraud) | Model predicts "not fraud" for everything | Focal loss, judge by average precision and precision at 10% recall |
| Kaggle limits | GPU quota and session time | Save artifacts after each notebook, keep stages small |
| Latency growth | Misses 100 ms target | Measure after every change, keep subgraph small |
| Overclaiming | Loses trust | Report XGBoost's AUC lead and negative results |

---

## Final checklist

- [ ] Contracts written and reviewed
- [ ] Time-based split saved and reused
- [ ] XGBoost baselines logged in MLflow
- [ ] Graph built, counts and velocity lift checked
- [ ] Hybrid beats PureSAGE on all three metrics
- [ ] Gate values explained
- [ ] Explainer uses train-period labels only
- [ ] Train/serve scores match on the sample
- [ ] `/score` works, Kafka replay works
- [ ] Five metrics visible in Grafana, drift test passes
- [ ] Load test numbers saved with machine details
- [ ] Failure drills done
- [ ] README, architecture note, demo and technical review done
