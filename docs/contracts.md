# FraudGraph Contracts & Specification

## 1. Problem Statement
- **User:** Fraud operations and risk investigation team.
- **Input:** Single payment transaction in real time.
- **Output:** Fraud risk score (`0.0 - 1.0`), automated decision (`APPROVE`, `FLAG`, `DECLINE`), and human-interpretable risk reasons.

---

## 2. Success Targets & Metrics

### Quality Targets (evaluated on time-based test set):
- **Primary Metric:** Precision at 10% Recall ($\ge 0.95$ target). Fraud teams have limited review capacity; top flagged items must be overwhelmingly true positives.
- **Secondary Metric:** ROC-AUC ($\ge 0.85$ target for GNN, benchmarked against XGBoost baselines).
- **Leakage Zero-Tolerance:** Strictly chronological splits (train 70%, val 10%, test 20%). Message-passing during training restricted to train-period subgraph.

### Latency Targets:
- **Decision Latency (p99):** $< 100\text{ ms}$ end-to-end (scoring + explainer + Redis velocity fetch).

---

## 3. System Contracts

### Contract 1: Request Payload (`POST /score`)
The input payment event received from upstream checkout / payment gateway.

```json
{
  "TransactionID": 3000001,
  "TransactionDT": 86400,
  "TransactionAmt": 150.00,
  "ProductCD": "W",
  "card1": 13926,
  "addr1": 315,
  "card2": 321,
  "card3": 150,
  "card4": "discover",
  "card5": 102,
  "card6": "credit",
  "addr2": 87,
  "P_emaildomain": "gmail.com",
  "R_emaildomain": null,
  "DeviceType": "mobile",
  "DeviceInfo": "iOS Device",
  "features": null
}
```

#### Fields Description:
- **Required Core Fields:**
  - `TransactionID` (`int`): Unique identifier for transaction.
  - `TransactionDT` (`int`): Timestep in seconds from reference point.
  - `TransactionAmt` (`float`): Transaction payment amount.
  - `ProductCD` (`str`): Product code category (e.g. `'W'`, `'C'`, `'R'`).
  - `card1` (`int` / `str`): Primary payment card identifier.
  - `addr1` (`int` / `float`): Billing region / address code.
- **Optional Fields:**
  - Card & Address attributes: `card2`-`card6`, `addr2`, `P_emaildomain`, `R_emaildomain`.
  - Identity / Device attributes: `DeviceType`, `DeviceInfo`, `id_01` to `id_38`.
  - `features` (`list[float]`, optional): 437 pre-computed numerical features. When provided by an upstream feature store, used directly. If omitted, filled using live entity lookup + training medians.

---

### Contract 2: Feature Vector
Fixed 437-dimensional float vector feeding the transaction encoder:
$$\mathbf{x} \in \mathbb{R}^{437}$$

- **Dimension 0:** `TransactionAmt` (log1p transformed and z-score standardized).
- **Dimensions 1 - 20:** Categorical frequency and target encodings (`ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, etc.).
- **Dimensions 21 - 436:** IEEE-CIS numeric features (`C1`-`C14`, `D1`-`D15`, `V1`-`V339`, `id_01`-`id_11`) imputing missing values with training-set medians and scaled using standard scaler computed strictly on the training set.

---

### Contract 3: Response Payload
Returned by FastAPI `/score` and emitted to Kafka `fraud-scores` topic.

```json
{
  "transaction_id": 3000001,
  "fraud_score": 0.8421,
  "decision": "DECLINE",
  "thresholds": {
    "approve_below": 0.3,
    "decline_above": 0.7
  },
  "top_reasons": [
    {
      "factor": "velocity_burst",
      "weight": 0.45,
      "edge_type": "velocity_cluster",
      "description": "3 transactions on this card in 5 minutes with 2 confirmed fraud charges"
    },
    {
      "factor": "high_risk_device",
      "weight": 0.32,
      "edge_type": "on_device",
      "description": "Device shared with 4 distinct cards having previous fraud history"
    }
  ],
  "latency_ms": 14.2
}
```

---

## 4. Decision Thresholds

| Fraud Score Range | Decision | Action |
|---|---|---|
| $[0.00, 0.30)$ | `APPROVE` | Transaction processed seamlessly. Low risk. |
| $[0.30, 0.70]$ | `FLAG` | Queued for step-up auth (3DS OTP / biometric) or manual analyst review. |
| $(0.70, 1.00]$ | `DECLINE` | Automated decline to prevent immediate chargeback losses. |

---

## 5. Development & Git Workflow Rules
1. **Branching:** Work performed on feature branches (`phase-X-<name>`), merged via PR or reviewed milestone commits.
2. **Commit Policy:** Atomic, descriptive commits corresponding to each phase step.
3. **No Leakage Policy:** All scalers, encoders, and explainer entity statistics must be fitted strictly on the train partition ($T < T_{\text{val}}$).
4. **Reproducibility:** Fix random seeds across PyTorch, NumPy, and XGBoost ($42$).
