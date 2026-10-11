# FraudGraph Demo Guide (3-5 Minutes Walkthrough)

This guide walks through the live end-to-end demonstration of the FraudGraph payment intelligence system.

---

## Step 1: Health & Readiness Probe
Verify that the microservice is healthy:
```bash
curl -X GET http://localhost:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "fraudgraph-scorer",
  "version": "1.0.0",
  "redis_connected": true
}
```

---

## Step 2: Live Scoring with Plain-Text Explanations

### Scenario A: Low-Risk Legitimate Payment
Send a standard payment with established card history:
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 4000001,
    "TransactionDT": 15000000,
    "TransactionAmt": 45.00,
    "ProductCD": "W",
    "card1": 20005,
    "card2": 321,
    "card3": 150,
    "card4": "visa",
    "card5": 102,
    "card6": "debit",
    "addr1": 315.0
  }'
```
**Decision:** `APPROVE` with protective explanation:
`"Card has established history of 30 verified legitimate transactions with 0 disputes"`

### Scenario B: High-Risk Velocity Burst & Device Sharing
Send an aggressive card-testing payment:
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 4000002,
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
    "DeviceInfo": "iOS Device"
  }'
```
**Decision:** `FLAG` or `DECLINE` with actionable reasons:
- `"Device shared across 7 distinct cards with prior fraud history"`
- `"Payment amount ($999.00) exceeds the 99th percentile of platform traffic"`

---

## Step 3: Kafka Streaming Replay
Replay 100 transactions into the streaming pipeline:
```bash
python -m src.serving.producer --n 100 --rate 25
python -m src.serving.consumer --max_messages 100
```
Inspect produced decisions in `artifacts/fraud_scores_stream.jsonl`.

---

## Step 4: Live Prometheus & Grafana Monitoring
Open `http://localhost:3000` (User: `admin` / Password: `admin`):
- Observe **Decisions Issued** (Approve / Flag / Decline distribution).
- Verify **p99 Latency** is safely below the 100 ms red line.
- Observe **Drift Alert Status** tile currently displaying **HEALTHY (Green)**.

---

## Step 5: Drift Injection & Alert Verification
Inject an anomalous stream of high-value payments ($9,999):
```bash
python -m src.drift_simulation
```
**Result in Grafana:**
1. The rolling 5-minute amount mean spikes from $85 to $4,491.
2. The **Drift Alert Status** tile automatically flips to **DRIFT ALERT (Red)**.
