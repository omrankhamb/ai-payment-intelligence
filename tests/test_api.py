"""Integration & API Contract Tests.

Validates:
1. Valid payload succeeds with Contract 3 response schema.
2. Missing required fields returns HTTP 422.
3. Unknown card entity receives cold-start mean embedding and scores gracefully.
4. Latency is included and is < 100 ms.
"""

from fastapi.testclient import TestClient

from src.serving.api import app


def get_test_client():
    return TestClient(app)


def test_api_valid_payload(client):
    payload = {
        "TransactionID": 9000001,
        "TransactionDT": 100000,
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "card1": 13926,
        "addr1": 315.0,
        "card4": "visa",
        "card6": "debit",
        "DeviceType": "mobile",
        "DeviceInfo": "iOS Device",
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["transaction_id"] == 9000001
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["decision"] in ["APPROVE", "FLAG", "DECLINE"]
    assert "thresholds" in data
    assert isinstance(data["top_reasons"], list)
    assert len(data["top_reasons"]) > 0
    assert data["latency_ms"] < 100.0, f"Latency {data['latency_ms']}ms exceeded 100ms threshold"


def test_api_missing_fields(client):
    # Missing required TransactionAmt and ProductCD
    bad_payload = {
        "TransactionID": 9000002,
        "TransactionDT": 100000,
        "card1": 13926,
    }
    resp = client.post("/score", json=bad_payload)
    assert resp.status_code == 422


def test_api_cold_start_unknown_card(client):
    # Completely novel card and merchant
    payload = {
        "TransactionID": 9000003,
        "TransactionDT": 100050,
        "TransactionAmt": 75.00,
        "ProductCD": "S",
        "card1": "totally_new_unseen_card_99999",
        "addr1": 999.0,
        "card4": "discover",
        "card6": "credit",
        "DeviceType": "desktop",
        "DeviceInfo": "Chrome 120",
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["decision"] in ["APPROVE", "FLAG", "DECLINE"]
