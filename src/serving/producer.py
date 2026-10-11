"""Kafka Transaction Stream Producer.

Replays real or sample payment transactions into Kafka 'transactions' topic.
Guarantees:
- Partition key = card identifier (ensures strict in-order delivery per card).
- Controlled replay rate (--rate events/sec).
- Supports simulated fallback stream if Kafka cluster is offline.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

logger = logging.getLogger("fraudgraph.producer")
logging.basicConfig(level=logging.INFO)


def create_producer(bootstrap_servers: str = "localhost:9092") -> Optional[Any]:
    if KafkaProducer is None:
        logger.warning("kafka-python not installed; running in local simulated streaming mode.")
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
            retries=3,
            request_timeout_ms=3000,
        )
        logger.info(f"Connected Kafka Producer to {bootstrap_servers}")
        return producer
    except Exception as e:
        logger.warning(f"Could not connect to Kafka at {bootstrap_servers} ({e}). Using local simulated stream.")
        return None


def run_producer(
    sample_file: str = "data/test_sample_full.csv",
    topic: str = "transactions",
    bootstrap_servers: str = "localhost:9092",
    n: int = 100,
    rate: float = 20.0,
    output_sink: Optional[str] = "artifacts/stream_buffer.jsonl",
):
    """Publish transactions to Kafka topic or local simulation sink."""
    df = pd.read_csv(sample_file)
    n = min(n, len(df))
    records = df.iloc[:n].to_dict(orient="records")

    producer = create_producer(bootstrap_servers)
    delay = 1.0 / rate if rate > 0 else 0.0

    print(f"\n[Producer] Replaying {n} transactions to topic '{topic}' at {rate} events/sec...")

    sink_file = open(output_sink, "w", encoding="utf-8") if not producer and output_sink else None

    count = 0
    t_start = time.perf_counter()

    for row in records:
        # Parse features vector if present
        if "features_json" in row and isinstance(row["features_json"], str):
            try:
                row["features"] = json.loads(row["features_json"])
            except Exception:
                pass

        card_key = str(row.get("card1", "unknown"))

        if producer:
            producer.send(topic, key=card_key, value=row)
        elif sink_file:
            sink_file.write(json.dumps(row) + "\n")
            sink_file.flush()

        count += 1
        if delay > 0:
            time.sleep(delay)

    if producer:
        producer.flush()
        producer.close()
    if sink_file:
        sink_file.close()

    elapsed = time.perf_counter() - t_start
    print(f"[Producer] Completed. Produced {count} events in {elapsed:.2f}s ({count/elapsed:.1f} eps).\n")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGraph Kafka Producer")
    parser.add_argument("--sample_file", type=str, default="data/test_sample_full.csv")
    parser.add_argument("--topic", type=str, default="transactions")
    parser.add_argument("--bootstrap_servers", type=str, default="localhost:9092")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()

    run_producer(
        sample_file=args.sample_file,
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        n=args.n,
        rate=args.rate,
    )
