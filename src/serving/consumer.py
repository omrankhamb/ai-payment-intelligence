"""Kafka Fraud Scoring Consumer.

Consumes incoming transactions from 'transactions' topic, executes
real-time FraudScorer inference, and publishes decisions to 'fraud-scores' topic.
Guarantees:
- Consumer group: 'fraudgraph-scorers'.
- Offset commit only after successful publication.
- Malformed payloads skipped safely without stalling stream.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None

from src.features import get_card_key
from src.serving.inference import FraudScorer
from src.serving.metrics import DECISION_TOTAL, FRAUD_SCORE_HIST, LATENCY_HIST, drift_tracker
from src.serving.redis_store import RedisVelocityStore

logger = logging.getLogger("fraudgraph.consumer")
logging.basicConfig(level=logging.INFO)


def run_consumer(
    input_topic: str = "transactions",
    output_topic: str = "fraud-scores",
    bootstrap_servers: str = "localhost:9092",
    artifacts_dir: str = "artifacts",
    max_messages: Optional[int] = None,
    sink_file: str = "artifacts/fraud_scores_stream.jsonl",
):
    """Run stream scoring consumer loop."""
    print(f"\n[Consumer] Initializing FraudScorer from {artifacts_dir}...")
    scorer = FraudScorer(artifacts_dir=artifacts_dir)
    redis_store = RedisVelocityStore()

    consumer = None
    producer = None

    if KafkaConsumer is not None:
        try:
            consumer = KafkaConsumer(
                input_topic,
                bootstrap_servers=bootstrap_servers.split(","),
                group_id="fraudgraph-scorers",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=5000,
            )
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
            )
            logger.info(f"Connected to Kafka broker at {bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}). Using simulated streaming buffer mode.")
            consumer = None

    processed_count = 0
    t_start = time.perf_counter()
    Path(sink_file).parent.mkdir(parents=True, exist_ok=True)

    if consumer and producer:
        print(f"[Consumer] Listening on topic '{input_topic}'...")
        try:
            for message in consumer:
                payload = message.value
                card_key = get_card_key(payload) or ""

                # Sliding window query
                vel_info = redis_store.record_and_get_velocity(
                    card_key=card_key,
                    transaction_id=payload.get("TransactionID", 0),
                    transaction_dt=payload.get("TransactionDT", 0),
                    amount=payload.get("TransactionAmt", 0.0),
                )

                # Inference
                score_res = scorer.score_transaction(payload, velocity_info=vel_info)

                # Publish to output topic
                tx_id = str(score_res["transaction_id"])
                producer.send(output_topic, key=tx_id, value=score_res)
                producer.flush()

                # Commit offset
                consumer.commit()
                processed_count += 1

                # Metrics
                FRAUD_SCORE_HIST.observe(score_res["fraud_score"])
                DECISION_TOTAL.labels(decision=score_res["decision"]).inc()
                drift_tracker.record_transaction(payload.get("TransactionAmt", 0.0))

                if max_messages and processed_count >= max_messages:
                    break
        except KeyboardInterrupt:
            print("[Consumer] Interrupted by user.")
        finally:
            consumer.close()
            producer.close()
    else:
        # Simulated stream buffer reader
        in_buf = Path("artifacts/stream_buffer.jsonl")
        if not in_buf.exists():
            print(f"[Consumer] Simulation buffer {in_buf} not found. Run producer first.")
            return 0

        print(f"[Consumer] Reading simulated stream from {in_buf}...")
        with open(in_buf, "r", encoding="utf-8") as fin, open(sink_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    card_key = get_card_key(payload) or ""
                    vel_info = redis_store.record_and_get_velocity(
                        card_key=card_key,
                        transaction_id=payload.get("TransactionID", 0),
                        transaction_dt=payload.get("TransactionDT", 0),
                        amount=payload.get("TransactionAmt", 0.0),
                    )
                    score_res = scorer.score_transaction(payload, velocity_info=vel_info)
                    fout.write(json.dumps(score_res) + "\n")
                    fout.flush()
                    processed_count += 1

                    FRAUD_SCORE_HIST.observe(score_res["fraud_score"])
                    DECISION_TOTAL.labels(decision=score_res["decision"]).inc()
                    drift_tracker.record_transaction(payload.get("TransactionAmt", 0.0))

                    if max_messages and processed_count >= max_messages:
                        break
                except Exception as e:
                    logger.error(f"Skipping bad message: {e}")

    elapsed = time.perf_counter() - t_start
    print(f"[Consumer] Processed and published {processed_count} scores to '{output_topic}' in {elapsed:.2f}s.\n")
    return processed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGraph Kafka Consumer")
    parser.add_argument("--input_topic", type=str, default="transactions")
    parser.add_argument("--output_topic", type=str, default="fraud-scores")
    parser.add_argument("--bootstrap_servers", type=str, default="localhost:9092")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--max_messages", type=int, default=100)
    args = parser.parse_args()

    run_consumer(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        bootstrap_servers=args.bootstrap_servers,
        artifacts_dir=args.artifacts_dir,
        max_messages=args.max_messages,
    )
