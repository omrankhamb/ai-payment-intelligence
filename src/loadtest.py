"""Load Testing and Latency Benchmarking.

Sends N scoring requests through the inference pipeline and computes:
- Throughput (requests / second)
- Latency distribution: p50, p90, p95, p99, max (ms)
- Machine hardware specifications
Saves benchmark report to artifacts/loadtest_results.json.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time
from typing import Dict, List
import numpy as np
import pandas as pd

from src.serving.inference import FraudScorer
from src.serving.redis_store import RedisVelocityStore


def run_load_test(
    sample_file: str = "data/test_sample_full.csv",
    n: int = 100,
    artifacts_dir: str = "artifacts",
    output_file: str = "artifacts/loadtest_results.json",
) -> Dict:
    scorer = FraudScorer(artifacts_dir=artifacts_dir)
    redis_store = RedisVelocityStore()

    df = pd.read_csv(sample_file)
    n = min(n, len(df))
    records = df.iloc[:n].to_dict(orient="records")

    print(f"\n--- [Phase 9: Load & Latency Test (N={n})] ---")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor() or 'CPU'}")

    latencies_ms: List[float] = []

    # Warmup
    scorer.score_transaction(records[0])

    t_total_start = time.perf_counter()
    for row in records:
        if "features_json" in row and isinstance(row["features_json"], str):
            try:
                row["features"] = json.loads(row["features_json"])
            except Exception:
                pass

        card_key = str(row.get("card1", "card_test"))
        t0 = time.perf_counter()

        vel_info = redis_store.record_and_get_velocity(
            card_key=card_key,
            transaction_id=row.get("TransactionID", 0),
            transaction_dt=row.get("TransactionDT", 0),
            amount=row.get("TransactionAmt", 0.0),
        )
        scorer.score_transaction(row, velocity_info=vel_info)

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    total_duration = time.perf_counter() - t_total_start
    throughput = round(n / total_duration, 1)

    latencies_arr = np.array(latencies_ms)
    p50 = round(float(np.percentile(latencies_arr, 50)), 2)
    p90 = round(float(np.percentile(latencies_arr, 90)), 2)
    p95 = round(float(np.percentile(latencies_arr, 95)), 2)
    p99 = round(float(np.percentile(latencies_arr, 99)), 2)
    max_lat = round(float(np.max(latencies_arr)), 2)
    mean_lat = round(float(np.mean(latencies_arr)), 2)

    results = {
        "n_requests": n,
        "total_duration_sec": round(total_duration, 3),
        "throughput_rps": throughput,
        "latency_ms": {
            "mean": mean_lat,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": max_lat,
        },
        "target_p99_ms": 100.0,
        "sla_pass": p99 < 100.0,
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }

    print("\n--- Latency Benchmark Results ---")
    print(f"Requests:   {n}")
    print(f"Throughput: {throughput} req/s")
    print(f"Mean:       {mean_lat} ms")
    print(f"p50:        {p50} ms")
    print(f"p95:        {p95} ms")
    print(f"p99:        {p99} ms (Target SLA: < 100.0 ms)")
    print(f"SLA Pass:   {results['sla_pass']}")
    print("---------------------------------\n")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark numbers to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGraph Load Test")
    parser.add_argument("--sample_file", type=str, default="data/test_sample_full.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--output_file", type=str, default="artifacts/loadtest_results.json")
    args = parser.parse_args()

    run_load_test(
        sample_file=args.sample_file,
        n=args.n,
        artifacts_dir=args.artifacts_dir,
        output_file=args.output_file,
    )
