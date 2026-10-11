"""Real-Time Redis Velocity & Window Store.

Per-Card Store:
1. Sorted set of timestamps (`zadd card:{id}:ts`) for 5m, 1h, 6h, and 24h sliding window counts.
2. Hash (`hset card:{id}:state`) tracking last amount and cumulative metrics.
3. Capped list (`lpush card:{id}:recent_tx`) storing last 20 transaction IDs.
4. Key TTL: 48 hours (sliding expiry).

Ordering Guarantee:
- Increment FIRST, then read, ensuring serving-time feature counts strictly match training.
- Graceful degradation fallback if Redis is unavailable.
"""

import logging
import time
from typing import Any, Dict, Optional

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger("fraudgraph.redis")


class RedisVelocityStore:
    """Sliding-window card velocity accumulator."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, timeout: float = 0.5):
        self.host = host
        self.port = port
        self.client: Optional[Any] = None

        if redis is not None:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    socket_connect_timeout=timeout,
                    socket_timeout=timeout,
                    decode_responses=True,
                )
                self.client.ping()
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}). Using in-memory fallback velocity store.")
                self.client = None
        else:
            logger.warning("redis-py not installed; velocity store running in fallback mode.")

        # In-memory fallback if Redis is unreachable
        self._fallback_zsets: Dict[str, list] = {}
        self._fallback_hashes: Dict[str, Dict[str, Any]] = {}
        self._fallback_lists: Dict[str, list] = {}

    def record_and_get_velocity(
        self,
        card_key: str,
        transaction_id: int,
        transaction_dt: int,
        amount: float,
    ) -> Dict[str, Any]:
        """Atomic sliding window increment and readout.

        Windows computed:
        - 5-minute burst (300s)
        - 1-hour window (3600s)
        - 6-hour window (21600s)
        - 24-hour window (86400s)
        """
        if not card_key:
            return {
                "tx_count_5m": 1,
                "tx_count_1h": 1,
                "tx_count_6h": 1,
                "tx_count_24h": 1,
                "last_amount": None,
                "prior_fraud_count": 0,
            }

        ts = float(transaction_dt)

        if self.client:
            try:
                pipe = self.client.pipeline()
                ts_key = f"card:{card_key}:ts"
                state_key = f"card:{card_key}:state"
                recent_key = f"card:{card_key}:recent"

                # 1. Add current timestamp
                pipe.zadd(ts_key, {f"{transaction_id}:{ts}": ts})

                # 2. Window count queries
                pipe.zcount(ts_key, ts - 300, ts)
                pipe.zcount(ts_key, ts - 3600, ts)
                pipe.zcount(ts_key, ts - 21600, ts)
                pipe.zcount(ts_key, ts - 86400, ts)

                # 3. Read prior state
                pipe.hgetall(state_key)

                # 4. Update state with current amount and timestamp
                pipe.hset(state_key, mapping={"last_amount": str(amount), "last_ts": str(ts)})

                # 5. Push transaction ID and trim list to 20
                pipe.lpush(recent_key, str(transaction_id))
                pipe.ltrim(recent_key, 0, 19)

                # 6. Set 48h expiration (172800s)
                pipe.expire(ts_key, 172800)
                pipe.expire(state_key, 172800)
                pipe.expire(recent_key, 172800)

                results = pipe.execute()

                c_5m = int(results[1])
                c_1h = int(results[2])
                c_6h = int(results[3])
                c_24h = int(results[4])
                prior_state = results[5] or {}
                last_amt = float(prior_state.get("last_amount")) if "last_amount" in prior_state else None
                prior_fraud = int(prior_state.get("prior_fraud_count", 0))

                return {
                    "tx_count_5m": max(1, c_5m),
                    "tx_count_1h": max(1, c_1h),
                    "tx_count_6h": max(1, c_6h),
                    "tx_count_24h": max(1, c_24h),
                    "last_amount": last_amt,
                    "prior_fraud_count": prior_fraud,
                }
            except Exception as e:
                logger.error(f"Redis operation failed ({e}), falling back to memory store.")

        # In-memory fallback
        zset = self._fallback_zsets.setdefault(card_key, [])
        zset.append((transaction_id, ts))
        # Keep only last 48h
        zset = [item for item in zset if item[1] >= ts - 172800]
        self._fallback_zsets[card_key] = zset

        c_5m = sum(1 for _, t in zset if ts - 300 <= t <= ts)
        c_1h = sum(1 for _, t in zset if ts - 3600 <= t <= ts)
        c_6h = sum(1 for _, t in zset if ts - 21600 <= t <= ts)
        c_24h = sum(1 for _, t in zset if ts - 86400 <= t <= ts)

        state = self._fallback_hashes.setdefault(card_key, {})
        last_amt = state.get("last_amount")
        state["last_amount"] = amount

        recent = self._fallback_lists.setdefault(card_key, [])
        recent.insert(0, transaction_id)
        self._fallback_lists[card_key] = recent[:20]

        return {
            "tx_count_5m": max(1, c_5m),
            "tx_count_1h": max(1, c_1h),
            "tx_count_6h": max(1, c_6h),
            "tx_count_24h": max(1, c_24h),
            "last_amount": last_amt,
            "prior_fraud_count": 0,
        }
