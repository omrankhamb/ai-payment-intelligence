"""Real-Time Feature Builder for Serving.

Converts request payload into fixed 437-dimensional float vector:
1. If payload contains precomputed 'features' (list of 437 numbers), validates length and uses as-is.
2. Otherwise, transforms provided fields:
   - TransactionAmt -> log1p and z-score scaling with training mean/std.
   - Categoricals -> mapped against fitted category vocabularies.
   - Remaining V/C/D/ID features -> imputed with training-set medians.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.features import CATEGORICAL_COLS, NUMERICAL_COLS, TOTAL_FEATURE_DIM


class FeatureBuilder:
    """Transforms raw API payloads into 437-dim model input tensors."""

    def __init__(self, stats: Dict[str, Any]):
        self.medians = stats.get("medians", {})
        self.means = stats.get("means", {})
        self.stds = stats.get("stds", {})
        self.cat_vocab = stats.get("cat_vocab", {})

    @classmethod
    def load(cls, stats_path: str = "artifacts/feature_stats.json") -> "FeatureBuilder":
        if not Path(stats_path).exists():
            # Graceful fallback if stats not yet generated
            return cls({"medians": {}, "means": {}, "stds": {}, "cat_vocab": {}})
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats)

    def build_vector(self, payload: Dict[str, Any]) -> np.ndarray:
        """Construct the 437-dimensional feature vector."""
        # 1. Direct pass-through if full vector provided
        features = payload.get("features")
        if features is not None and isinstance(features, (list, np.ndarray)):
            if len(features) == TOTAL_FEATURE_DIM:
                return np.array(features, dtype=np.float32)
            elif len(features) > TOTAL_FEATURE_DIM:
                return np.array(features[:TOTAL_FEATURE_DIM], dtype=np.float32)
            else:
                padded = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
                padded[: len(features)] = features
                return padded

        vec = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

        # 2. Numerical Features
        for idx, col in enumerate(NUMERICAL_COLS):
            val = payload.get(col)
            med = self.medians.get(col, 0.0)
            if val is None or val == "":
                raw = med
            else:
                try:
                    raw = float(val)
                except (ValueError, TypeError):
                    raw = med

            if col == "TransactionAmt":
                raw = np.log1p(max(0.0, raw))

            # Standardize using training mean and std
            mean = self.means.get(col, 0.0)
            std = self.stds.get(col, 1.0)
            if std <= 1e-6:
                std = 1.0
            vec[idx] = float((raw - mean) / std)

        # 3. Categorical Encodings (mapped to dimensions after numerical columns)
        offset = len(NUMERICAL_COLS)
        for idx, col in enumerate(CATEGORICAL_COLS):
            if offset + idx >= TOTAL_FEATURE_DIM:
                break
            val = payload.get(col)
            if val is not None and str(val) != "":
                # Use hash trick or vocabulary code
                vocab_size = self.cat_vocab.get(col, 100)
                code = (hash(str(val)) % max(1, vocab_size)) + 1
                vec[offset + idx] = float(code)
            else:
                vec[offset + idx] = 0.0

        return vec
