"""Test Runner using Python Standard Library."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import unittest
from fastapi.testclient import TestClient

from tests.test_unit import (
    test_feature_builder_deterministic,
    test_redis_store_sliding_windows,
    test_decision_threshold_boundaries,
    test_explainer_lift_requirement,
)
from tests.test_leakage_audit import test_leakage_audit_checklist
from tests.test_api import (
    test_api_valid_payload,
    test_api_missing_fields,
    test_api_cold_start_unknown_card,
)
from src.serving.api import app


class TestFraudGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_01_feature_builder(self):
        test_feature_builder_deterministic()

    def test_02_redis_sliding_windows(self):
        test_redis_store_sliding_windows()

    def test_03_decision_thresholds(self):
        test_decision_threshold_boundaries()

    def test_04_explainer_lift(self):
        test_explainer_lift_requirement()

    def test_05_leakage_audit(self):
        test_leakage_audit_checklist()

    def test_06_api_valid(self):
        test_api_valid_payload(self.client)

    def test_07_api_missing_fields(self):
        test_api_missing_fields(self.client)

    def test_08_api_cold_start(self):
        test_api_cold_start_unknown_card(self.client)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFraudGraph)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
