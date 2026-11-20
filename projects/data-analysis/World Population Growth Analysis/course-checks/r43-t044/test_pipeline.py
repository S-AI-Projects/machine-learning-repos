import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import calculate_classification_metrics, calculate_mae, load_fixture


class TestEvaluationMetricVerification(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]

    def test_canonical_classification_precision_recall(self):
        # Explicit Done When check:
        # truth[1,1,0,0]/prediction[1,0,1,0] -> precision=0.5, recall=0.5
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        metrics = calculate_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertEqual(metrics["tn"], 1.0)

    def test_canonical_regression_mae(self):
        # Explicit Done When check:
        # truth[1,2,3]/prediction[1,3,2] -> MAE=2/3
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 3.0, 2.0]
        mae = calculate_mae(y_true, y_pred)
        self.assertAlmostEqual(mae, 2.0 / 3.0, places=6)

    def test_zero_denominator_handling(self):
        # All negative predictions -> tp + fp == 0 -> precision=0.0
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        metrics = calculate_classification_metrics(y_true, y_pred, zero_division=0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["accuracy"], 1.0)

        # Custom zero_division value
        metrics_custom = calculate_classification_metrics(y_true, y_pred, zero_division=1.0)
        self.assertEqual(metrics_custom["precision"], 1.0)
        self.assertEqual(metrics_custom["recall"], 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            calculate_classification_metrics([1, 0], [1])
        with self.assertRaises(ValueError):
            calculate_mae([1.0, 2.0], [1.0])

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_vals, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
