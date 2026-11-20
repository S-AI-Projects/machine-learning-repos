import json
import math
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import BoundedCPUSmokeClassifier, load_fixture


class TestBoundedCPUSmokeRun(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]
        self.train_ids = set(self.fixture["metadata"]["split_policy"]["train_record_ids"])
        self.test_ids = set(self.fixture["metadata"]["split_policy"]["test_record_ids"])

        self.train_records = [r for r in self.records if r["record_id"] in self.train_ids]
        self.test_records = [r for r in self.records if r["record_id"] in self.test_ids]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_vals, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_smoke_training_bounds_and_finite_loss(self):
        model = BoundedCPUSmokeClassifier(max_steps=20)
        summary = model.train_smoke_run(self.train_records)

        # Assert bounds
        self.assertLessEqual(summary["samples_processed"], 32)
        self.assertLessEqual(summary["step_count"], 20)
        self.assertEqual(summary["step_count"], 4)  # 4 train samples processed in 1 epoch
        self.assertEqual(summary["device"], "cpu")

        # Assert finite outputs and loss
        self.assertTrue(math.isfinite(summary["average_loss"]))
        self.assertTrue(math.isfinite(summary["final_step_loss"]))
        for w in summary["weights"]:
            self.assertTrue(math.isfinite(w))
        self.assertTrue(math.isfinite(summary["bias"]))

    def test_prediction_shape_and_range(self):
        model = BoundedCPUSmokeClassifier()
        model.train_smoke_run(self.train_records)

        probs = model.predict_proba(self.test_records)
        preds = model.predict(self.test_records)

        self.assertEqual(len(probs), 4)
        self.assertEqual(len(preds), 4)
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)
            self.assertTrue(math.isfinite(p))
        for y in preds:
            self.assertIn(y, [0, 1])

    def test_exceeding_32_samples_rejected(self):
        oversized = [{"record_id": f"r_{i}", "x1": float(i), "x2": 1.0, "label": i % 2} for i in range(33)]
        model = BoundedCPUSmokeClassifier()
        with self.assertRaises(ValueError):
            model.train_smoke_run(oversized)

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
