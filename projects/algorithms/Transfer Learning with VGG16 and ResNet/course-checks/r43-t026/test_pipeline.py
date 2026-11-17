import json
import math
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import BoundedCPUSmokeClassifier


class TestTransferLearningSmokeRun(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]
        split_meta = self.fixture_data["split_metadata"]["stratified_split"]
        self.train_records = [r for r in self.records if r["record_id"] in set(split_meta["train"])]
        self.test_records = [r for r in self.records if r["record_id"] in set(split_meta["test"])]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_bounded_cpu_smoke_run(self):
        model = BoundedCPUSmokeClassifier(learning_rate=0.05, max_steps=20, seed=42)
        summary = model.train_smoke_run(self.train_records)

        # 1. Assert step count is bounded (<= 20) and non-zero
        self.assertGreater(summary["step_count"], 0)
        self.assertLessEqual(summary["step_count"], 20)
        self.assertEqual(summary["step_count"], len(self.train_records))

        # 2. Assert loss is finite, non-null, and positive
        self.assertTrue(math.isfinite(summary["average_loss"]))
        self.assertGreater(summary["average_loss"], 0.0)
        for loss in model.loss_history:
            self.assertTrue(math.isfinite(loss))

        # 3. Assert prediction outputs have expected shape and range
        test_preds = model.predict(self.test_records)
        self.assertEqual(len(test_preds), len(self.test_records))
        for p in test_preds:
            self.assertIn(p, (0, 1))

        # 4. Probabilities are strictly in [0.0, 1.0]
        test_probs = model.predict_proba(self.test_records)
        self.assertEqual(len(test_probs), len(self.test_records))
        for prob in test_probs:
            self.assertTrue(0.0 <= prob <= 1.0)

    def test_sample_bound_enforced(self):
        # Smoke run bound specifies at most 32 samples
        oversized_records = [{"record_id": f"rec_{i}", "x1": 1.0, "x2": 1.0, "label": 0} for i in range(35)]
        model = BoundedCPUSmokeClassifier()
        with self.assertRaises(ValueError):
            model.train_smoke_run(oversized_records)


if __name__ == "__main__":
    unittest.main()
