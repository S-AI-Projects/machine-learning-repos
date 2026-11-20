import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import MajorityClassBaseline, load_fixture


class TestMajorityClassBaseline(unittest.TestCase):
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

    def test_stratified_split_isolation(self):
        self.assertTrue(self.train_ids.isdisjoint(self.test_ids))
        self.assertEqual(len(self.train_records), 4)
        self.assertEqual(len(self.test_records), 4)

    def test_unfitted_prediction_raises(self):
        baseline = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            baseline.predict(self.test_records)

    def test_baseline_trained_strictly_on_train_data(self):
        baseline = MajorityClassBaseline()
        baseline.fit(self.train_records)
        self.assertEqual(baseline.train_samples, 4)
        self.assertEqual(baseline.majority_class, 0)

        preds = baseline.predict(self.test_records)
        self.assertEqual(preds, [0, 0, 0, 0])

        metrics = baseline.evaluate(self.test_records)
        self.assertEqual(metrics["eval_sample_count"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["correct_predictions"], 2)

    def test_majority_updates_with_skewed_training(self):
        # When training distribution changes, baseline adapts solely to training
        skewed_train = [
            {"record_id": "s1", "label": 1},
            {"record_id": "s2", "label": 1},
            {"record_id": "s3", "label": 0}
        ]
        baseline = MajorityClassBaseline().fit(skewed_train)
        self.assertEqual(baseline.majority_class, 1)
        self.assertEqual(baseline.predict([{"record_id": "test"}]), [1])

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
