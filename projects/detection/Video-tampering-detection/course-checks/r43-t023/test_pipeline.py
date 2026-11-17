import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import MajorityClassBaseline


class TestMajorityClassBaseline(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]
        split_meta = self.fixture_data["split_metadata"]["stratified_split"]
        self.train_ids = set(split_meta["train"])
        self.test_ids = set(split_meta["test"])

        self.train_records = [r for r in self.records if r["record_id"] in self.train_ids]
        self.test_records = [r for r in self.records if r["record_id"] in self.test_ids]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_fit_uses_only_training_data(self):
        model = MajorityClassBaseline()
        model.fit(self.train_records)
        # Train records: 3 rows with label 0 (rec_0, rec_1, rec_2), 2 rows with label 1 (rec_4, rec_5)
        self.assertEqual(model.train_samples, 5)
        self.assertEqual(model.class_counts, {0: 3, 1: 2})
        self.assertEqual(model.majority_class, 0)

    def test_prediction_without_fit_raises(self):
        model = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            model.predict(self.test_records)

    def test_held_out_evaluation_metric(self):
        model = MajorityClassBaseline()
        model.fit(self.train_records)
        eval_res = model.evaluate(self.test_records)
        
        # Test records: rec_3 (label 0), rec_6 (label 1), rec_7 (label 1)
        # Predictions will all be 0.
        # Exactly 1 prediction matches (rec_3). Accuracy = 1/3 ~ 0.3333333333333333
        self.assertEqual(eval_res["eval_sample_count"], 3)
        self.assertEqual(eval_res["correct_predictions"], 1)
        self.assertAlmostEqual(eval_res["accuracy"], 1.0 / 3.0, places=5)
        self.assertEqual(eval_res["majority_class"], 0)


if __name__ == "__main__":
    unittest.main()
