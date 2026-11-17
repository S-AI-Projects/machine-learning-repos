import json
import os
import random
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import RobustClassificationInterface, PerturbationRejectionError


class TestInputPerturbation(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]
        self.model = RobustClassificationInterface(threshold=3.5)

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_record_order_permutation_preserves_predictions_after_id_alignment(self):
        # 1. Baseline predictions in original order
        baseline_preds = self.model.predict_batch_aligned(self.records)

        # 2. Perturbed order (reversed and deterministic pseudorandom permutation)
        reversed_records = list(reversed(self.records))
        reversed_preds = self.model.predict_batch_aligned(reversed_records)

        # Ensure prediction values matched by record_id are strictly identical
        self.assertEqual(baseline_preds, reversed_preds)

        # Shuffled records
        rng = random.Random(42)
        shuffled_records = list(self.records)
        rng.shuffle(shuffled_records)
        shuffled_preds = self.model.predict_batch_aligned(shuffled_records)
        self.assertEqual(baseline_preds, shuffled_preds)

    def test_unsupported_missing_field_rejected(self):
        # Record with missing x1 is rejected
        corrupted_record = {"record_id": "bad_rec", "group_id": "grp_0", "x2": 1.0}
        with self.assertRaises(PerturbationRejectionError) as ctx:
            self.model.predict_record(corrupted_record)
        self.assertIn("x1", str(ctx.exception))

    def test_unsupported_input_explicit_fallback_mapping(self):
        # Model configured with explicit fallback
        fallback_model = RobustClassificationInterface(threshold=3.5, fallback_prediction=-1)
        corrupted_record = {"record_id": "bad_rec_fallback", "x2": 1.0}
        pred = fallback_model.predict_record(corrupted_record)
        self.assertEqual(pred, -1)


if __name__ == "__main__":
    unittest.main()
