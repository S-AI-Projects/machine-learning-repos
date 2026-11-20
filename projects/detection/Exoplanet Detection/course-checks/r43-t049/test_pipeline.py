import json
import os
import random
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import RobustClassificationInterface, PerturbationRejectionError, load_fixture


class TestInputPerturbationRobustness(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_vals, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_record_order_permutation_invariance_with_aligned_ids(self):
        clf = RobustClassificationInterface(threshold=3.5)
        
        # Original canonical ordering
        baseline_preds = clf.predict_batch_aligned(self.records)

        # Perturbed ordering: reversed
        reversed_records = list(reversed(self.records))
        reversed_preds = clf.predict_batch_aligned(reversed_records)
        self.assertEqual(baseline_preds, reversed_preds)

        # Perturbed ordering: fixed pseudorandom shuffle
        rng = random.Random(42)
        shuffled_records = list(self.records)
        rng.shuffle(shuffled_records)
        shuffled_preds = clf.predict_batch_aligned(shuffled_records)
        self.assertEqual(baseline_preds, shuffled_preds)

        # Verify expected binary classifications
        expected = {"r1": 0, "r2": 0, "r3": 0, "r4": 0, "r5": 1, "r6": 1, "r7": 1, "r8": 1}
        self.assertEqual(baseline_preds, expected)

    def test_unsupported_input_missing_field_rejected(self):
        clf = RobustClassificationInterface(fallback_prediction=None)
        
        # Missing required feature x1
        corrupted_record = {"record_id": "r_bad", "x2": 1.0}
        with self.assertRaises(PerturbationRejectionError) as ctx:
            clf.predict_record(corrupted_record)
        self.assertIn("Missing or malformed field 'x1'", str(ctx.exception))

    def test_unsupported_input_missing_id_rejected(self):
        clf = RobustClassificationInterface()
        missing_id_record = {"x1": 2.0, "x2": 1.0}
        with self.assertRaises(PerturbationRejectionError) as ctx:
            clf.predict_record(missing_id_record)
        self.assertIn("missing required 'record_id'", str(ctx.exception))

    def test_explicit_fallback_mapping(self):
        # When configured with explicit fallback, corrupted input maps to fallback
        clf_with_fallback = RobustClassificationInterface(fallback_prediction=0)
        corrupted_record = {"record_id": "r_corrupt", "x1": None, "x2": 1.0}
        pred = clf_with_fallback.predict_record(corrupted_record)
        self.assertEqual(pred, 0)

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
