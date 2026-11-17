import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import calculate_classification_metrics, calculate_mae, classify_diagonal_image


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]

    def test_fixture_exact_dimensions_and_structure(self):
        # 6 synthetic 4x4 grayscale images: 3 all-zero (label 0), 3 white-diagonal (label 1)
        self.assertEqual(len(self.records), 6)
        labels = [r["label"] for r in self.records]
        self.assertEqual(labels, [0, 0, 0, 1, 1, 1])
        
        # Verify shape of each image is 4x4
        for r in self.records:
            img = r["image"]
            self.assertEqual(len(img), 4)
            for row in img:
                self.assertEqual(len(row), 4)

        # Check unique record IDs and groups
        ids = [r["record_id"] for r in self.records]
        self.assertEqual(len(set(ids)), 6)
        groups = {r["group_id"] for r in self.records}
        self.assertEqual(groups, {"grp_zero", "grp_diag"})

    def test_canonical_classification_metric_contract(self):
        # Explicit Done-When criterion: truth=[1,1,0,0], prediction=[1,0,1,0] -> precision=recall=0.5
        truth = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        metrics = calculate_classification_metrics(truth, prediction)
        
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertEqual(metrics["tn"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=7)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=7)
        self.assertAlmostEqual(metrics["f1"], 0.5, places=7)

    def test_canonical_regression_metric_contract(self):
        # Explicit Done-When criterion: truth=[1,2,3], prediction=[1,3,2] -> MAE = 2/3
        truth = [1, 2, 3]
        prediction = [1, 3, 2]
        mae = calculate_mae(truth, prediction)
        self.assertAlmostEqual(mae, 2.0 / 3.0, places=6)

    def test_zero_denominator_behavior(self):
        # When TP + FP = 0 (no positive predictions)
        truth = [1, 1]
        prediction = [0, 0]
        metrics = calculate_classification_metrics(truth, prediction, zero_division=0.0)
        self.assertEqual(metrics["precision"], 0.0)

        # When TP + FN = 0 (no positive true labels)
        truth_no_pos = [0, 0]
        prediction_pos = [1, 0]
        metrics2 = calculate_classification_metrics(truth_no_pos, prediction_pos, zero_division=0.0)
        self.assertEqual(metrics2["recall"], 0.0)

    def test_image_fixture_classification_and_metrics(self):
        y_true = [r["label"] for r in self.records]
        y_pred = [classify_diagonal_image(r["image"]) for r in self.records]
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
