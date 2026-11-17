import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import generate_error_analysis_report


class TestErrorAnalysisMonitoring(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_injected_error_appears_in_report(self):
        report = generate_error_analysis_report(self.records)

        # Injected wrong prediction is at rec_6 (expected 1, predicted 0)
        self.assertEqual(report.error_count, 1)
        self.assertEqual(report.total_samples, 8)
        self.assertEqual(report.correct_count, 7)

        # Error row verification
        error_row = report.error_rows[0]
        self.assertEqual(error_row["record_id"], "rec_6")
        self.assertEqual(error_row["expected"], 1)
        self.assertEqual(error_row["predicted"], 0)
        self.assertEqual(error_row["reason_category"], "False Negative: Underprediction near decision boundary")

    def test_correct_predictions_excluded_from_error_rows(self):
        report = generate_error_analysis_report(self.records)
        error_record_ids = [row["record_id"] for row in report.error_rows]
        
        # Verify that all 7 correctly predicted rows do NOT appear in error rows
        correct_ids = ["rec_0", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5", "rec_7"]
        for cid in correct_ids:
            self.assertNotIn(cid, error_record_ids)

    def test_aggregate_error_counts_match_detailed_rows(self):
        report = generate_error_analysis_report(self.records)
        
        # Aggregate count matches detailed rows count
        self.assertEqual(report.error_count, len(report.error_rows))
        self.assertEqual(sum(report.reason_summary.values()), len(report.error_rows))
        self.assertEqual(report.correct_count + report.error_count, report.total_samples)

    def test_clean_predictions_have_zero_errors(self):
        # Perfect predictions: y_pred == y_true
        perfect_predictions = [r["label"] for r in self.records]
        report = generate_error_analysis_report(self.records, predictions=perfect_predictions)
        self.assertEqual(report.error_count, 0)
        self.assertEqual(len(report.error_rows), 0)
        self.assertEqual(report.correct_count, 8)


if __name__ == "__main__":
    unittest.main()
