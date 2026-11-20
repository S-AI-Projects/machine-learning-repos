import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import generate_error_analysis_report, load_fixture


class TestErrorAnalysisReport(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]
        self.injected = self.fixture["metadata"]["injected_error"]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_vals, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_injected_wrong_prediction_appears_in_report(self):
        report = generate_error_analysis_report(self.records)

        # Injected error r5 must appear
        error_record_ids = [row["record_id"] for row in report.error_rows]
        self.assertIn(self.injected["record_id"], error_record_ids)

        # Match exact details of injected failure
        r5_row = next(row for row in report.error_rows if row["record_id"] == "r5")
        self.assertEqual(r5_row["expected"], self.injected["expected"])
        self.assertEqual(r5_row["predicted"], self.injected["predicted"])
        self.assertEqual(r5_row["reason_category"], self.injected["reason_category"])

    def test_correct_predictions_do_not_appear_in_report(self):
        report = generate_error_analysis_report(self.records)
        error_record_ids = set(row["record_id"] for row in report.error_rows)

        # Records r1, r2, r3, r4, r6, r7, r8 are correct and MUST NOT be in error_rows
        correct_ids = {"r1", "r2", "r3", "r4", "r6", "r7", "r8"}
        for cid in correct_ids:
            self.assertNotIn(cid, error_record_ids)

    def test_aggregate_error_counts_match_detailed_rows(self):
        report = generate_error_analysis_report(self.records)

        self.assertEqual(report.total_samples, 8)
        self.assertEqual(report.error_count, 1)
        self.assertEqual(report.correct_count, 7)
        self.assertEqual(len(report.error_rows), report.error_count)
        self.assertEqual(sum(report.reason_summary.values()), report.error_count)
        self.assertEqual(report.total_samples, report.correct_count + report.error_count)

    def test_multiple_error_categories(self):
        # Test multiple errors (both FP and FN)
        custom_preds = [0, 1, 0, 0, 0, 1, 1, 1]  # r2 false positive (0->1), r5 false negative (1->0)
        report = generate_error_analysis_report(self.records, predictions=custom_preds)

        self.assertEqual(report.error_count, 2)
        self.assertEqual(len(report.error_rows), 2)
        err_ids = {row["record_id"] for row in report.error_rows}
        self.assertEqual(err_ids, {"r2", "r5"})
        self.assertTrue(any("False Positive" in k for k in report.reason_summary))
        self.assertTrue(any("False Negative" in k for k in report.reason_summary))

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
