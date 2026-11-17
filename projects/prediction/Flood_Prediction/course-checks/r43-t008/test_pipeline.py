"""
Unit tests for R43-T008 error-analysis report.
NOTE: SYNTHETIC FIXTURE only. NOT real flood data. Pipeline verification only.
"""
import unittest
from pipeline import (
    FIXTURE_RECORDS, build_error_report, aggregate_report,
    inject_wrong_prediction, TEST_IDS, classify_error
)


def get_split(ids):
    id_map = {r["record_id"]: r for r in FIXTURE_RECORDS}
    return [id_map[i] for i in ids]


class TestClassifyError(unittest.TestCase):
    def test_correct(self):
        self.assertEqual(classify_error(1, 1), "correct")
        self.assertEqual(classify_error(0, 0), "correct")

    def test_false_positive(self):
        self.assertEqual(classify_error(0, 1), "false_positive")

    def test_false_negative(self):
        self.assertEqual(classify_error(1, 0), "false_negative")


class TestErrorReport(unittest.TestCase):
    def test_injected_wrong_appears_in_report(self):
        test_recs = get_split(TEST_IDS)
        preds = inject_wrong_prediction(test_recs, "f6", 0)
        report = build_error_report(test_recs, preds)
        f6_rows = [row for row in report if row["record_id"] == "f6"]
        self.assertEqual(len(f6_rows), 1)
        self.assertNotEqual(f6_rows[0]["expected"], f6_rows[0]["predicted"])

    def test_correct_prediction_not_in_errors(self):
        test_recs = get_split(TEST_IDS)
        preds = [r["label"] for r in test_recs]
        report = build_error_report(test_recs, preds)
        errors = [row for row in report if row["reason_category"] != "correct"]
        self.assertEqual(errors, [])

    def test_aggregate_matches_detail(self):
        test_recs = get_split(TEST_IDS)
        preds = inject_wrong_prediction(test_recs, "f6", 0)
        report = build_error_report(test_recs, preds)
        agg = aggregate_report(report)
        detail_errors = sum(1 for row in report if row["reason_category"] != "correct")
        total_errors = agg["false_positive"] + agg["false_negative"]
        self.assertEqual(detail_errors, total_errors)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_error_report(get_split(TEST_IDS), [0])


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        self.assertNotEqual(classify_error(1, 0), "correct")


if __name__ == "__main__":
    unittest.main()
