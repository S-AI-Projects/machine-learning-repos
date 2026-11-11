import copy, json, unittest
from pathlib import Path
from pipeline import analyze, validate_fixture
HERE = Path(__file__).parent
class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
        self.predictions = json.loads((HERE / "predictions.json").read_text(encoding="utf-8"))
    def test_injected_wrong_prediction_has_expected_report_row(self):
        result = analyze(self.fixture, self.predictions)
        self.assertEqual(result["errors"], [{"record_id":"r6", "expected":1, "predicted":0, "reason_category":"threshold_boundary_case"}])
    def test_correct_prediction_does_not_appear_in_error_report(self):
        result = analyze(self.fixture, self.predictions)
        self.assertNotIn("r5", {row["record_id"] for row in result["errors"]})
    def test_aggregate_and_split_counts_match_detail(self):
        result = analyze(self.fixture, self.predictions)
        self.assertEqual(result["counts"], {"total":8, "correct":7, "errors":1})
        self.assertEqual(sum(v["errors"] for v in result["by_split"].values()), len(result["errors"]))
        self.assertEqual(result["by_split"]["validation"]["errors"], 1)
    def test_fixture_is_fixed_stratified_and_synthetic(self):
        rows = validate_fixture(self.fixture)
        self.assertEqual([r["x1"] for r in rows], list(range(8)))
        self.assertEqual([r["label"] for r in rows], [0,0,0,0,1,1,1,1])
        self.assertIn("synthetic", self.fixture["data_status"])
        for ids in self.fixture["split_ids"].values():
            self.assertEqual({next(r["label"] for r in rows if r["record_id"] == rid) for rid in ids}, {0,1})
    def test_missing_prediction_is_rejected(self):
        data = copy.deepcopy(self.predictions); del data["predictions"]["r8"]
        with self.assertRaisesRegex(ValueError, "cover every"):
            analyze(self.fixture, data)
    def test_wrong_prediction_without_reason_is_rejected(self):
        data = copy.deepcopy(self.predictions); data["reason_categories"] = {}
        with self.assertRaisesRegex(ValueError, "reason category"):
            analyze(self.fixture, data)
    def test_group_contamination_is_rejected(self):
        fixture = copy.deepcopy(self.fixture); fixture["records"][1]["group_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            analyze(fixture, self.predictions)
    def test_unknown_prediction_id_is_rejected(self):
        data = copy.deepcopy(self.predictions); data["predictions"]["unknown"] = 0
        with self.assertRaisesRegex(ValueError, "cover every"):
            analyze(self.fixture, data)
if __name__ == "__main__": unittest.main()
