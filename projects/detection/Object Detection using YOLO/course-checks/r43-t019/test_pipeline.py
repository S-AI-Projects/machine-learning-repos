import copy, json, unittest
from pathlib import Path
from pipeline import predict_by_id, validate_fixture
HERE = Path(__file__).parent
class RecordOrderRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
        self.expected = json.loads((HERE / "expected_predictions.json").read_text(encoding="utf-8"))
    def test_fixture_schema_and_stratified_splits(self):
        rows = validate_fixture(self.fixture)
        self.assertEqual([r["x1"] for r in rows], list(range(8)))
        self.assertEqual([r["label"] for r in rows], [0,0,0,0,1,1,1,1])
        self.assertIn("synthetic", self.fixture["data_status"])
        for ids in self.fixture["split_ids"].values():
            self.assertEqual({next(r["label"] for r in rows if r["record_id"] == rid) for rid in ids}, {0,1})
    def test_record_order_permutation_preserves_predictions_by_id(self):
        rows = validate_fixture(self.fixture)
        permuted = list(reversed(rows))
        self.assertEqual(predict_by_id(rows), self.expected)
        self.assertEqual(predict_by_id(permuted), self.expected)
    def test_missing_feature_is_rejected(self):
        rows = copy.deepcopy(self.fixture["records"]); del rows[0]["x2"]
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_by_id(rows)
    def test_unseen_field_is_rejected(self):
        rows = copy.deepcopy(self.fixture["records"]); rows[0]["token"] = "unseen"
        with self.assertRaisesRegex(ValueError, "fields"):
            predict_by_id(rows)
    def test_duplicate_stable_id_is_rejected(self):
        rows = copy.deepcopy(self.fixture["records"]); rows[1]["record_id"] = rows[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            predict_by_id(rows)
    def test_group_contamination_is_rejected(self):
        fixture = copy.deepcopy(self.fixture); fixture["records"][1]["group_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_fixture(fixture)
if __name__ == "__main__": unittest.main()
