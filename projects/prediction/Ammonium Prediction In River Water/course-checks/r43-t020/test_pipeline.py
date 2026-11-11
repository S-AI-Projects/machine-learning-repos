import copy, json, tempfile, unittest
from pathlib import Path
from pipeline import load_parameters, predict, save_parameters, validate_fixture

HERE = Path(__file__).parent

class SavedPipelineInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
        self.parameters = json.loads((HERE / "parameters.json").read_text(encoding="utf-8"))
        self.expected = json.loads((HERE / "expected_predictions.json").read_text(encoding="utf-8"))

    def test_fixture_has_eight_synthetic_rows_and_fixed_splits(self):
        rows = validate_fixture(self.fixture)
        self.assertEqual([r["x1"] for r in rows], list(range(8)))
        self.assertEqual([r["label"] for r in rows], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertIn("synthetic", self.fixture["data_status"])
        self.assertEqual({k: len(v) for k, v in self.fixture["split_ids"].items()}, {"train": 4, "validation": 2, "test": 2})

    def test_json_save_reload_preserves_predictions_within_tolerance(self):
        rows = validate_fixture(self.fixture)
        before = predict(rows, self.parameters)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parameters.json"
            save_parameters(self.parameters, path)
            restored = load_parameters(path)
        after = predict(rows, restored)
        self.assertEqual(before.keys(), after.keys())
        for rid in before:
            self.assertAlmostEqual(before[rid]["probability"], after[rid]["probability"], delta=1e-12)
            self.assertEqual(before[rid]["label"], after[rid]["label"])
            self.assertAlmostEqual(after[rid]["probability"], self.expected[rid]["probability"], delta=1e-12)
            self.assertEqual(after[rid]["label"], self.expected[rid]["label"])

    def test_inference_needs_only_records_and_saved_parameters(self):
        self.assertEqual(predict(self.fixture["records"], self.parameters), self.expected)

    def test_input_schema_mismatch_is_rejected(self):
        rows = copy.deepcopy(self.fixture["records"]); del rows[0]["x2"]
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            predict(rows, self.parameters)

    def test_duplicate_record_id_is_rejected(self):
        rows = copy.deepcopy(self.fixture["records"]); rows[1]["record_id"] = rows[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            predict(rows, self.parameters)

    def test_saved_parameter_schema_and_version_mismatches_fail(self):
        bad = copy.deepcopy(self.parameters); bad["schema_version"] = 99
        with self.assertRaisesRegex(ValueError, "schema/version"):
            predict(self.fixture["records"], bad)
        bad = copy.deepcopy(self.parameters); bad["feature_order"] = ["x2", "x1"]
        with self.assertRaisesRegex(ValueError, "schema/version"):
            predict(self.fixture["records"], bad)

    def test_split_group_contamination_is_rejected(self):
        bad = copy.deepcopy(self.fixture); bad["records"][1]["group_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_fixture(bad)

if __name__ == "__main__": unittest.main()
