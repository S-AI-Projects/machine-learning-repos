import copy, json, unittest
from pathlib import Path
from pipeline import load_json, run, validate_config, validate_fixture

HERE = Path(__file__).parent
class ReproducibleConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = load_json(HERE / "fixture.json")
        self.config = load_json(HERE / "config.json")
    def test_fixture_schema_and_fixed_stratified_splits(self):
        rows = validate_fixture(self.fixture)
        self.assertEqual(len(rows), 8)
        self.assertEqual([r["x1"] for r in rows], list(range(8)))
        self.assertEqual([r["label"] for r in rows], [0,0,0,0,1,1,1,1])
        self.assertEqual(len(set(r["group_id"] for r in rows)), 8)
        for ids in self.config["split_ids"].values():
            self.assertEqual({next(r["label"] for r in rows if r["record_id"] == rid) for rid in ids}, {0,1})
    def test_reload_reconstructs_splits_and_predictions(self):
        expected = {"split_ids": self.config["split_ids"], "predictions": {f"r{i+1}": int(i >= 4) for i in range(8)}}
        self.assertEqual(run(self.fixture, self.config), expected)
        self.assertEqual(run(json.loads(json.dumps(self.fixture)), json.loads(json.dumps(self.config))), expected)
    def test_unknown_configuration_field_rejected(self):
        cfg = copy.deepcopy(self.config); cfg["mystery"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_config(cfg, self.fixture["records"])
    def test_incompatible_feature_order_rejected(self):
        cfg = copy.deepcopy(self.config); cfg["feature_order"] = ["x2", "x1"]
        with self.assertRaisesRegex(ValueError, "feature"):
            validate_config(cfg, self.fixture["records"])
    def test_incompatible_input_shape_rejected(self):
        rows = copy.deepcopy(self.fixture["records"]); rows[0]["x3"] = rows[0].pop("x2")
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_fixture({**self.fixture, "records": rows})
    def test_missing_or_crossed_split_ids_rejected(self):
        cfg = copy.deepcopy(self.config); cfg["split_ids"]["test"].remove("r8")
        with self.assertRaisesRegex(ValueError, "partition"):
            validate_config(cfg, self.fixture["records"])
    def test_contradictory_algorithm_labels_rejected(self):
        cfg = copy.deepcopy(self.config); cfg["algorithm"]["negative_label"] = 1
        with self.assertRaisesRegex(ValueError, "contradictory"):
            validate_config(cfg, self.fixture["records"])

if __name__ == "__main__": unittest.main()
