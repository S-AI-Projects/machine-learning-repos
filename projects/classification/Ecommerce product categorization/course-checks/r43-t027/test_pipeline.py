import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import (
    ReproduciblePipeline,
    create_default_config,
    validate_config,
    ConfigurationError
)


class TestReproducibleConfiguration(unittest.TestCase):
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

    def test_serialization_roundtrip_reconstructs_identical_predictions(self):
        pipeline = ReproduciblePipeline()
        serialized = pipeline.serialize_config()

        reloaded_pipeline = ReproduciblePipeline.from_serialized_config(serialized)
        
        preds_original = pipeline.predict(self.records)
        preds_reloaded = reloaded_pipeline.predict(self.records)

        self.assertEqual(preds_original, preds_reloaded)
        self.assertEqual(pipeline.config["split_ids"], reloaded_pipeline.config["split_ids"])
        self.assertEqual(preds_original, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_reject_unknown_configuration_field(self):
        cfg = create_default_config()
        cfg["unexpected_hyperparameter"] = 12345
        with self.assertRaises(ConfigurationError) as ctx:
            validate_config(cfg)
        self.assertIn("unexpected_hyperparameter", str(ctx.exception))

    def test_reject_contradictory_split_settings(self):
        cfg = create_default_config()
        # Overlapping split
        cfg["split_ids"]["train"].append("rec_2")  # rec_2 is already in test!
        with self.assertRaises(ConfigurationError) as ctx:
            validate_config(cfg)
        self.assertIn("overlap", str(ctx.exception))

    def test_reject_incompatible_input_shape(self):
        pipeline = ReproduciblePipeline()
        # Incompatible record missing 'x2'
        incompatible_records = [{"record_id": "err_1", "x1": 2.0}]
        with self.assertRaises(ConfigurationError) as ctx:
            pipeline.predict(incompatible_records)
        self.assertIn("x2", str(ctx.exception))

    def test_environment_versions_recorded(self):
        cfg = create_default_config()
        self.assertIn("python_version", cfg["environment_metadata"])
        self.assertIn("platform", cfg["environment_metadata"])
        self.assertTrue(len(cfg["environment_metadata"]["python_version"]) > 0)


if __name__ == "__main__":
    unittest.main()
