import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import ReproduciblePipeline, ConfigurationError, validate_config, load_fixture


class TestReproducibleConfiguration(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]
        self.config = self.fixture["metadata"]["config"]

    def test_reloaded_config_reconstructs_identical_splits_and_predictions(self):
        pipe1 = ReproduciblePipeline(self.config)
        serialized = pipe1.serialize_config()

        pipe2 = ReproduciblePipeline.from_serialized_config(serialized)
        self.assertEqual(pipe1.config["split_ids"], pipe2.config["split_ids"])
        self.assertEqual(pipe1.config["seed"], pipe2.config["seed"])
        self.assertEqual(pipe1.config["algorithm_settings"], pipe2.config["algorithm_settings"])

        preds1 = pipe1.predict(self.records)
        preds2 = pipe2.predict(self.records)
        self.assertEqual(preds1, preds2)
        self.assertEqual(preds1, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_unknown_field_rejected(self):
        bad_config = dict(self.config)
        bad_config["unsupported_custom_key"] = "leak_param"
        with self.assertRaises(ConfigurationError) as ctx:
            ReproduciblePipeline(bad_config)
        self.assertIn("Unknown configuration field", str(ctx.exception))

    def test_contradictory_settings_rejected(self):
        # 1. Overlapping train/test splits
        bad_splits = dict(self.config)
        bad_splits["split_ids"] = {
            "train": ["r1", "r2", "r3"],
            "test": ["r3", "r4"]  # r3 in both
        }
        with self.assertRaises(ConfigurationError) as ctx:
            validate_config(bad_splits)
        self.assertIn("Contradictory split_ids", str(ctx.exception))

        # 2. Negative seed
        bad_seed = dict(self.config)
        bad_seed["seed"] = -1
        with self.assertRaises(ConfigurationError):
            validate_config(bad_seed)

        # 3. Unknown algorithm
        bad_algo = dict(self.config)
        bad_algo["algorithm_settings"] = {"algorithm": "unsupported_magic_model"}
        with self.assertRaises(ConfigurationError):
            validate_config(bad_algo)

    def test_incompatible_input_shape_rejected(self):
        pipe = ReproduciblePipeline(self.config)
        # Missing required feature x2
        bad_input = [{"record_id": "r_bad", "x1": 2.0}]
        with self.assertRaises(ConfigurationError) as ctx:
            pipe.predict(bad_input)
        self.assertIn("Incompatible input shape", str(ctx.exception))

    def test_environment_versions_recorded(self):
        env_meta = self.config.get("environment_metadata", {})
        self.assertIn("python_version", env_meta)
        self.assertIn("platform", env_meta)

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
