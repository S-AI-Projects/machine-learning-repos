import json
import os
import sys
import tempfile
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import SavedClassificationPipeline, ModelInterfaceError


class TestSavedPipelineInterface(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]
        split_meta = self.fixture_data["split_metadata"]["stratified_split"]
        self.train_records = [r for r in self.records if r["record_id"] in set(split_meta["train"])]
        self.test_records = [r for r in self.records if r["record_id"] in set(split_meta["test"])]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_save_and_reload_predictions_agree_within_tolerance(self):
        # 1. Fit initial pipeline
        pipeline = SavedClassificationPipeline()
        pipeline.fit(self.train_records)
        preds_before = pipeline.predict(self.test_records)

        # 2. Save parameters to temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pipeline.save_parameters_json(tmp_path)

            # 3. Reload into completely separate instance without training data
            reloaded_pipeline = SavedClassificationPipeline.load_parameters_json(tmp_path)
            preds_after = reloaded_pipeline.predict(self.test_records)

            # 4. Assert predictions before and after match exactly
            self.assertEqual(preds_before, preds_after)
            for pb, pa in zip(preds_before, preds_after):
                self.assertAlmostEqual(pb, pa, delta=1e-6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_schema_and_version_mismatch_fails_clearly(self):
        # Corrupted version file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump({
                "interface_version": "9.9.9",
                "weights": [1.0, 0.0],
                "bias": 0.0,
                "threshold": 0.5
            }, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ModelInterfaceError) as ctx:
                SavedClassificationPipeline.load_parameters_json(tmp_path)
            self.assertIn("Version mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_inference_schema_validation(self):
        pipeline = SavedClassificationPipeline(weights=[1.0, 0.0], bias=-3.5, threshold=0.0)

        # Missing required field 'x2'
        bad_record = {"record_id": "r_err", "x1": 5.0}
        with self.assertRaises(ModelInterfaceError) as ctx:
            pipeline.predict_single(bad_record)
        self.assertIn("x2", str(ctx.exception))

        # Missing record_id
        bad_id_record = {"x1": 5.0, "x2": 1.0}
        with self.assertRaises(ModelInterfaceError) as ctx:
            pipeline.predict_single(bad_id_record)
        self.assertIn("record_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
