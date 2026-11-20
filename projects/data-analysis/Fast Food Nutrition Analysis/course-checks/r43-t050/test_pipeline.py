import json
import os
import sys
import tempfile
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import SavedClassificationPipeline, ModelInterfaceError, load_fixture


class TestSavedPipelineInterface(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]
        self.train_ids = set(self.fixture["metadata"]["split_policy"]["train_record_ids"])
        self.test_ids = set(self.fixture["metadata"]["split_policy"]["test_record_ids"])

        self.train_records = [r for r in self.records if r["record_id"] in self.train_ids]
        self.test_records = [r for r in self.records if r["record_id"] in self.test_ids]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_vals, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_save_and_reload_predictions_agree_within_tolerance(self):
        pipe_fitted = SavedClassificationPipeline().fit(self.train_records)
        preds_before = pipe_fitted.predict(self.test_records)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pipe_fitted.save_parameters_json(tmp_path)
            pipe_reloaded = SavedClassificationPipeline.load_parameters_json(tmp_path)
            preds_after = pipe_reloaded.predict(self.test_records)

            self.assertEqual(preds_before, preds_after)
            self.assertEqual(preds_after, [0, 0, 1, 1])
            self.assertEqual(pipe_fitted.weights, pipe_reloaded.weights)
            self.assertAlmostEqual(pipe_fitted.bias, pipe_reloaded.bias, delta=1e-6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_schema_mismatch_fails_clearly(self):
        pipe = SavedClassificationPipeline(weights=[1.0, 0.0], bias=-3.5, threshold=0.0)
        # Missing required field x1
        bad_record = {"record_id": "rec_bad", "x2": 1.0}
        with self.assertRaises(ModelInterfaceError) as ctx:
            pipe.predict([bad_record])
        self.assertIn("missing required field 'x1'", str(ctx.exception))

    def test_version_mismatch_fails_clearly(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump({
                "interface_version": "99.0.0",  # Incompatible version
                "weights": [1.0, 0.0],
                "bias": 0.0,
                "threshold": 0.0
            }, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ModelInterfaceError) as ctx:
                SavedClassificationPipeline.load_parameters_json(tmp_path)
            self.assertIn("Version mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_no_training_data_needed_at_inference(self):
        # Inference pipeline instantiated purely from parameters, zero training data
        inference_pipe = SavedClassificationPipeline(weights=[1.0, 0.0], bias=-3.5, threshold=0.0)
        test_sample = {"record_id": "r_eval", "x1": 5.0, "x2": 1.0}
        pred = inference_pipe.predict_single(test_sample)
        self.assertEqual(pred, 1)

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
