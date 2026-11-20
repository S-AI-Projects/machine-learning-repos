import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import ExplicitNumericPreprocessor, PreprocessingContaminationError, load_fixture


class TestPreprocessingContract(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]
        self.train_ids = set(self.fixture["metadata"]["split_policy"]["train_record_ids"])
        self.test_ids = set(self.fixture["metadata"]["split_policy"]["test_record_ids"])

        self.train_records = [r for r in self.records if r["record_id"] in self.train_ids]
        self.test_records = [r for r in self.records if r["record_id"] in self.test_ids]

    def test_fixture_dimensions_and_features(self):
        self.assertEqual(len(self.records), 8)
        self.assertEqual(self.fixture["metadata"]["feature_order"], ["x1", "x2"])
        self.assertEqual(self.fixture["metadata"]["label_set"], [0, 1])

    def test_preprocessor_fitted_strictly_on_train(self):
        preprocessor = ExplicitNumericPreprocessor()
        # Train records have x1=[0.0, 1.0, 4.0, 5.0], x2=[1.0, 1.0, 1.0, 1.0]
        # min(x1)=0.0, max(x1)=5.0, scale=5.0
        preprocessor.fit(self.train_records)
        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(preprocessor.params["x1"]["min"], 0.0)
        self.assertEqual(preprocessor.params["x1"]["max"], 5.0)
        self.assertEqual(preprocessor.params["x1"]["scale"], 5.0)

        # Transform train records
        train_transformed = preprocessor.transform(self.train_records)
        self.assertEqual(len(train_transformed), 4)
        self.assertEqual(train_transformed[0], [0.0, 0.0])  # (0.0-0.0)/5.0, x2 constant -> 0.0
        self.assertEqual(train_transformed[3], [1.0, 0.0])  # (5.0-0.0)/5.0

    def test_transform_held_out_cannot_alter_parameters(self):
        preprocessor = ExplicitNumericPreprocessor()
        preprocessor.fit(self.train_records)
        params_before = dict(preprocessor.params["x1"])

        # Test records have x1=[2.0, 3.0, 6.0, 7.0] - contains values > max train value (7.0 > 5.0)
        test_transformed = preprocessor.transform(self.test_records)
        params_after = dict(preprocessor.params["x1"])

        self.assertEqual(params_before, params_after)
        # Scaled values on test will be (x1 - 0.0) / 5.0 -> r7: 6.0/5.0 = 1.2, r8: 7.0/5.0 = 1.4
        self.assertEqual(len(test_transformed), 4)
        self.assertAlmostEqual(test_transformed[2][0], 1.2)
        self.assertAlmostEqual(test_transformed[3][0], 1.4)

    def test_unfitted_transform_raises(self):
        preprocessor = ExplicitNumericPreprocessor()
        with self.assertRaises(RuntimeError):
            preprocessor.transform(self.test_records)

    def test_deliberate_contamination_fixture_rejected(self):
        preprocessor = ExplicitNumericPreprocessor()
        # 1. Missing feature rejected
        bad_record_missing = [{"record_id": "bad1", "x1": 1.0}]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.fit(bad_record_missing)

        # 2. Non-numeric type rejected
        bad_record_str = [{"record_id": "bad2", "x1": "invalid", "x2": 1.0, "label": 0}]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.fit(bad_record_str)

        # 3. Invalid label rejected
        bad_label = [{"record_id": "bad3", "x1": 1.0, "x2": 1.0, "label": 99}]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.fit(bad_label)

    def test_synthetic_verification_label(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
