import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import ExplicitNumericPreprocessor, PreprocessingContaminationError


class TestExplicitPreprocessing(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]
        split_meta = self.fixture_data["split_metadata"]["stratified_split"]
        self.train_ids = set(split_meta["train"])
        self.test_ids = set(split_meta["test"])

        self.train_records = [r for r in self.records if r["record_id"] in self.train_ids]
        self.test_records = [r for r in self.records if r["record_id"] in self.test_ids]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_fit_on_train_only(self):
        preprocessor = ExplicitNumericPreprocessor()
        preprocessor.fit(self.train_records)

        # Train records have x1 in [0.0, 1.0, 4.0, 5.0]
        # Min = 0.0, Max = 5.0, scale = 5.0
        self.assertEqual(preprocessor.params["x1"]["min"], 0.0)
        self.assertEqual(preprocessor.params["x1"]["max"], 5.0)
        self.assertEqual(preprocessor.params["x1"]["scale"], 5.0)
        
        # x2 has min = 1.0, max = 1.0, scale = 1.0
        self.assertEqual(preprocessor.params["x2"]["min"], 1.0)
        self.assertEqual(preprocessor.params["x2"]["max"], 1.0)

    def test_output_values_dtype_and_shape(self):
        preprocessor = ExplicitNumericPreprocessor()
        preprocessor.fit(self.train_records)
        train_transformed = preprocessor.transform(self.train_records)

        self.assertEqual(len(train_transformed), len(self.train_records))
        for row in train_transformed:
            self.assertEqual(len(row), 2)
            self.assertIsInstance(row[0], float)
            self.assertIsInstance(row[1], float)

        # First train record (rec_0): x1=0.0 -> (0-0)/5 = 0.0, x2=1.0 -> (1-1)/1 = 0.0
        self.assertEqual(train_transformed[0], [0.0, 0.0])
        # Fourth train record (rec_5): x1=5.0 -> (5-0)/5 = 1.0, x2=1.0 -> 0.0
        self.assertEqual(train_transformed[3], [1.0, 0.0])

    def test_fitting_on_held_out_is_impossible_via_transform(self):
        preprocessor = ExplicitNumericPreprocessor()
        preprocessor.fit(self.train_records)

        initial_params = dict(preprocessor.params["x1"])
        
        # Transform held-out test records (where x1 values are 2.0, 3.0, 6.0, 7.0)
        test_transformed = preprocessor.transform(self.test_records)
        
        # Parameters must remain strictly unchanged!
        self.assertEqual(preprocessor.params["x1"], initial_params)
        
        # In test records, rec_7 has x1=7.0 -> scaled value is (7-0)/5 = 1.4 > 1.0 (out of train range)
        # verifying that training-derived scaling was preserved
        rec_7_transformed = test_transformed[3]
        self.assertAlmostEqual(rec_7_transformed[0], 1.4, places=6)

    def test_contamination_rejections(self):
        preprocessor = ExplicitNumericPreprocessor()
        preprocessor.fit(self.train_records)

        # 1. Missing feature in input
        missing_feat_record = [{"record_id": "bad_1", "group_id": "grp_0", "x1": 1.0}]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.transform(missing_feat_record)

        # 2. Malformed numeric feature
        malformed_record = [{"record_id": "bad_2", "group_id": "grp_0", "x1": "text", "x2": 1.0}]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.transform(malformed_record)

        # 3. Invalid label during fit
        invalid_label_record = [{"record_id": "bad_3", "group_id": "grp_0", "x1": 1.0, "x2": 1.0, "label": 999}]
        with self.assertRaises(PreprocessingContaminationError):
            ExplicitNumericPreprocessor().fit(invalid_label_record)


if __name__ == "__main__":
    unittest.main()
