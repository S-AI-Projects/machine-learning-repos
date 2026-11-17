import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import validate_schema



class TestGradePredictionSchemaValidator(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.valid_records = self.fixture_data["records"]

    def test_valid_fixture_passes(self):
        result = validate_schema(self.valid_records)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.valid_records), 8)

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.valid_records), 8)
        x1_values = [r["x1"] for r in self.valid_records]
        x2_values = [r["x2"] for r in self.valid_records]
        labels = [r["label"] for r in self.valid_records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_missing_record_id_fails(self):
        bad_records = [
            {"group_id": "grp_0", "x1": 1.0, "x2": 1.0, "label": 0}
        ]
        result = validate_schema(bad_records)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e["field"] == "record_id" and "Missing" in e["reason"] for e in result.errors))

    def test_duplicate_record_id_fails(self):
        bad_records = [
            {"record_id": "rec_0", "group_id": "grp_0", "x1": 1.0, "x2": 1.0, "label": 0},
            {"record_id": "rec_0", "group_id": "grp_0", "x1": 2.0, "x2": 1.0, "label": 0}
        ]
        result = validate_schema(bad_records)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e["field"] == "record_id" and "Duplicate" in e["reason"] for e in result.errors))

    def test_malformed_numeric_and_label_fails(self):
        bad_records = [
            {"record_id": "bad_1", "group_id": "grp_0", "x1": "invalid_number", "x2": 1.0, "label": 0},
            {"record_id": "bad_2", "group_id": "grp_0", "x1": 1.0, "x2": None, "label": 0},
            {"record_id": "bad_3", "group_id": "grp_0", "x1": 1.0, "x2": 1.0, "label": 99}
        ]
        result = validate_schema(bad_records)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 3)
        self.assertEqual(result.errors[0]["field"], "x1")
        self.assertEqual(result.errors[1]["field"], "x2")
        self.assertEqual(result.errors[2]["field"], "label")

    def test_non_silent_reporting(self):
        # Multiple failures in one record and across records
        mixed = [
            {"record_id": "rec_ok", "group_id": "grp_0", "x1": 0.0, "x2": 1.0, "label": 0},
            {"record_id": "rec_bad", "group_id": "", "x1": "bad", "x2": 1.0, "label": 3}
        ]
        result = validate_schema(mixed)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.valid_records), 1)
        self.assertEqual(len(result.errors), 3)  # group_id, x1, and label failed
        failed_fields = {e["field"] for e in result.errors}
        self.assertEqual(failed_fields, {"group_id", "x1", "label"})


if __name__ == "__main__":
    unittest.main()
