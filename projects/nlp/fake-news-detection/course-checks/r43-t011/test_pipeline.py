
import json
import random
import unittest
from pathlib import Path

from pipeline import validate_records


HERE = Path(__file__).parent


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(HERE / "fixture.json", encoding="utf-8") as file:
            cls.fixture = json.load(file)["records"]

    def test_six_synthetic_records_pass(self):
        self.assertEqual(len(self.fixture), 6)
        self.assertEqual(
            [record["record_id"] for record in self.fixture],
            ["t1", "t2", "t3", "t4", "t5", "t6"]
        )
        self.assertEqual(validate_records(self.fixture), [])

    def test_missing_record_id_fails(self):
        record = {"text": "red apple", "label": 1}
        errors = validate_records([record])
        self.assertTrue(any(
            error["field"] == "record_id"
            and "missing" in error["reason"]
            for error in errors
        ))

    def test_duplicate_record_id_fails(self):
        records = [
            {"record_id": "t1", "text": "red apple", "label": 1},
            {"record_id": "t1", "text": "green apple", "label": 1}
        ]
        errors = validate_records(records)
        self.assertTrue(any(
            "Duplicate" in error["reason"] for error in errors
        ))

    def test_empty_text_fails(self):
        record = {"record_id": "t7", "text": "  ", "label": 1}
        errors = validate_records([record])
        self.assertTrue(any(
            error["field"] == "text" for error in errors
        ))

    def test_numeric_text_fails(self):
        record = {"record_id": "t7", "text": 123, "label": 1}
        errors = validate_records([record])
        self.assertTrue(any(
            error["field"] == "text" for error in errors
        ))

    def test_invalid_label_types_fail(self):
        for label in ("1", 1.0, True, 2):
            with self.subTest(label=label):
                record = {
                    "record_id": "t7",
                    "text": "red apple",
                    "label": label
                }
                errors = validate_records([record])
                self.assertTrue(any(
                    error["field"] == "label" for error in errors
                ))

    def test_missing_text_and_label_fail(self):
        errors = validate_records([{"record_id": "t7"}])
        fields = {error["field"] for error in errors}
        self.assertIn("text", fields)
        self.assertIn("label", fields)

    def test_malformed_record_fails(self):
        errors = validate_records([["t7", "red apple", 1]])
        self.assertTrue(any(
            error["field"] == "record" for error in errors
        ))

    def test_tensor_like_input_fails(self):
        class FakeTensor:
            pass

        errors = validate_records([
            {"record_id": "t7", "text": FakeTensor(), "label": 1}
        ])
        self.assertTrue(any(
            error["field"] == "text" for error in errors
        ))

    def test_unexpected_field_fails(self):
        record = {
            "record_id": "t7",
            "text": "red apple",
            "label": 1,
            "extra": "not allowed"
        }
        errors = validate_records([record])
        self.assertTrue(any(
            error["field"] == "extra" for error in errors
        ))

    def test_invalid_top_level_input_fails(self):
        errors = validate_records({"record_id": "t1"})
        self.assertTrue(any(
            error["field"] == "records" for error in errors
        ))

    def test_fixed_seed_keeps_validation_deterministic(self):
        random.seed(42)
        first = validate_records(self.fixture)

        random.seed(42)
        second = validate_records(self.fixture)

        self.assertEqual(first, second)


class TestDataSplits(unittest.TestCase):

    def test_splits_are_deterministic_and_disjoint(self):
        from pipeline import split_records

        with open(HERE / "fixture.json", encoding="utf-8") as file:
            records = json.load(file)["records"]

        first = split_records(records, seed=42)
        second = split_records(records, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {"train", "validation", "test"}
        )

        train_ids = {r["record_id"] for r in first["train"]}
        validation_ids = {r["record_id"] for r in first["validation"]}
        test_ids = {r["record_id"] for r in first["test"]}

        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(validation_ids & test_ids)

        all_ids = train_ids | validation_ids | test_ids
        self.assertEqual(all_ids, {f"t{i}" for i in range(1, 7)})

    def test_three_records_produce_non_empty_splits(self):
        from pipeline import split_records

        records = [
            {"record_id": "a", "text": "first", "label": 1},
            {"record_id": "b", "text": "second", "label": 0},
            {"record_id": "c", "text": "third", "label": 1},
        ]

        result = split_records(records, seed=42)

        self.assertTrue(result["train"])
        self.assertTrue(result["validation"])
        self.assertTrue(result["test"])


if __name__ == "__main__":
    unittest.main()
