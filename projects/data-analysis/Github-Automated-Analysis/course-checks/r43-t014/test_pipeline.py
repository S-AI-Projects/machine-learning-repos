import json
from pathlib import Path
import unittest

from pipeline import binary_precision_recall, validate_stratified_splits


FIXTURE = Path(__file__).with_name("fixture.json")


class MetricCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.records = cls.fixture["records"]
        cls.splits = cls.fixture["splits"]

    def test_required_binary_precision_recall_example(self):
        result = binary_precision_recall([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result, {"precision": 0.5, "recall": 0.5})

    def test_zero_denominators_default_to_zero(self):
        result = binary_precision_recall([0, 0], [0, 0])
        self.assertEqual(result, {"precision": 0.0, "recall": 0.0})

    def test_rejects_mismatched_or_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            binary_precision_recall([1], [])
        with self.assertRaisesRegex(ValueError, "at least one"):
            binary_precision_recall([], [])

    def test_fixed_fixture_has_eight_rows_and_stratified_splits(self):
        self.assertTrue(self.fixture["synthetic"])
        self.assertEqual(len(self.records), 8)
        self.assertEqual(
            [record["x1"] for record in self.records], list(range(8))
        )
        self.assertTrue(validate_stratified_splits(self.records, self.splits))

    def test_rejects_a_group_crossing_splits(self):
        bad_splits = {name: ids[:] for name, ids in self.splits.items()}
        bad_splits["validation"][0] = "row-001"
        with self.assertRaisesRegex(ValueError, "only one split"):
            validate_stratified_splits(self.records, bad_splits)


if __name__ == "__main__":
    unittest.main()
