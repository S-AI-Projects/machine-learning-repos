import json
from pathlib import Path
import unittest

from pipeline import fit_training_mean, predict_held_out


FIXTURE = Path(__file__).with_name("fixture.json")


class TrainingMeanBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]

    def test_fit_uses_training_targets_only(self):
        self.assertEqual(fit_training_mean(self.records), 4.0)
        changed_held_out = [
            {**record, "target": 1000}
            if record["split"] != "train" else record
            for record in self.records
        ]
        self.assertEqual(fit_training_mean(changed_held_out), 4.0)

    def test_predicts_validation_and_test_with_training_mean(self):
        predictions = predict_held_out(
            self.records, fit_training_mean(self.records)
        )
        self.assertEqual(predictions, [
            {"id": "r5", "split": "validation", "actual": 9, "prediction": 4.0},
            {"id": "r6", "split": "test", "actual": 11, "prediction": 4.0},
        ])

    def test_fit_requires_training_records(self):
        with self.assertRaisesRegex(ValueError, "at least one training record"):
            fit_training_mean([{"id": "r5", "split": "validation", "target": 9}])

    def test_fit_rejects_non_numeric_training_target(self):
        with self.assertRaisesRegex(TypeError, "targets must be numeric"):
            fit_training_mean([{"id": "r1", "split": "train", "target": "one"}])


if __name__ == "__main__":
    unittest.main()
