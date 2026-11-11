import json
import math
from pathlib import Path
import unittest
from pipeline import MAX_UPDATES, TinyLogisticBaseline, run_smoke, stratified_split, validate_group_separation, validate_records

HERE = Path(__file__).resolve().parent
with (HERE / "fixture.json").open(encoding="utf-8") as source:
    FIXTURE = json.load(source)

class SmokeTests(unittest.TestCase):
    def test_fixture_values_and_ids(self):
        rows = FIXTURE["records"]
        self.assertTrue(validate_records(rows))
        self.assertEqual(FIXTURE["split_seed"], 43)
        self.assertEqual([r["x1"] for r in rows], list(range(8)))
        self.assertEqual([r["x2"] for r in rows], [1]*8)
        self.assertEqual([r["label"] for r in rows], [0]*4+[1]*4)
        self.assertEqual(len({r["record_id"] for r in rows}), 8)
        self.assertEqual(len({r["group_id"] for r in rows}), 8)

    def test_fixed_seed_split_is_repeatable_and_stratified(self):
        rows, seed = FIXTURE["records"], FIXTURE["split_seed"]
        first, second = stratified_split(rows, seed), stratified_split(rows, seed)
        self.assertEqual(first, second)
        self.assertEqual([len(first[s]) for s in ("train","validation","test")], [4,2,2])
        for split in first.values():
            self.assertEqual({r["label"] for r in split}, {0,1})
        self.assertTrue(validate_group_separation(first))

    def test_smoke_run_has_finite_loss_predictions_shapes_and_bounded_steps(self):
        result = run_smoke(FIXTURE["records"], FIXTURE["split_seed"])
        self.assertEqual((result["epochs"], result["steps"]), (1,4))
        self.assertLessEqual(result["steps"], MAX_UPDATES)
        self.assertTrue(math.isfinite(result["train_loss"]))
        self.assertEqual(result["validation_prediction_shape"], [2,1])
        self.assertEqual(result["test_prediction_shape"], [2,1])
        for key in ("validation_predictions", "test_predictions"):
            for row in result[key]:
                self.assertEqual(len(row), 1)
                self.assertTrue(math.isfinite(row[0]))
                self.assertGreaterEqual(row[0], 0.0)
                self.assertLessEqual(row[0], 1.0)

    def test_smoke_result_is_repeatable(self):
        args = (FIXTURE["records"], FIXTURE["split_seed"])
        self.assertEqual(run_smoke(*args), run_smoke(*args))

    def test_group_contamination_is_rejected(self):
        splits = stratified_split(FIXTURE["records"], FIXTURE["split_seed"])
        bad = {name: [dict(r) for r in rows] for name, rows in splits.items()}
        bad["validation"][0]["group_id"] = bad["train"][0]["group_id"]
        with self.assertRaisesRegex(ValueError, "crosses split boundaries"):
            validate_group_separation(bad)

    def test_invalid_rows_and_training_limits_are_rejected(self):
        bad = [dict(r) for r in FIXTURE["records"]]
        bad[1]["record_id"] = bad[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "record_id"):
            validate_records(bad)
        bad = [dict(r) for r in FIXTURE["records"]]
        bad[0]["x2"] = 0
        with self.assertRaisesRegex(ValueError, "x2"):
            validate_records(bad)
        train = stratified_split(FIXTURE["records"], 43)["train"]
        with self.assertRaisesRegex(ValueError, "one training epoch"):
            TinyLogisticBaseline().fit(train, epochs=2)
        with self.assertRaisesRegex(ValueError, "update steps"):
            TinyLogisticBaseline().fit(train, max_updates=3)

if __name__ == "__main__":
    unittest.main()
