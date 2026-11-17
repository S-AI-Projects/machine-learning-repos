import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import create_deterministic_split, validate_split_manifest, SplitLeakageError


class TestSplitLeakagePrevention(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(curr_dir, "fixture.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture_data = json.load(f)
        self.records = self.fixture_data["records"]

    def test_fixture_exact_dimensions_and_labels(self):
        self.assertEqual(len(self.records), 8)
        x1_values = [r["x1"] for r in self.records]
        x2_values = [r["x2"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(x1_values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(x2_values, [1.0] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_splits_are_strictly_disjoint(self):
        manifest = create_deterministic_split(self.records)
        train_set = set(manifest["train"])
        val_set = set(manifest["val"])
        test_set = set(manifest["test"])

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 8)

    def test_deterministic_reproducibility(self):
        manifest1 = create_deterministic_split(self.records)
        manifest2 = create_deterministic_split(self.records)
        self.assertEqual(manifest1, manifest2)

    def test_overlapping_ids_rejected(self):
        leaky_manifest = {
            "train": ["rec_0", "rec_1", "rec_4"],
            "val": ["rec_2", "rec_4"],  # rec_4 leaked into val!
            "test": ["rec_3", "rec_5", "rec_6", "rec_7"]
        }
        with self.assertRaises(SplitLeakageError) as ctx:
            validate_split_manifest(leaky_manifest)
        self.assertIn("rec_4", str(ctx.exception))

    def test_group_isolation_leakage_rejected(self):
        # grp_0 contains rec_0 and rec_1
        leaky_group_manifest = {
            "train": ["rec_0"],
            "val": ["rec_1"],  # rec_1 shares grp_0 with rec_0 in train
            "test": ["rec_2", "rec_3", "rec_4", "rec_5", "rec_6", "rec_7"]
        }
        with self.assertRaises(SplitLeakageError) as ctx:
            validate_split_manifest(leaky_group_manifest, self.records, enforce_group_isolation=True)
        self.assertIn("Group leakage detected", str(ctx.exception))

    def test_time_series_chronological_ordering(self):
        manifest = create_deterministic_split(self.records, is_time_series=True)
        rec_map = {r["record_id"]: r["timestamp"] for r in self.records}

        train_times = [rec_map[rid] for rid in manifest["train"]]
        val_times = [rec_map[rid] for rid in manifest["val"]]
        test_times = [rec_map[rid] for rid in manifest["test"]]

        self.assertTrue(max(train_times) < min(val_times))
        self.assertTrue(max(val_times) < min(test_times))


if __name__ == "__main__":
    unittest.main()
