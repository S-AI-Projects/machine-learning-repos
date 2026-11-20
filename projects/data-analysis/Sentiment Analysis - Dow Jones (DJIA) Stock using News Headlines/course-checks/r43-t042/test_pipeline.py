import json
import os
import sys
import unittest

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from pipeline import create_deterministic_split, validate_split_manifest, SplitLeakageError, load_fixture


class TestDeterministicSplitManifest(unittest.TestCase):
    def setUp(self):
        curr_dir = os.path.dirname(__file__)
        fixture_path = os.path.join(curr_dir, "fixture.json")
        self.fixture = load_fixture(fixture_path)
        self.records = self.fixture["records"]

    def test_fixture_exact_contents(self):
        self.assertEqual(len(self.records), 6)
        ids = [r["record_id"] for r in self.records]
        texts = [r["text"] for r in self.records]
        labels = [r["label"] for r in self.records]
        self.assertEqual(ids, ["t1", "t2", "t3", "t4", "t5", "t6"])
        self.assertEqual(texts, ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"])
        self.assertEqual(labels, [1, 1, 0, 0, 1, 0])

    def test_split_sets_disjoint(self):
        manifest = create_deterministic_split(self.records, is_time_series=True)
        train_set = set(manifest["train"])
        val_set = set(manifest["val"])
        test_set = set(manifest["test"])

        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))
        self.assertEqual(train_set | val_set | test_set, {"t1", "t2", "t3", "t4", "t5", "t6"})

    def test_determinism_across_repeated_runs(self):
        manifest_1 = create_deterministic_split(self.records, is_time_series=True)
        manifest_2 = create_deterministic_split(self.records, is_time_series=True)
        self.assertEqual(manifest_1, manifest_2)
        self.assertEqual(manifest_1["train"], ["t1", "t2", "t3", "t4"])
        self.assertEqual(manifest_1["val"], ["t5"])
        self.assertEqual(manifest_1["test"], ["t6"])

    def test_chronological_ordering_preserved(self):
        manifest = create_deterministic_split(self.records, is_time_series=True)
        self.assertEqual(manifest["train"], ["t1", "t2", "t3", "t4"])
        self.assertEqual(manifest["val"], ["t5"])
        self.assertEqual(manifest["test"], ["t6"])

    def test_overlapping_id_leakage_rejected(self):
        bad_manifest = {
            "train": ["t1", "t2", "t3"],
            "val": ["t3", "t4"],  # t3 leaked into val
            "test": ["t5", "t6"]
        }
        with self.assertRaises(SplitLeakageError) as ctx:
            validate_split_manifest(bad_manifest, self.records, enforce_chronological=False)
        self.assertIn("t3", str(ctx.exception))

    def test_chronological_inversion_rejected(self):
        bad_time_manifest = {
            "train": ["t1", "t2", "t5"],  # t5 is 2026-01-05
            "val": ["t3"],                # t3 is 2026-01-03 (earlier than train max!)
            "test": ["t4", "t6"]
        }
        with self.assertRaises(SplitLeakageError) as ctx:
            validate_split_manifest(bad_time_manifest, self.records, enforce_chronological=True)
        self.assertIn("Chronological leakage", str(ctx.exception))

    def test_synthetic_fixture_labeled(self):
        self.assertTrue(self.fixture["metadata"]["is_synthetic"])
        self.assertIn("pipeline verification", self.fixture["metadata"]["purpose"].lower())


if __name__ == "__main__":
    unittest.main()
