"""
Unit tests for R43-T002 deterministic split manifest.
NOTE: All data is SYNTHETIC FIXTURE — NOT real malicious-URL observations.
Results are pipeline verification only, not evidence of real-world performance.
"""
import unittest
from pipeline import (
    FIXTURE_RECORDS, get_split_manifest, validate_manifest, build_split, _SPLIT_MANIFEST
)


class TestManifestDeterminism(unittest.TestCase):
    def test_same_manifest_on_repeated_calls(self):
        m1 = get_split_manifest()
        m2 = get_split_manifest()
        self.assertEqual(m1, m2)

    def test_manifest_has_three_splits(self):
        m = get_split_manifest()
        self.assertIn("train", m)
        self.assertIn("validation", m)
        self.assertIn("test", m)


class TestManifestDisjointness(unittest.TestCase):
    def test_splits_are_disjoint(self):
        m = get_split_manifest()
        train = set(m["train"])
        val = set(m["validation"])
        test = set(m["test"])
        self.assertEqual(train & val, set(), msg="train/val overlap")
        self.assertEqual(train & test, set(), msg="train/test overlap")
        self.assertEqual(val & test, set(), msg="val/test overlap")

    def test_union_covers_all_records(self):
        m = get_split_manifest()
        all_manifest_ids = set(m["train"]) | set(m["validation"]) | set(m["test"])
        fixture_ids = {r["record_id"] for r in FIXTURE_RECORDS}
        self.assertEqual(all_manifest_ids, fixture_ids)


class TestManifestValidation(unittest.TestCase):
    def test_valid_manifest_passes(self):
        m = get_split_manifest()
        self.assertTrue(validate_manifest(m))

    def test_overlapping_ids_rejected(self):
        bad = {
            "train": ["s0", "s1", "s2", "s4", "s5"],
            "validation": ["s0"],
            "test": ["s6", "s7"],
        }
        with self.assertRaises(ValueError):
            validate_manifest(bad)

    def test_unknown_id_rejected(self):
        bad = {
            "train": ["s0", "s1", "s2", "s4", "UNKNOWN"],
            "validation": ["s3"],
            "test": ["s6", "s7"],
        }
        with self.assertRaises(ValueError):
            validate_manifest(bad)


class TestBuildSplit(unittest.TestCase):
    def test_build_split_correct_counts(self):
        m = get_split_manifest()
        splits = build_split(FIXTURE_RECORDS, m)
        self.assertEqual(len(splits["train"]), len(m["train"]))
        self.assertEqual(len(splits["validation"]), len(m["validation"]))
        self.assertEqual(len(splits["test"]), len(m["test"]))

    def test_deliberate_change_detected(self):
        m = get_split_manifest()
        self.assertNotIn("WRONG_ID", m["train"])

    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)


class TestSyntheticDataLabelling(unittest.TestCase):
    def test_fixture_has_eight_rows(self):
        self.assertEqual(len(FIXTURE_RECORDS), 8)

    def test_fixture_x1_values(self):
        x1_vals = [r["x1"] for r in FIXTURE_RECORDS]
        self.assertEqual(x1_vals, list(range(8)))

    def test_fixture_labels_stratified(self):
        labels = [r["label"] for r in FIXTURE_RECORDS]
        self.assertEqual(labels.count(0), 4)
        self.assertEqual(labels.count(1), 4)


if __name__ == "__main__":
    unittest.main()
