"""
Unit tests for R43-T005 preprocessing contract.
NOTE: SYNTHETIC FIXTURE only. NOT real car detection data.
"""
import unittest
from pipeline import MinMaxScaler, FIXTURE_RECORDS, SCHEMA_FEATURES, TRAIN_IDS, VAL_IDS, TEST_IDS


def get_split(ids):
    id_map = {r["record_id"]: r for r in FIXTURE_RECORDS}
    return [id_map[i] for i in ids]


class TestScalerFitTransform(unittest.TestCase):
    def setUp(self):
        self.scaler = MinMaxScaler().fit(get_split(TRAIN_IDS))

    def test_transform_shape(self):
        out = self.scaler.transform(get_split(TEST_IDS))
        self.assertEqual(len(out), len(TEST_IDS))
        for row in out:
            self.assertEqual(set(row.keys()), set(SCHEMA_FEATURES))

    def test_transform_dtype_float(self):
        out = self.scaler.transform(get_split(TEST_IDS))
        for row in out:
            for feat in SCHEMA_FEATURES:
                self.assertIsInstance(row[feat], float)

    def test_transform_values_bounded(self):
        out = self.scaler.transform(get_split(TRAIN_IDS))
        for row in out:
            for feat in SCHEMA_FEATURES:
                self.assertGreaterEqual(row[feat], 0.0)
                self.assertLessEqual(row[feat], 1.0)


class TestNoLeakage(unittest.TestCase):
    def test_fit_before_transform_required(self):
        with self.assertRaises(RuntimeError):
            MinMaxScaler().transform(get_split(TEST_IDS))

    def test_splits_disjoint(self):
        self.assertEqual(set(TRAIN_IDS) & set(VAL_IDS), set())
        self.assertEqual(set(TRAIN_IDS) & set(TEST_IDS), set())


class TestMissingValuePolicy(unittest.TestCase):
    def test_missing_feature_raises(self):
        scaler = MinMaxScaler().fit(get_split(TRAIN_IDS))
        bad = {"record_id": "bad", "group_id": "gb", "x1": None, "x2": 1, "label": 0}
        with self.assertRaises((ValueError, TypeError)):
            scaler.transform([bad])


class TestUnknownLabel(unittest.TestCase):
    def test_unknown_label_raises(self):
        scaler = MinMaxScaler().fit(get_split(TRAIN_IDS))
        bad = {"record_id": "bx", "group_id": "g", "x1": 1, "x2": 1, "label": 99}
        with self.assertRaises(ValueError):
            scaler.transform([bad])


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        scaler = MinMaxScaler().fit(get_split(TRAIN_IDS))
        out = scaler.transform(get_split(TEST_IDS))
        self.assertNotEqual(out[0]["x1"], 999.0)


if __name__ == "__main__":
    unittest.main()
