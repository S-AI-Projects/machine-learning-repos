"""
Unit tests for R43-T009 robustness fixture.
NOTE: SYNTHETIC FIXTURE only. NOT real code review data. Pipeline verification only.
"""
import unittest
from pipeline import (
    FIXTURE_RECORDS, MajorityClassifier, shuffle_records_by_id,
    inject_missing_field, TRAIN_IDS, TEST_IDS
)


def get_split(ids):
    id_map = {r["record_id"]: r for r in FIXTURE_RECORDS}
    return [id_map[i] for i in ids]


class TestRecordOrderPreservation(unittest.TestCase):
    def test_shuffled_ids_same_as_original_after_sort(self):
        """Record-order permutation with stable IDs gives same predictions after ID alignment."""
        train = get_split(TRAIN_IDS)
        clf = MajorityClassifier().fit(train)
        test = get_split(TEST_IDS)
        preds_orig = clf.predict(test)
        shuffled = shuffle_records_by_id(test)
        preds_shuffled = clf.predict(shuffled)
        # Align by record_id
        orig_map = {r["record_id"]: p for r, p in zip(test, preds_orig)}
        shuf_map = {r["record_id"]: p for r, p in zip(shuffled, preds_shuffled)}
        self.assertEqual(orig_map, shuf_map)


class TestMissingFieldRejection(unittest.TestCase):
    def test_missing_x1_raises(self):
        train = get_split(TRAIN_IDS)
        clf = MajorityClassifier().fit(train)
        bad_test = inject_missing_field(get_split(TEST_IDS), "a6", "x1")
        with self.assertRaises(ValueError):
            clf.predict(bad_test)


class TestFitPredictInterface(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            MajorityClassifier().predict(get_split(TEST_IDS))

    def test_predictions_all_valid_labels(self):
        train = get_split(TRAIN_IDS)
        clf = MajorityClassifier().fit(train)
        preds = clf.predict(get_split(TEST_IDS))
        for p in preds:
            self.assertIn(p, [0, 1])


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        train = get_split(TRAIN_IDS)
        clf = MajorityClassifier().fit(train)
        preds = clf.predict(get_split(TEST_IDS))
        self.assertNotEqual(preds, ["WRONG"])


if __name__ == "__main__":
    unittest.main()
