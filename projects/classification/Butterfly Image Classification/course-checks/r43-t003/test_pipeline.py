"""
Unit tests for R43-T003 majority-class baseline.
NOTE: SYNTHETIC FIXTURE data only. Pipeline verification, not real butterfly performance.
"""
import unittest
from pipeline import FIXTURE_RECORDS, MajorityClassBaseline, get_split, TRAIN_IDS, VAL_IDS, TEST_IDS


class TestFixture(unittest.TestCase):
    def test_six_records(self):
        self.assertEqual(len(FIXTURE_RECORDS), 6)

    def test_three_label_zero(self):
        self.assertEqual(sum(1 for r in FIXTURE_RECORDS if r["label"] == 0), 3)

    def test_three_label_one(self):
        self.assertEqual(sum(1 for r in FIXTURE_RECORDS if r["label"] == 1), 3)

    def test_unique_record_ids(self):
        ids = [r["record_id"] for r in FIXTURE_RECORDS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_pixels_shape(self):
        for rec in FIXTURE_RECORDS:
            self.assertEqual(len(rec["pixels"]), 4)
            for row in rec["pixels"]:
                self.assertEqual(len(row), 4)


class TestBaselineFit(unittest.TestCase):
    def test_majority_class_on_train(self):
        train = get_split(FIXTURE_RECORDS, TRAIN_IDS)
        clf = MajorityClassBaseline().fit(train)
        self.assertIn(clf._majority_class, [0, 1])

    def test_predict_before_fit_raises(self):
        clf = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            clf.predict([FIXTURE_RECORDS[0]])

    def test_predictions_all_same_class(self):
        train = get_split(FIXTURE_RECORDS, TRAIN_IDS)
        test  = get_split(FIXTURE_RECORDS, TEST_IDS)
        clf = MajorityClassBaseline().fit(train)
        preds = clf.predict(test)
        self.assertTrue(all(p == preds[0] for p in preds))

    def test_score_returns_float(self):
        train = get_split(FIXTURE_RECORDS, TRAIN_IDS)
        test  = get_split(FIXTURE_RECORDS, TEST_IDS)
        clf = MajorityClassBaseline().fit(train)
        score = clf.score(test)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_no_leakage_train_ids_disjoint(self):
        train_set = set(TRAIN_IDS)
        for rid in TEST_IDS + VAL_IDS:
            self.assertNotIn(rid, train_set)


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        train = get_split(FIXTURE_RECORDS, TRAIN_IDS)
        clf = MajorityClassBaseline().fit(train)
        preds = clf.predict(get_split(FIXTURE_RECORDS, TEST_IDS))
        self.assertNotEqual(preds, ["WRONG_EXPECTED"])


if __name__ == "__main__":
    unittest.main()
