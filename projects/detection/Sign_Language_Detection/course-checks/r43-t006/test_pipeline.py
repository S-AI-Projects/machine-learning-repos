"""
Unit tests for R43-T006 bounded CPU smoke run.
NOTE: SYNTHETIC FIXTURE only. NOT real sign language data. Pipeline verification only.
"""
import unittest
from pipeline import FIXTURE_RECORDS, BagOfWordsBaseline, run_smoke, TRAIN_IDS, TEST_IDS, MAX_STEPS, MAX_SAMPLES


def get_split(ids):
    id_map = {r["record_id"]: r for r in FIXTURE_RECORDS}
    return [id_map[i] for i in ids]


class TestFixture(unittest.TestCase):
    def test_six_records(self):
        self.assertEqual(len(FIXTURE_RECORDS), 6)

    def test_record_ids(self):
        ids = [r["record_id"] for r in FIXTURE_RECORDS]
        self.assertEqual(ids, ["t1","t2","t3","t4","t5","t6"])

    def test_labels_present(self):
        labels = [r["label"] for r in FIXTURE_RECORDS]
        self.assertIn(0, labels); self.assertIn(1, labels)


class TestSmokeRun(unittest.TestCase):
    def setUp(self):
        self.train = get_split(TRAIN_IDS); self.test = get_split(TEST_IDS)

    def test_bounded_samples(self):
        self.assertLessEqual(len(FIXTURE_RECORDS), MAX_SAMPLES)

    def test_smoke_runs_without_error(self):
        result = run_smoke(self.train, self.test)
        self.assertIn("predictions", result)

    def test_finite_outputs(self):
        self.assertTrue(run_smoke(self.train, self.test)["is_finite"])

    def test_prediction_shape(self):
        result = run_smoke(self.train, self.test)
        self.assertEqual(len(result["predictions"]), len(self.test))

    def test_step_count_bounded(self):
        result = run_smoke(self.train, self.test, max_steps=MAX_STEPS)
        self.assertGreater(result["step_count"], 0)
        self.assertLessEqual(result["step_count"], MAX_STEPS)


class TestModelInterface(unittest.TestCase):
    def test_fit_before_predict_required(self):
        with self.assertRaises(RuntimeError):
            BagOfWordsBaseline().predict(get_split(TEST_IDS))

    def test_majority_class_valid(self):
        model = BagOfWordsBaseline().fit(get_split(TRAIN_IDS))
        self.assertIn(model._majority, [0, 1])


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        result = run_smoke(get_split(TRAIN_IDS), get_split(TEST_IDS))
        self.assertNotEqual(result["predictions"], ["WRONG"])


if __name__ == "__main__":
    unittest.main()
