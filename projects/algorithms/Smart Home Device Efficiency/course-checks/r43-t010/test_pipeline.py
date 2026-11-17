"""
Unit tests for R43-T010 saved-pipeline interface.
NOTE: SYNTHETIC FIXTURE only. Deterministic formula: target=2*x1+1. Pipeline verification only.
"""
import unittest
from pipeline import LinearRegressionFixture, FIXTURE_RECORDS, TRAIN_IDS, TEST_IDS, TOLERANCE, SCHEMA_VERSION


def get_split(ids):
    id_map = {r["record_id"]: r for r in FIXTURE_RECORDS}
    return [id_map[i] for i in ids]


class TestFitPredict(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegressionFixture().fit(get_split(TRAIN_IDS))

    def test_slope_correct(self):
        self.assertAlmostEqual(self.model._slope, 2.0, places=9)

    def test_intercept_correct(self):
        self.assertAlmostEqual(self.model._intercept, 1.0, places=9)

    def test_predict_test_set(self):
        preds = self.model.predict(get_split(TEST_IDS))
        expected = [2*r["x1"]+1 for r in get_split(TEST_IDS)]
        for p, e in zip(preds, expected):
            self.assertAlmostEqual(p, e, delta=TOLERANCE)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LinearRegressionFixture().predict(get_split(TEST_IDS))


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip_predictions_agree(self):
        m1 = LinearRegressionFixture().fit(get_split(TRAIN_IDS))
        preds1 = m1.predict(get_split(TEST_IDS))
        m2 = LinearRegressionFixture.from_json(m1.to_json())
        preds2 = m2.predict(get_split(TEST_IDS))
        for p1, p2 in zip(preds1, preds2):
            self.assertAlmostEqual(p1, p2, delta=TOLERANCE)

    def test_wrong_schema_version_raises(self):
        import json
        m = LinearRegressionFixture().fit(get_split(TRAIN_IDS))
        data = json.loads(m.to_json())
        data["schema_version"] = "9.9.9"
        with self.assertRaises(ValueError):
            LinearRegressionFixture.from_json(json.dumps(data))

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            LinearRegressionFixture.predict(LinearRegressionFixture().fit(get_split(TRAIN_IDS)),
                                            [{"record_id":"bad","x2":1,"target":0}])


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        m = LinearRegressionFixture().fit(get_split(TRAIN_IDS))
        preds = m.predict(get_split(TEST_IDS))
        self.assertFalse(all(abs(p - 999.0) < TOLERANCE for p in preds))


if __name__ == "__main__":
    unittest.main()
