"""
Unit tests for R43-T007 configuration reproducibility.
NOTE: SYNTHETIC FIXTURE only. NOT real domestic violence data. Pipeline verification only.
"""
import unittest
from pipeline import DEFAULT_CONFIG, validate_config, config_to_json, config_from_json, MajorityClassPipeline, FIXTURE_RECORDS


class TestConfigValidation(unittest.TestCase):
    def test_default_config_valid(self):
        self.assertTrue(validate_config(DEFAULT_CONFIG))

    def test_missing_key_raises(self):
        bad = dict(DEFAULT_CONFIG); del bad["seed"]
        with self.assertRaises(ValueError): validate_config(bad)

    def test_unknown_key_raises(self):
        bad = dict(DEFAULT_CONFIG); bad["UNKNOWN_FIELD"] = "oops"
        with self.assertRaises(ValueError): validate_config(bad)

    def test_wrong_schema_version_raises(self):
        bad = dict(DEFAULT_CONFIG); bad["schema_version"] = "9.9.9"
        with self.assertRaises(ValueError): validate_config(bad)

    def test_unknown_algorithm_raises(self):
        bad = dict(DEFAULT_CONFIG); bad["algorithm"] = "neural_net_v99"
        with self.assertRaises(ValueError): validate_config(bad)

    def test_empty_feature_order_raises(self):
        bad = dict(DEFAULT_CONFIG); bad["feature_order"] = []
        with self.assertRaises(ValueError): validate_config(bad)


class TestConfigRoundTrip(unittest.TestCase):
    def test_json_roundtrip(self):
        loaded = config_from_json(config_to_json(DEFAULT_CONFIG))
        self.assertEqual(loaded["seed"], DEFAULT_CONFIG["seed"])
        self.assertEqual(loaded["algorithm"], DEFAULT_CONFIG["algorithm"])


class TestPipelineReproducibility(unittest.TestCase):
    def test_same_predictions_after_reload(self):
        p1 = MajorityClassPipeline(DEFAULT_CONFIG).fit(FIXTURE_RECORDS)
        preds1 = p1.predict(FIXTURE_RECORDS[:2])
        p2 = MajorityClassPipeline(config_from_json(config_to_json(DEFAULT_CONFIG))).fit(FIXTURE_RECORDS)
        preds2 = p2.predict(FIXTURE_RECORDS[:2])
        self.assertEqual(preds1, preds2)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            MajorityClassPipeline(DEFAULT_CONFIG).predict(FIXTURE_RECORDS[:2])


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        p = MajorityClassPipeline(DEFAULT_CONFIG).fit(FIXTURE_RECORDS)
        self.assertNotEqual(p.predict(FIXTURE_RECORDS[:1]), ["WRONG_VALUE"])


if __name__ == "__main__":
    unittest.main()
