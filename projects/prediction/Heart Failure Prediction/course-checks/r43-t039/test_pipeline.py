import json
from pathlib import Path
import unittest
from pipeline import predict_by_id


class PerturbationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())

    def test_order_invariance_after_id_alignment(self):
        rows = [r for r in self.fixture['rows'] if r['record_id'] in self.fixture['split']['test']]
        expected = predict_by_id(rows)
        self.assertEqual(expected, {'r3': 0, 'r7': 1})
        self.assertEqual(predict_by_id(list(reversed(rows))), expected)

    def test_missing_feature_rejected(self):
        bad = dict(self.fixture['rows'][0])
        del bad['x1']
        with self.assertRaisesRegex(ValueError, 'x1'):
            predict_by_id([bad])
