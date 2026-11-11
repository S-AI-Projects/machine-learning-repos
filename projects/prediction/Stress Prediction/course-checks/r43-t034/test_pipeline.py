import json
from pathlib import Path
import unittest
from pipeline import precision_recall


class MetricTest(unittest.TestCase):
    def test_cross_check(self):
        self.assertEqual(precision_recall([1, 1, 0, 0], [1, 0, 1, 0]), (0.5, 0.5))

    def test_zero_denominator(self):
        self.assertEqual(precision_recall([0, 0], [0, 0]), (0.0, 0.0))
        self.assertEqual(precision_recall([1, 1], [0, 0]), (0.0, 0.0))

    def test_fixture_holdout_and_invalid(self):
        fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())
        rows = [r for r in fixture['rows'] if r['record_id'] in fixture['split']['test']]
        self.assertEqual(precision_recall([r['label'] for r in rows], [0, 0]), (0.0, 0.0))
        with self.assertRaises(ValueError):
            precision_recall([1, 0], [1])
