import json
from pathlib import Path
import unittest
from pipeline import error_report


class ErrorTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())
        self.predictions = {row['record_id']: int(row['x1'] >= 4) for row in self.fixture['rows']}

    def test_injected_error(self):
        self.predictions['r7'] = 0
        held = [r for r in self.fixture['rows'] if r['record_id'] in self.fixture['split']['test']]
        report = error_report(held, {r['record_id']: self.predictions[r['record_id']] for r in held})
        self.assertEqual(report['errors'], [{'record_id': 'r7', 'expected': 1,
                                             'predicted': 0, 'reason': 'false_negative'}])
        self.assertEqual(report['counts'], {'false_negative': 1})
        self.assertNotIn('r3', [r['record_id'] for r in report['errors']])
        self.assertEqual(sum(report['counts'].values()), len(report['errors']))

    def test_aligned_inputs(self):
        with self.assertRaisesRegex(ValueError, 'align'):
            error_report(self.fixture['rows'], {'r0': 0})
