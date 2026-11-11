import json
from pathlib import Path
import unittest
from pipeline import accuracy, fit_majority, predict


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())

    def test_holdout(self):
        rows, split = self.fixture['rows'], self.fixture['split']
        majority = fit_majority(rows, split['train'])
        self.assertEqual(majority, 0)  # deterministic tie rule
        for name in ('validation', 'test'):
            held = [r for r in rows if r['record_id'] in split[name]]
            self.assertEqual(accuracy([r['label'] for r in held], predict(majority, held)), 0.5)

    def test_held_out_labels_do_not_fit(self):
        rows = [dict(r, label=0) if r['record_id'] in ('r2', 'r3', 'r6', 'r7') else r
                for r in self.fixture['rows']]
        self.assertEqual(fit_majority(rows, self.fixture['split']['train']), 0)

    def test_missing_training_id(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            fit_majority(self.fixture['rows'], ['absent'])
