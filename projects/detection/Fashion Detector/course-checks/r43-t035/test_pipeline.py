import json
from pathlib import Path
import unittest
from pipeline import fit, transform


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())
        s = self.fixture['split']
        self.model = fit(self.fixture['rows'], s['train'], s['validation'], s['test'])

    def test_values_dtype_shape(self):
        self.assertEqual(self.model['means'], (2.5, 1.0))
        rows = transform(self.model, [self.fixture['rows'][2]])
        self.assertEqual((len(rows), len(rows[0]), rows[0].typecode), (1, 2, 'd'))
        self.assertAlmostEqual(rows[0][0], -0.5 / (4.25 ** 0.5))
        self.assertEqual(rows[0][1], 0.0)

    def test_heldout_cannot_refit(self):
        changed = [dict(r, x1=1000) if r['record_id'] == 'r7' else r for r in self.fixture['rows']]
        s = self.fixture['split']
        self.assertEqual(fit(changed, s['train'], s['validation'], s['test']), self.model)
        with self.assertRaisesRegex(ValueError, 'contamination'):
            fit(changed, s['train'] + ['r7'], s['validation'], s['test'])

    def test_reject_missing_or_reordered(self):
        with self.assertRaisesRegex(ValueError, 'x1'):
            transform(self.model, [{'x2': 1}])
        with self.assertRaisesRegex(ValueError, 'order'):
            transform(self.model, self.fixture['rows'][:1], ('x2', 'x1'))
