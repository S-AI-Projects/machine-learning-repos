import json
from pathlib import Path
import unittest
from pipeline import manifest, validate_manifest


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())

    def test_fixed_stratified_manifest(self):
        actual = manifest(self.fixture['rows'])
        self.assertEqual(actual, self.fixture['split'])
        self.assertEqual(manifest(list(reversed(self.fixture['rows']))), actual)

    def test_overlap_is_rejected(self):
        bad = dict(self.fixture['split'])
        bad['test'] = bad['test'] + ['r0']
        with self.assertRaisesRegex(ValueError, 'record IDs'):
            validate_manifest(self.fixture['rows'], bad)
        rows = [dict(r) for r in self.fixture['rows']]
        rows[7]['group_id'] = 'g0'
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            validate_manifest(rows, self.fixture['split'])

    def test_chronological_manifest(self):
        rows = [dict(r, time=i) for i, r in enumerate(self.fixture['rows'])]
        actual = manifest(rows)
        self.assertEqual(actual, {'train': ['r0','r1','r2','r3'], 'validation': ['r4','r5','r6'], 'test': ['r7']})
        bad = dict(actual, test=['r0'], train=['r1','r2','r3','r7'])
        with self.assertRaises(ValueError):
            validate_manifest(rows, bad)

    def test_partial_time_rejected(self):
        rows = [dict(r) for r in self.fixture['rows']]
        rows[0]['time'] = 0
        with self.assertRaisesRegex(ValueError, 'every row'):
            manifest(rows)
