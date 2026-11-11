import json
import math
from pathlib import Path
import unittest
from pipeline import run


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())

    def test_bounded_cpu_run(self):
        rows = self.fixture['rows']
        self.assertEqual(len(rows), 6)
        self.assertEqual([r['label'] for r in rows], [0, 0, 0, 1, 1, 1])
        for part in self.fixture['split'].values():
            actual = run([r for r in rows if r['record_id'] in part])
            self.assertEqual(len(actual['predictions']), len(part))
            self.assertEqual(actual['loss'], 0.0)
            self.assertTrue(math.isfinite(actual['loss']))
            self.assertLessEqual(actual['steps'], 20)

    def test_bad_shape(self):
        bad = dict(self.fixture['rows'][0], pixels=[[0]])
        with self.assertRaisesRegex(ValueError, '4x4'):
            run([bad])
