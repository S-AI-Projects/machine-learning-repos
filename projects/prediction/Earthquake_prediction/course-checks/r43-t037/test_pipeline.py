import json
from pathlib import Path
import unittest
from pipeline import load, predict


class ConfigTest(unittest.TestCase):
    def setUp(self):
        here = Path(__file__).parent
        self.rows = json.loads((here / 'fixture.json').read_text())['rows']
        self.text = (here / 'config.json').read_text()

    def test_round_trip(self):
        a = load(self.text, self.rows)
        b = load(json.dumps(a), self.rows)
        self.assertEqual(a['split'], b['split'])
        self.assertEqual(predict(a, self.rows), predict(b, self.rows))
        self.assertEqual(predict(a, self.rows), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_unknown_and_shape(self):
        bad = json.loads(self.text)
        bad['mystery'] = True
        with self.assertRaisesRegex(ValueError, 'configuration'):
            load(json.dumps(bad), self.rows)
        with self.assertRaisesRegex(ValueError, 'shape'):
            predict(load(self.text, self.rows), [{'x1': [1], 'x2': 1}])
