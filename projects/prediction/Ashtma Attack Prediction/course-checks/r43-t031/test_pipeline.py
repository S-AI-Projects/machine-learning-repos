import json
from pathlib import Path
import unittest
from pipeline import validate


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads(Path(__file__).with_name('fixture.json').read_text())['rows']

    def test_valid_fixture(self):
        self.assertEqual(validate(self.rows), [])
        self.assertEqual([r['x1'] for r in self.rows], list(range(8)))

    def test_missing_id(self):
        rows = [dict(self.rows[0])]
        del rows[0]['record_id']
        self.assertIn(('row:0', 'record_id', 'missing'), validate(rows))

    def test_bad_values_and_duplicate(self):
        rows = [dict(self.rows[0]), dict(self.rows[0])]
        rows[0].update(x1='bad', x2=[1], label=True, group_id=42)
        errors = validate(rows)
        for failure in [('r0', 'x1', 'expected finite number'),
                        ('r0', 'x2', 'expected finite number'),
                        ('r0', 'label', 'expected class 0 or 1'),
                        ('r0', 'group_id', 'expected nonempty text'),
                        ('r0', 'record_id', 'duplicate')]:
            self.assertIn(failure, errors)

    def test_nonfinite_and_malformed_row(self):
        self.assertIn(('r0', 'x1', 'expected finite number'), validate([dict(self.rows[0], x1=float('nan'))]))
        self.assertEqual(validate([['tensor']]), [('row:0', 'row', 'expected object')])
