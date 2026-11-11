import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from pipeline import fit, load, predict, save


class SavedPipelineTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())

    def test_round_trip_without_training_data_at_inference(self):
        rows, split = self.fixture['rows'], self.fixture['split']
        model = fit(rows, split['train'])
        self.assertEqual(model['threshold'], 2.5)
        held = [r for r in rows if r['record_id'] in split['test']]
        before = predict(model, held)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            save(model, path)
            loaded = load(path)
            self.assertEqual(predict(loaded, held), before)  # exact tolerance: 0
            self.assertEqual(before, {'r3': 1, 'r7': 1})

    def test_schema_and_version_rejected(self):
        model = fit(self.fixture['rows'], self.fixture['split']['train'])
        with self.assertRaisesRegex(ValueError, 'version'):
            predict(dict(model, schema_version=2), self.fixture['rows'][:1])
        with self.assertRaisesRegex(ValueError, 'x2'):
            predict(model, [{'record_id': 'new', 'group_id': 'new', 'x1': 1}])
