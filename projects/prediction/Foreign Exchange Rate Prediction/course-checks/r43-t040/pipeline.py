import json
import math

SCHEMA_VERSION = 1
FEATURES = ['x1', 'x2']


def _check_row(row):
    if not isinstance(row, dict) or not isinstance(row.get('record_id'), str) or not row['record_id']:
        raise ValueError('record_id: expected nonempty text')
    if not isinstance(row.get('group_id'), str) or not row['group_id']:
        raise ValueError('group_id: expected nonempty text')
    for key in FEATURES:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key}: expected finite number')


def fit(rows, train_ids):
    if len(train_ids) != len(set(train_ids)) or not train_ids:
        raise ValueError('training IDs invalid')
    by_id = {row['record_id']: row for row in rows}
    if len(by_id) != len(rows) or not set(train_ids) <= set(by_id):
        raise ValueError('training ID missing or duplicated')
    selected = [by_id[record_id] for record_id in train_ids]
    for row in selected:
        _check_row(row)
    negatives = [row['x1'] for row in selected if row.get('label') == 0]
    positives = [row['x1'] for row in selected if row.get('label') == 1]
    if len(negatives) + len(positives) != len(selected) or not negatives or not positives:
        raise ValueError('binary training labels required')
    return {'schema_version': SCHEMA_VERSION, 'features': FEATURES, 'algorithm': 'x1_threshold',
            'threshold': (max(negatives) + min(positives)) / 2}


def validate_model(model):
    if set(model) != {'schema_version', 'features', 'algorithm', 'threshold'}:
        raise ValueError('model schema mismatch')
    if model['schema_version'] != SCHEMA_VERSION or model['features'] != FEATURES or model['algorithm'] != 'x1_threshold':
        raise ValueError('model version or feature order mismatch')
    t = model['threshold']
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        raise ValueError('invalid threshold')


def predict(model, rows):
    validate_model(model)
    result = {}
    for row in rows:
        _check_row(row)
        record_id = row['record_id']
        if record_id in result:
            raise ValueError('duplicate record ID')
        result[record_id] = int(row['x1'] >= model['threshold'])
    return result


def save(model, path):
    validate_model(model)
    path.write_text(json.dumps(model, sort_keys=True) + '\n')


def load(path):
    model = json.loads(path.read_text())
    validate_model(model)
    return model
