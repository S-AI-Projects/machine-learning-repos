from array import array
import math

FEATURES = ('x1', 'x2')


def fit(rows, train_ids, validation_ids, test_ids):
    parts = [set(part) for part in (train_ids, validation_ids, test_ids)]
    if any(len(part) != len(set(original)) for part, original in zip(parts, (train_ids, validation_ids, test_ids))):
        raise ValueError('duplicate split ID')
    if any(parts[a] & parts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('split contamination')
    by_id = {row['record_id']: row for row in rows}
    if len(by_id) != len(rows) or set.union(*parts) != set(by_id) or not train_ids:
        raise ValueError('invalid split coverage')
    if any(type(row.get('label')) is not int or row['label'] not in (0, 1) for row in rows):
        raise ValueError('invalid label')
    train = [by_id[record_id] for record_id in train_ids]
    values = [tuple(_number(row, feature) for feature in FEATURES) for row in train]
    means = tuple(sum(v[i] for v in values) / len(values) for i in range(2))
    stds = tuple((sum((v[i] - means[i]) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
                 for i in range(2))
    return {'features': FEATURES, 'means': means, 'stds': stds}


def _number(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key}: expected finite number; missing values rejected')
    return value


def transform(model, rows, feature_order=FEATURES):
    if tuple(feature_order) != FEATURES or tuple(model['features']) != FEATURES:
        raise ValueError('feature order mismatch')
    return [array('d', [(_number(row, key) - model['means'][i]) / model['stds'][i]
                        for i, key in enumerate(FEATURES)]) for row in rows]
