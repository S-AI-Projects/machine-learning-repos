import json
import math

KEYS = {'schema_version', 'seed', 'features', 'split', 'algorithm'}


def validate(config, rows):
    if set(config) != KEYS or config['schema_version'] != 1 or type(config['seed']) is not int:
        raise ValueError('unknown or contradictory configuration')
    if config['features'] != ['x1', 'x2'] or config['algorithm'] != {'kind': 'threshold', 'feature': 'x1', 'cutoff': 4}:
        raise ValueError('unsupported feature order or algorithm')
    split = config['split']
    if set(split) != {'train', 'validation', 'test'}:
        raise ValueError('split names')
    ids = [record_id for part in split.values() for record_id in part]
    if len(ids) != len(set(ids)) or set(ids) != {row['record_id'] for row in rows}:
        raise ValueError('split coverage or overlap')


def predict(config, rows):
    if config['features'] != ['x1', 'x2']:
        raise ValueError('feature order mismatch')
    out = []
    for row in rows:
        if any(isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float))
               or not math.isfinite(row[k]) for k in ('x1', 'x2')):
            raise ValueError('incompatible input shape')
        out.append(int(row['x1'] >= config['algorithm']['cutoff']))
    return out


def load(text, rows):
    config = json.loads(text)
    validate(config, rows)
    return config
