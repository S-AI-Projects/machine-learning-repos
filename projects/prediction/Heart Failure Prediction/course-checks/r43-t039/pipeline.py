import math


def predict_by_id(rows):
    """Fixed threshold verifies row-order invariance, not model robustness."""
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('expected record object')
        record_id = row.get('record_id')
        if not isinstance(record_id, str) or not record_id or record_id in result:
            raise ValueError('missing or duplicate record ID')
        for key in ('x1', 'x2'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{record_id}: {key} missing or invalid')
        result[record_id] = int(row['x1'] >= 4)
    return result
