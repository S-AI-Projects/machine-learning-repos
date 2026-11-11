import math

FIELDS = ('record_id', 'group_id', 'x1', 'x2', 'label')


def validate(rows):
    """Return every (record_id, field, reason) failure; never discard a row."""
    errors, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append((f'row:{index}', 'row', 'expected object'))
            continue
        record_id = row.get('record_id')
        identity = record_id if isinstance(record_id, str) and record_id else f'row:{index}'
        for field in FIELDS:
            if field not in row:
                errors.append((identity, field, 'missing'))
        for field in ('record_id', 'group_id'):
            if field in row and (not isinstance(row[field], str) or not row[field].strip()):
                errors.append((identity, field, 'expected nonempty text'))
        for field in ('x1', 'x2'):
            if field in row and (isinstance(row[field], bool) or not isinstance(row[field], (int, float))
                                 or not math.isfinite(row[field])):
                errors.append((identity, field, 'expected finite number'))
        if 'label' in row and (type(row['label']) is not int or row['label'] not in (0, 1)):
            errors.append((identity, 'label', 'expected class 0 or 1'))
        if isinstance(record_id, str) and record_id:
            if record_id in seen:
                errors.append((identity, 'record_id', 'duplicate'))
            seen.add(record_id)
    return errors
