from collections import Counter


def error_report(rows, predictions):
    by_id = {row['record_id']: row for row in rows}
    if len(by_id) != len(rows) or set(predictions) != set(by_id):
        raise ValueError('record IDs must align exactly')
    errors = []
    for record_id, row in by_id.items():
        expected, predicted = row['label'], predictions[record_id]
        if type(expected) is not int or type(predicted) is not int or expected not in (0, 1) or predicted not in (0, 1):
            raise ValueError('binary labels required')
        if expected != predicted:
            errors.append({'record_id': record_id, 'expected': expected, 'predicted': predicted,
                           'reason': 'false_negative' if expected == 1 else 'false_positive'})
    return {'errors': errors, 'counts': dict(Counter(item['reason'] for item in errors))}
