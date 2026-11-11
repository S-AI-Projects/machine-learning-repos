from collections import Counter


def fit_majority(rows, train_ids):
    if not train_ids or len(train_ids) != len(set(train_ids)):
        raise ValueError('invalid training IDs')
    selected = [row for row in rows if row['record_id'] in train_ids]
    if len(selected) != len(train_ids):
        raise ValueError('missing training ID')
    labels = [row['label'] for row in selected]
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError('invalid label')
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def predict(majority, rows):
    return [majority for _ in rows]


def accuracy(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError('aligned nonempty inputs required')
    return sum(a == b for a, b in zip(expected, predicted)) / len(expected)
