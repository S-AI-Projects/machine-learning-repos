def precision_recall(expected, predicted):
    """Positive class 1; zero denominator gives 0.0 (binary, no averaging)."""
    if not expected or len(expected) != len(predicted):
        raise ValueError('aligned nonempty inputs required')
    if any(type(v) is not int or v not in (0, 1) for v in expected + predicted):
        raise ValueError('binary labels required')
    tp = sum(a == b == 1 for a, b in zip(expected, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(expected, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(expected, predicted))
    return (tp / (tp + fp) if tp + fp else 0.0,
            tp / (tp + fn) if tp + fn else 0.0)
