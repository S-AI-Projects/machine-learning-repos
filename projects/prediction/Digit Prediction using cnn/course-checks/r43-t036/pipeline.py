import math


def run(rows):
    """Fixed diagonal detector: a naive, untrained CPU baseline with zero updates."""
    if len(rows) > 32:
        raise ValueError('too many samples')
    predictions = []
    for row in rows:
        pixels = row.get('pixels')
        if not isinstance(pixels, list) or len(pixels) != 4 or any(
            not isinstance(line, list) or len(line) != 4 or any(
                type(value) is not int or not 0 <= value <= 255 for value in line)
            for line in pixels
        ):
            raise ValueError('pixels: expected 4x4 grayscale integers')
        predictions.append(int(all(pixels[i][i] == 255 for i in range(4))))
    labels = [row['label'] for row in rows]
    if any(type(v) is not int or v not in (0, 1) for v in labels):
        raise ValueError('binary labels required')
    loss = sum((a - b) ** 2 for a, b in zip(predictions, labels)) / len(rows) if rows else 0.0
    assert math.isfinite(loss)
    return {'predictions': predictions, 'loss': loss, 'steps': 0, 'epochs': 0}
