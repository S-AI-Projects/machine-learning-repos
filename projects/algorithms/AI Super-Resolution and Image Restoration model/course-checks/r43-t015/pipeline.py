"""Stateless preprocessing contract for the synthetic R43-T015 fixture."""

from array import array
from dataclasses import dataclass
import math
from numbers import Real

IMAGE_SHAPE = (4, 4)
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ImageTensor:
    """Flat float32 storage with explicit tensor shape and dtype metadata."""

    data: array
    shape: tuple = (4, 4, 1)

    @property
    def dtype(self):
        return "float32"


def transform_pixels(pixels):
    """Scale a 4x4 grayscale image to float32 [0, 1] with a channel axis."""
    if not isinstance(pixels, (list, tuple)) or len(pixels) != 4:
        raise ValueError("pixels must have shape (4, 4)")
    if any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in pixels):
        raise ValueError("pixels must have shape (4, 4)")

    flattened = []
    for row in pixels:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("pixels must be numeric")
            if not math.isfinite(value):
                raise ValueError("pixels must be finite")
            if value < 0 or value > 255:
                raise ValueError("pixels must be in the inclusive range 0..255")
            flattened.append(float(value) / 255.0)

    values = array("f", flattened)
    if values.itemsize != 4:
        raise RuntimeError("this platform does not provide 32-bit array('f') values")
    return ImageTensor(data=values)


def validate_splits(records):
    """Reject duplicate IDs, missing splits, and groups crossing split boundaries."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    record_ids = set()
    group_splits = {}
    observed_splits = set()

    for record in records:
        record_id = record.get("record_id")
        group_id = record.get("group_id")
        split = record.get("split")

        if not record_id or record_id in record_ids:
            raise ValueError("record_id values must be present and unique")
        if not group_id:
            raise ValueError("every record must declare a group_id")
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")

        record_ids.add(record_id)
        observed_splits.add(split)
        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            raise ValueError(f"group {group_id!r} crosses split boundaries")

    if observed_splits != VALID_SPLITS:
        raise ValueError(f"records must include all splits: {sorted(VALID_SPLITS)}")
    return True
