import json
from collections import Counter
from pathlib import Path
import unittest

from pipeline import transform_pixels, validate_splits


HERE = Path(__file__).resolve().parent
with (HERE / "fixture.json").open(encoding="utf-8") as fixture_file:
    FIXTURE = json.load(fixture_file)


class PreprocessingContractTests(unittest.TestCase):
    def test_fixture_has_six_records_and_balanced_explicit_splits(self):
        records = FIXTURE["records"]
        self.assertEqual(len(records), 6)
        self.assertEqual(FIXTURE["split_seed"], 43)
        self.assertTrue(validate_splits(records))
        self.assertEqual(len({record["record_id"] for record in records}), 6)
        self.assertEqual(
            Counter(record["split"] for record in records),
            {"train": 2, "validation": 2, "test": 2},
        )
        for split in ("train", "validation", "test"):
            self.assertEqual(
                {record["label"] for record in records if record["split"] == split},
                {0, 1},
            )

    def test_transform_values_float32_shape_and_repeatability(self):
        zero = transform_pixels(FIXTURE["records"][0]["pixels"])
        diagonal = transform_pixels(FIXTURE["records"][1]["pixels"])

        self.assertEqual(zero.shape, (4, 4, 1))
        self.assertEqual(diagonal.shape, (4, 4, 1))
        self.assertEqual(zero.dtype, "float32")
        self.assertEqual(zero.data.itemsize, 4)
        self.assertEqual(len(zero.data), 16)
        self.assertEqual(list(zero.data), [0.0] * 16)
        self.assertEqual(
            [diagonal.data[index * 4 + index] for index in range(4)],
            [1.0] * 4,
        )
        self.assertEqual(
            list(transform_pixels(FIXTURE["records"][1]["pixels"]).data),
            list(diagonal.data),
        )

    def test_transform_is_stateless_and_has_no_fit_interface(self):
        self.assertFalse(hasattr(transform_pixels, "fit"))
        self.assertFalse(hasattr(transform_pixels, "fit_transform"))

    def test_transform_rejects_invalid_shapes_and_pixel_values(self):
        invalid_inputs = (
            [[0, 0], [0, 0]],
            [[0, 0, 0, 0]] * 3,
            [[0, 0, 0, 0]] * 3 + [[0, 0, 0]],
            [[0, 0, 0, 0]] * 3 + [[0, 0, 0, 256]],
            [[0, 0, 0, 0]] * 3 + [[0, 0, 0, -1]],
            [[0, 0, 0, 0]] * 3 + [[0, 0, 0, float("nan")]],
            [["bad", 0, 0, 0]] + [[0, 0, 0, 0]] * 3,
        )
        for pixels in invalid_inputs:
            with self.subTest(pixels=pixels), self.assertRaises(ValueError):
                transform_pixels(pixels)

    def test_split_validation_rejects_group_contamination(self):
        contaminated = [dict(record) for record in FIXTURE["records"]]
        contaminated[2]["group_id"] = contaminated[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "crosses split boundaries"):
            validate_splits(contaminated)

    def test_split_validation_rejects_duplicate_record_ids(self):
        duplicated = [dict(record) for record in FIXTURE["records"]]
        duplicated[1]["record_id"] = duplicated[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_splits(duplicated)


if __name__ == "__main__":
    unittest.main()
