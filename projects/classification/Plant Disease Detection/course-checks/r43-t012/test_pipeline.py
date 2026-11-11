import unittest
from pathlib import Path

from pipeline import create_split_manifest, load_fixture


HERE = Path(__file__).resolve().parent


class SplitManifestTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(HERE / "fixture.json")
        cls.records = cls.fixture["records"]

    def test_fixture_has_eight_synthetic_records(self):
        self.assertEqual(len(self.records), 8)
        self.assertIn("Synthetic", self.fixture["description"])

    def test_all_ids_appear_exactly_once(self):
        manifest = create_split_manifest(self.records, seed=43)
        ids = [
            record_id
            for split in manifest["splits"].values()
            for record_id in split
        ]
        expected = [row["record_id"] for row in self.records]
        self.assertCountEqual(ids, expected)
        self.assertEqual(len(ids), len(set(ids)))

    def test_split_is_deterministic(self):
        first = create_split_manifest(self.records, seed=43)
        second = create_split_manifest(self.records, seed=43)
        self.assertEqual(first, second)

    def test_split_names_and_counts(self):
        manifest = create_split_manifest(self.records, seed=43)
        self.assertEqual(
            set(manifest["splits"]),
            {"train", "validation", "test"},
        )
        self.assertEqual(
            [
                len(manifest["splits"][name])
                for name in ("train", "validation", "test")
            ],
            [4, 2, 2],
        )

    def test_groups_do_not_cross_splits(self):
        manifest = create_split_manifest(self.records, seed=43)
        id_to_group = {
            row["record_id"]: row["group_id"]
            for row in self.records
        }
        group_sets = [
            {id_to_group[item] for item in manifest["splits"][name]}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

    def test_duplicate_ids_are_rejected(self):
        duplicate = [dict(row) for row in self.records]
        duplicate[1]["record_id"] = duplicate[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "record_id"):
            create_split_manifest(duplicate)

    def test_chronological_split(self):
        rows = [
            {
                "record_id": f"T{i}",
                "timestamp": f"2026-01-{i:02d}",
                "label": i % 2,
            }
            for i in range(1, 9)
        ]
        manifest = create_split_manifest(rows, time_field="timestamp")
        self.assertEqual(manifest["strategy"], "chronological")
        self.assertEqual(
            manifest["splits"]["train"],
            ["T1", "T2", "T3", "T4"],
        )
        self.assertEqual(
            manifest["splits"]["validation"],
            ["T5", "T6"],
        )
        self.assertEqual(
            manifest["splits"]["test"],
            ["T7", "T8"],
        )

    def test_missing_time_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing time field"):
            create_split_manifest(self.records, time_field="timestamp")


if __name__ == "__main__":
    unittest.main()
