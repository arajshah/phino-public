"""Tests for deterministic asset-disjoint physics probe splits."""

import unittest

from train.data.splits import split_metadata_by_asset


class TestGroupedSplits(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"asset_id": f"asset_{asset}", "mass": float(mass)}
            for asset in range(20)
            for mass in range(5)
        ]

    def test_no_asset_overlap(self):
        parts = {s: split_metadata_by_asset(self.rows, s) for s in ("train", "val", "test")}
        ids = {s: {r["asset_id"] for r in rows} for s, rows in parts.items()}
        self.assertTrue(ids["train"].isdisjoint(ids["val"]))
        self.assertTrue(ids["train"].isdisjoint(ids["test"]))
        self.assertTrue(ids["val"].isdisjoint(ids["test"]))

    def test_all_entries_preserved(self):
        parts = [split_metadata_by_asset(self.rows, s) for s in ("train", "val", "test")]
        self.assertEqual(sum(map(len, parts)), len(self.rows))

    def test_expected_asset_counts(self):
        counts = [len({r["asset_id"] for r in split_metadata_by_asset(self.rows, s)})
                  for s in ("train", "val", "test")]
        self.assertEqual(counts, [14, 3, 3])

    def test_reproducible(self):
        self.assertEqual(
            split_metadata_by_asset(self.rows, "train", seed=12),
            split_metadata_by_asset(self.rows, "train", seed=12),
        )

    def test_smallest_valid_input(self):
        counts = [len(split_metadata_by_asset(self.rows[:15], s))
                  for s in ("train", "val", "test")]
        self.assertEqual(counts, [5, 5, 5])

    def test_unknown_split(self):
        with self.assertRaises(ValueError):
            split_metadata_by_asset(self.rows, "validation")

    def test_missing_id_or_too_few_assets(self):
        with self.assertRaises(ValueError):
            split_metadata_by_asset([{"mass": 1.0}], "train")
        with self.assertRaises(ValueError):
            split_metadata_by_asset(self.rows[:10], "train")


if __name__ == "__main__":
    unittest.main()
