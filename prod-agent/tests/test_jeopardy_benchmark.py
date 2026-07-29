from __future__ import annotations

import unittest

from evals.jeopardy_benchmark import (
    board_items,
    category_slug,
    percentile,
    safe_relative_path,
    stable_split,
)


class JeopardyBenchmarkTests(unittest.TestCase):
    def test_stable_split_is_balanced_and_deterministic(self) -> None:
        task_ids = ["PR-A1", "PR-A1-2"]
        first = stable_split(
            task_ids, test_fraction=0.5, seed="fixed"
        )
        second = stable_split(
            list(reversed(task_ids)), test_fraction=0.5, seed="fixed"
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "test"})

    def test_board_items_split_variants_within_each_cell(self) -> None:
        payload = {
            "boards": {
                "practice": [
                    {
                        "name": "Cryptic",
                        "tiles": [
                            {
                                "points": 100,
                                "title": "Layer Cake",
                                "variants": [
                                    {"id": "PR-C1"},
                                    {"id": "PR-C1-2"},
                                ],
                            }
                        ],
                    }
                ]
            }
        }
        items = board_items(
            payload,
            "practice",
            split_policy="within-cell",
            test_fraction=0.5,
            seed="fixed",
        )
        self.assertEqual(len(items), 2)
        self.assertEqual({item["split"] for item in items}, {"train", "test"})
        self.assertEqual(len({item["cell_key"] for item in items}), 1)

    def test_all_test_policy_preserves_held_out_board(self) -> None:
        payload = {
            "boards": {
                "qual": [
                    {
                        "name": "Ship It",
                        "tiles": [
                            {
                                "points": 200,
                                "title": "Spec vs Code",
                                "variants": [
                                    {"id": "Q-S2"},
                                    {"id": "Q-S2-2"},
                                ],
                            }
                        ],
                    }
                ]
            }
        }
        items = board_items(
            payload,
            "qual",
            split_policy="all-test",
            test_fraction=0.5,
            seed="ignored",
        )
        self.assertEqual({item["split"] for item in items}, {"test"})

    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6], 0.95), 6)

    def test_task_filenames_cannot_escape_dataset(self) -> None:
        self.assertEqual(str(safe_relative_path("nested/data.csv")), "nested/data.csv")
        for unsafe in ("", "../secret", "/absolute", "a/../secret"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    safe_relative_path(unsafe)

    def test_category_slug_is_portable(self) -> None:
        self.assertEqual(
            category_slug("Needle in the Haystack"),
            "needle-in-the-haystack",
        )


if __name__ == "__main__":
    unittest.main()
