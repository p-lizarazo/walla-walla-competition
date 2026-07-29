from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import unittest

from evals.jeopardy_benchmark import evaluate_dataset
from evals.synthetic_suite import generate


class SyntheticSuiteTests(unittest.TestCase):
    def test_generates_and_evaluates_all_categories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = pathlib.Path(directory) / "suite"
            generate(
                argparse.Namespace(
                    output=str(dataset),
                    seed="test-seed",
                    train_per_category=1,
                    test_per_category=1,
                )
            )
            manifest = json.loads(
                (dataset / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["tasks"]), 12)
            self.assertEqual(
                {task["category"] for task in manifest["tasks"]},
                {
                    "Ancient Scrolls",
                    "Cryptic",
                    "Heavy Compute",
                    "Needle in the Haystack",
                    "Ship It",
                    "The Dark Web",
                },
            )
            exit_code = evaluate_dataset(
                argparse.Namespace(
                    dataset=str(dataset),
                    split="test",
                    category=None,
                    workers=4,
                    submit_practice=False,
                    results=None,
                )
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
