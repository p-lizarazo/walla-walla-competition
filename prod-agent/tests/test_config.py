from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import Config


class ConfigTests(unittest.TestCase):
    def base_env(self) -> dict[str, str]:
        return {
            "JEOPARDY_BASE_URL": "https://example.test",
            "TEAM_API_KEY": "team_test",
        }

    def test_defaults_are_production_safe(self) -> None:
        with patch.dict(os.environ, self.base_env(), clear=True):
            config = Config.from_env()
        self.assertEqual(config.mode, "auto")
        self.assertEqual(config.model, "claude-haiku-4-5")
        self.assertGreaterEqual(config.submission_interval_seconds, 3)
        self.assertEqual(config.workers, 6)
        self.assertEqual(config.verifier_workers, 1)
        self.assertEqual(config.cpu_workers, 2)
        self.assertEqual(config.board_poll_seconds, 1.5)
        self.assertEqual(config.solve_turns(300), 6)
        self.assertEqual(config.solve_turns(400), 10)
        self.assertEqual(config.solve_turns(500), 12)
        self.assertEqual(config.solve_tokens(300), 1536)
        self.assertEqual(config.solve_tokens(400), 3072)
        self.assertEqual(config.solve_tokens(500), 4096)

    def test_rejects_invalid_confidence_order(self) -> None:
        environment = {
            **self.base_env(),
            "STRONG_CONFIDENCE_THRESHOLD": "0.8",
            "URGENT_CONFIDENCE_FLOOR": "0.9",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValueError):
                Config.from_env()

    def test_allows_zero_verifier_workers(self) -> None:
        environment = {**self.base_env(), "VERIFIER_WORKERS": "0"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(Config.from_env().verifier_workers, 0)

    def test_tier_budgets_are_independently_configurable(self) -> None:
        environment = {
            **self.base_env(),
            "MAX_TURNS": "4",
            "MAX_TURNS_400": "7",
            "MAX_TURNS_500": "9",
            "MAX_TOKENS": "1000",
            "MAX_TOKENS_400": "2000",
            "MAX_TOKENS_500": "3000",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = Config.from_env()
        self.assertEqual(
            [config.solve_turns(points) for points in (100, 400, 500)],
            [4, 7, 9],
        )
        self.assertEqual(
            [config.solve_tokens(points) for points in (100, 400, 500)],
            [1000, 2000, 3000],
        )


if __name__ == "__main__":
    unittest.main()
