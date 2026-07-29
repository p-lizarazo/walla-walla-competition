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
        self.assertEqual(config.cpu_workers, 1)

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


if __name__ == "__main__":
    unittest.main()
