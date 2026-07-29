from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import Mock

from config import Config
from game_client import GameClient
from models import Phase


def config() -> Config:
    return Config(
        base_url="https://example.test",
        team_api_key="team_test",
        anthropic_base_url="https://example.test/anthropic",
        anthropic_api_key="team_test",
        model="claude-haiku-4-5",
        mode="scored",
        workers=4,
        verifier_workers=1,
        cpu_workers=1,
        max_turns=20,
        max_tokens=4096,
        max_tool_output=12000,
        python_timeout_seconds=60,
        python_memory_mb=512,
        board_poll_seconds=3,
        run_duration_seconds=0,
        submission_interval_seconds=3.1,
        strong_confidence_threshold=0.9,
        urgent_confidence_floor=0.8,
        temperatures=(0.0, 0.25, 0.5),
        thinking_enabled=True,
        thinking_min_points=400,
        thinking_budget_400=1024,
        thinking_budget_500=2048,
        playbooks_path="playbooks.json",
        practice_results_path="practice_results.jsonl",
        task_filter=(),
        experiment_id="test",
    )


class GameClientTests(unittest.TestCase):
    def test_flattens_every_open_variant(self) -> None:
        response = Mock()
        response.status_code = 200
        response.ok = True
        response.json.return_value = {
            "phase": "round1",
            "server_time": "now",
            "you": {"solved_ids": ["Q-A"]},
            "boards": {
                "qual": [
                    {
                        "name": "Cryptic",
                        "tiles": [
                            {
                                "points": 500,
                                "open_ids": ["Q-A", "Q-B", "Q-C"],
                                "remaining": 3,
                                "total": 3,
                            }
                        ],
                    }
                ]
            },
        }
        session = Mock()
        session.get.return_value = response
        client = GameClient(config())
        client._local.session = session

        board = client.board()

        self.assertEqual(board.phase, Phase.ROUND1)
        self.assertEqual([tile.id for tile in board.tiles], ["Q-B", "Q-C"])

    def test_dashboard_redacts_secret_named_fields(self) -> None:
        response = Mock()
        response.status_code = 200
        response.ok = True
        response.json.return_value = {
            "score": 100,
            "api_key": "secret",
            "token_budget": 123,
        }
        session = Mock()
        session.get.return_value = response
        client = GameClient(config())
        client._local.session = session

        dashboard = client.dashboard()

        self.assertEqual(
            dashboard.payload, {"score": 100, "token_budget": 123}
        )

    def test_practice_eval_includes_previously_solved_tiles(self) -> None:
        response = Mock()
        response.status_code = 200
        response.ok = True
        response.json.return_value = {
            "phase": "practice",
            "you": {"solved_ids": ["PR-A1"]},
            "boards": {
                "practice": [
                    {
                        "name": "Ancient Scrolls",
                        "tiles": [
                            {
                                "points": 100,
                                "open_ids": ["PR-A1"],
                                "remaining": 1,
                                "total": 1,
                            }
                        ],
                    }
                ]
            },
        }
        practice_config = replace(config(), mode="practice_eval")
        session = Mock()
        session.get.return_value = response
        client = GameClient(practice_config)
        client._local.session = session

        self.assertEqual([tile.id for tile in client.board().tiles], ["PR-A1"])


if __name__ == "__main__":
    unittest.main()
