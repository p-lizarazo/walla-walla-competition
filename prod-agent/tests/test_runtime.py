from __future__ import annotations

import pathlib
import tempfile
import unittest

from config import Config
from models import TaskDetail
from runtime import ProductionToolRuntime
from tools import WebSessionPool


def config() -> Config:
    return Config(
        base_url="https://example.test",
        team_api_key="team_test",
        anthropic_base_url="https://example.test/anthropic",
        anthropic_api_key="team_test",
        model="claude-haiku-4-5",
        mode="practice_eval",
        workers=2,
        verifier_workers=0,
        cpu_workers=1,
        max_turns=3,
        max_tokens=4096,
        max_tool_output=12000,
        python_timeout_seconds=10,
        python_memory_mb=256,
        board_poll_seconds=3,
        run_duration_seconds=0,
        submission_interval_seconds=3.1,
        strong_confidence_threshold=0.9,
        urgent_confidence_floor=0.8,
        temperatures=(0.0,),
        thinking_enabled=True,
        thinking_min_points=400,
        thinking_budget_400=1024,
        thinking_budget_500=2048,
        playbooks_path="playbooks.json",
        practice_results_path="practice_results.jsonl",
    )


class RuntimeTests(unittest.TestCase):
    def test_candidate_requires_exactly_one_answer_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = TaskDetail(
                id="PR-X",
                category="Cryptic",
                title="test",
                points=500,
                prompt="decode",
                files=(),
                answer_format="exact",
            )
            runtime = ProductionToolRuntime(
                task,
                pathlib.Path(directory),
                config(),
                WebSessionPool("https://example.test"),
                lambda task_id: {"open": True},
                lambda: {"usage": "ok"},
                0.0,
            )
            result = runtime.execute(
                "record_candidate",
                {
                    "method": "missing",
                    "deterministic_checks": [],
                    "independent_checks": [],
                    "assumptions": [],
                    "tool_errors": [],
                    "input_complete": True,
                    "direct_provenance": True,
                },
            )
        self.assertTrue(result.is_error)

    def test_records_existing_answer_file_without_retyping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "computed.txt").write_text("TOKEN-123\n")
            task = TaskDetail(
                id="PR-X",
                category="Cryptic",
                title="test",
                points=500,
                prompt="decode",
                files=(),
                answer_format="exact",
            )
            runtime = ProductionToolRuntime(
                task,
                path,
                config(),
                WebSessionPool("https://example.test"),
                lambda task_id: {"open": True},
                lambda: {"usage": "ok"},
                0.0,
            )

            result = runtime.execute(
                "record_candidate",
                {
                    "answer_file": "computed.txt",
                    "method": "decoded with Python",
                    "deterministic_checks": ["format checked"],
                    "independent_checks": [],
                    "assumptions": [],
                    "tool_errors": [],
                    "input_complete": True,
                    "direct_provenance": True,
                },
            )

        self.assertFalse(result.is_error)
        self.assertEqual(runtime.candidate.answer, "TOKEN-123")


if __name__ == "__main__":
    unittest.main()
