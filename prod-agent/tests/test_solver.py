from __future__ import annotations

import hashlib
from types import SimpleNamespace
import unittest

from config import Config
from models import Candidate, Evidence, TaskDetail
from solver import AnthropicSolver, ToolExecution


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
        python_timeout_seconds=60,
        python_memory_mb=512,
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
        task_filter=(),
        experiment_id="test",
    )


class Playbooks:
    def select_for_task(self, task: TaskDetail) -> tuple[str, ...]:
        return ("Use deterministic parsing.",)


class Runtime:
    def __init__(self) -> None:
        self._candidate = None

    @property
    def candidate(self):
        return self._candidate

    def execute(self, name, arguments):
        if name == "record_candidate":
            answer = arguments["answer"]
            self._candidate = Candidate(
                task_id="PR-X",
                answer=answer,
                answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                evidence=Evidence(
                    method=arguments["method"],
                    deterministic_checks=tuple(arguments["deterministic_checks"]),
                ),
                model_temperature=0,
                elapsed_seconds=0,
                tool_turns=0,
            )
        return ToolExecution("ok")


class SolverTests(unittest.TestCase):
    def test_tool_schemas_avoid_unsupported_combinators(self) -> None:
        from solver import TOOLS

        for tool in TOOLS:
            schema = tool["input_schema"]
            self.assertNotIn("oneOf", schema)
            self.assertNotIn("allOf", schema)
            self.assertNotIn("anyOf", schema)

    def test_returns_candidate_without_submitting(self) -> None:
        tool_use = SimpleNamespace(
            type="tool_use",
            name="record_candidate",
            id="call-1",
            input={
                "answer": "42",
                "method": "computed",
                "deterministic_checks": ["recomputed"],
                "independent_checks": [],
                "assumptions": [],
                "tool_errors": [],
                "input_complete": True,
                "direct_provenance": True,
            },
        )
        response = SimpleNamespace(content=[tool_use])
        messages = SimpleNamespace(create=lambda **kwargs: response)
        client = SimpleNamespace(messages=messages)
        solver = AnthropicSolver(
            config(), Playbooks(), client_factory=lambda: client
        )
        task = TaskDetail(
            id="PR-X",
            category="Heavy Compute",
            title="test",
            points=500,
            prompt="Compute it.",
            files=(),
            answer_format="numeric",
        )

        candidate = solver.solve(task, "/tmp/test", 0.0, Runtime())

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.answer, "42")
        self.assertEqual(candidate.tool_turns, 1)


if __name__ == "__main__":
    unittest.main()
