from __future__ import annotations

import json
import unittest

from evals.race_simulator import DEFAULT_SUBMISSION_SPACING, simulate


def task(
    task_id: str,
    points: int,
    *,
    expected_answer: str | None = None,
    accepted_answers: list[str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": task_id,
        "category": "Cryptic",
        "points": points,
    }
    if accepted_answers is not None:
        result["accepted_answers"] = accepted_answers
    else:
        result["expected_answer"] = expected_answer
    return result


def agent(
    agent_id: str,
    *submissions: tuple[float, str, str],
) -> dict[str, object]:
    return {
        "id": agent_id,
        "submissions": [
            {
                "ready_at": ready_at,
                "task_id": task_id,
                "answer": answer,
            }
            for ready_at, task_id, answer in submissions
        ],
    }


def scoreboard_by_agent(result: dict[str, object]) -> dict[str, dict]:
    return {
        row["agent_id"]: row
        for row in result["scoreboard"]
    }


class RaceSimulatorTests(unittest.TestCase):
    def test_first_claim_makes_later_candidate_stale(self) -> None:
        scenario = {
            "tasks": [
                task(
                    "T100",
                    100,
                    expected_answer="do-not-emit-this-answer",
                )
            ],
            "agents": [
                agent("alpha", (1.0, "T100", "do-not-emit-this-answer")),
                agent("beta", (2.0, "T100", "do-not-emit-this-answer")),
            ],
        }

        result = simulate(scenario)

        self.assertEqual(
            [submission["result"] for submission in result["submissions"]],
            ["correct", "stale"],
        )
        self.assertFalse(result["submissions"][1]["submitted"])
        self.assertIsNone(result["submissions"][1]["submitted_at"])
        self.assertEqual(
            result["claims"],
            [
                {
                    "sequence": 0,
                    "submission_sequence": 0,
                    "task_id": "T100",
                    "category": "Cryptic",
                    "points": 100,
                    "agent_id": "alpha",
                    "claimed_at": 1,
                }
            ],
        )
        scores = scoreboard_by_agent(result)
        self.assertEqual(scores["alpha"]["score"], 100)
        self.assertEqual(scores["beta"]["score"], 0)
        self.assertNotIn(
            "do-not-emit-this-answer",
            json.dumps(result, sort_keys=True),
        )

    def test_submission_lane_spacing_is_per_agent(self) -> None:
        scenario = {
            "tasks": [
                task("A", 100, expected_answer="a"),
                task("B", 100, expected_answer="b"),
                task("C", 100, expected_answer="c"),
            ],
            "agents": [
                agent(
                    "alpha",
                    (0.0, "A", "a"),
                    (0.0, "B", "b"),
                ),
                agent("beta", (1.0, "C", "c")),
            ],
        }

        result = simulate(scenario)

        submitted_at = {
            (row["agent_id"], row["task_id"]): row["submitted_at"]
            for row in result["submissions"]
        }
        self.assertEqual(submitted_at[("alpha", "A")], 0)
        self.assertEqual(
            submitted_at[("alpha", "B")],
            float(DEFAULT_SUBMISSION_SPACING),
        )
        self.assertEqual(submitted_at[("beta", "C")], 1)
        self.assertEqual(
            [claim["task_id"] for claim in result["claims"]],
            ["A", "C", "B"],
        )

    def test_wrong_penalty_and_cooldown_double_and_cap(self) -> None:
        attempts = [
            (0.0, "T200", "wrong-1"),
            (1.0, "T200", "wrong-2"),
            (2.0, "T200", "wrong-3"),
            (3.0, "T200", "wrong-4"),
            (4.0, "T200", "wrong-5"),
            (5.0, "T200", "wrong-6"),
            (6.0, "T200", "accepted-alternative"),
        ]
        scenario = {
            "tasks": [
                task(
                    "T200",
                    200,
                    accepted_answers=["expected", "accepted-alternative"],
                )
            ],
            "agents": [
                agent("alpha", *attempts),
                agent("beta"),
            ],
        }

        result = simulate(scenario)

        wrong = [
            row
            for row in result["submissions"]
            if row["result"] == "incorrect"
        ]
        self.assertEqual(
            [row["submitted_at"] for row in wrong],
            [0, 30, 90, 210, 450, 930],
        )
        self.assertEqual(
            [row["cooldown_seconds"] for row in wrong],
            [30, 60, 120, 240, 480, 480],
        )
        final = result["submissions"][-1]
        self.assertEqual(final["result"], "correct")
        self.assertEqual(final["submitted_at"], 1410)
        scores = scoreboard_by_agent(result)
        self.assertEqual(scores["alpha"]["score"], -100)
        self.assertEqual(scores["alpha"]["incorrect"], 6)
        self.assertEqual(scores["alpha"]["correct"], 1)

    def test_simultaneous_ties_use_agent_id_order(self) -> None:
        tasks = [task("T500", 500, expected_answer="winner")]
        alpha = agent("alpha", (0.0, "T500", "winner"))
        beta = agent("beta", (0.0, "T500", "winner"))

        forward = simulate({"tasks": tasks, "agents": [alpha, beta]})
        reversed_agents = simulate(
            {"tasks": tasks, "agents": [beta, alpha]}
        )

        self.assertEqual(forward, reversed_agents)
        self.assertEqual(forward["claims"][0]["agent_id"], "alpha")
        self.assertEqual(
            [row["result"] for row in forward["submissions"]],
            ["correct", "already_claimed"],
        )
        loser = forward["submissions"][1]
        self.assertTrue(loser["submitted"])
        self.assertEqual(loser["claimed_by"], "alpha")
        self.assertEqual(loser["submitted_at"], 0)


if __name__ == "__main__":
    unittest.main()
