"""Deterministic local simulation of an Agent Jeopardy race.

Scenario files contain ``tasks`` and at least two ``agents``.  A task has an
``id``, ``category``, ``points``, and either ``expected_answer`` (``answer`` is
also accepted) or ``accepted_answers``.  Each agent has an ``id`` and a list
of ``submissions`` with ``ready_at``, ``task_id``, and ``answer``.

Candidates are ordered by effective event time, agent id, then their original
per-agent index.  A candidate resolved after an earlier claim is ``stale`` and
is not submitted.  A same-time loser is ``already_claimed`` and consumes its
agent's submission lane.  Raw candidate and expected answers are deliberately
omitted from the result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import heapq
import json
import pathlib
import sys
from typing import Any


DEFAULT_SUBMISSION_SPACING = Decimal("3.05")
WRONG_ANSWER_PENALTY = Decimal("0.25")
INITIAL_COOLDOWN_SECONDS = Decimal("30")
MAX_COOLDOWN_SECONDS = Decimal("480")
ZERO = Decimal("0")


class ScenarioError(ValueError):
    """Raised when a race scenario is malformed."""


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    points: int
    accepted_answers: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    agent_id: str
    submission_index: int
    ready_at: Decimal
    task_id: str
    answer: str


@dataclass(frozen=True)
class Claim:
    task_id: str
    agent_id: str
    claimed_at: Decimal
    submission_sequence: int


@dataclass
class AgentState:
    score: Decimal = ZERO
    lane_available_at: Decimal = ZERO
    cooldown_until: dict[str, Decimal] = field(default_factory=dict)
    wrong_attempts: dict[str, int] = field(default_factory=dict)
    result_counts: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    submissions: int = 0
    claims: int = 0


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScenarioError(f"{label} must be a JSON array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScenarioError(f"{label} must be a nonempty string")
    return value


def _time_value(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, Decimal)
    ):
        raise ScenarioError(f"{label} must be a finite nonnegative number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ScenarioError(
            f"{label} must be a finite nonnegative number"
        ) from error
    if not result.is_finite() or result < ZERO:
        raise ScenarioError(f"{label} must be a finite nonnegative number")
    return result


def _points_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ScenarioError(f"{label} must be a positive integer")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise ScenarioError(f"{label} must be a positive integer") from error
    if (
        not decimal.is_finite()
        or decimal <= ZERO
        or decimal != decimal.to_integral_value()
    ):
        raise ScenarioError(f"{label} must be a positive integer")
    return int(decimal)


def _accepted_answers(raw: Mapping[str, Any], label: str) -> frozenset[str]:
    expected_keys = [
        key for key in ("expected_answer", "answer") if key in raw
    ]
    has_accepted = "accepted_answers" in raw
    if has_accepted and expected_keys:
        raise ScenarioError(
            f"{label} must use either an expected answer or accepted answers"
        )
    if len(expected_keys) > 1:
        raise ScenarioError(f"{label} has multiple expected-answer fields")
    if has_accepted:
        answers = _list(raw["accepted_answers"], f"{label}.accepted_answers")
        if not answers:
            raise ScenarioError(
                f"{label}.accepted_answers must not be empty"
            )
    elif expected_keys:
        answers = [raw[expected_keys[0]]]
    else:
        raise ScenarioError(
            f"{label} needs expected_answer or accepted_answers"
        )
    if any(not isinstance(answer, str) for answer in answers):
        raise ScenarioError(f"{label} answers must be strings")
    return frozenset(answers)


def _parse_tasks(raw_tasks: Any) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for index, value in enumerate(_list(raw_tasks, "tasks")):
        label = f"tasks[{index}]"
        raw = _mapping(value, label)
        task_id = _nonempty_string(raw.get("id"), f"{label}.id")
        if task_id in tasks:
            raise ScenarioError(f"duplicate task id: {task_id}")
        tasks[task_id] = Task(
            id=task_id,
            category=_nonempty_string(
                raw.get("category"), f"{label}.category"
            ),
            points=_points_value(raw.get("points"), f"{label}.points"),
            accepted_answers=_accepted_answers(raw, label),
        )
    if not tasks:
        raise ScenarioError("tasks must not be empty")
    return tasks


def _parse_agents(
    raw_agents: Any,
    tasks: Mapping[str, Task],
) -> tuple[dict[str, AgentState], list[Candidate]]:
    states: dict[str, AgentState] = {}
    candidates: list[Candidate] = []
    for agent_index, value in enumerate(_list(raw_agents, "agents")):
        label = f"agents[{agent_index}]"
        raw = _mapping(value, label)
        agent_id = _nonempty_string(raw.get("id"), f"{label}.id")
        if agent_id in states:
            raise ScenarioError(f"duplicate agent id: {agent_id}")
        states[agent_id] = AgentState()
        submissions = _list(
            raw.get("submissions"), f"{label}.submissions"
        )
        for submission_index, submission_value in enumerate(submissions):
            submission_label = (
                f"{label}.submissions[{submission_index}]"
            )
            submission = _mapping(submission_value, submission_label)
            task_id = _nonempty_string(
                submission.get("task_id"),
                f"{submission_label}.task_id",
            )
            if task_id not in tasks:
                raise ScenarioError(
                    f"{submission_label} references unknown task {task_id}"
                )
            answer = submission.get("answer")
            if not isinstance(answer, str):
                raise ScenarioError(
                    f"{submission_label}.answer must be a string"
                )
            candidates.append(
                Candidate(
                    agent_id=agent_id,
                    submission_index=submission_index,
                    ready_at=_time_value(
                        submission.get("ready_at"),
                        f"{submission_label}.ready_at",
                    ),
                    task_id=task_id,
                    answer=answer,
                )
            )
    if len(states) < 2:
        raise ScenarioError("agents must contain at least two agents")
    return states, candidates


def _json_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _cooldown_seconds(wrong_attempts: int) -> Decimal:
    if wrong_attempts >= 5:
        return MAX_COOLDOWN_SECONDS
    return min(
        MAX_COOLDOWN_SECONDS,
        INITIAL_COOLDOWN_SECONDS * (2 ** (wrong_attempts - 1)),
    )


def _response_record(
    *,
    sequence: int,
    candidate: Candidate,
    resolved_at: Decimal,
    result: str,
    submitted: bool,
    points_delta: Decimal,
    score: Decimal,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "agent_id": candidate.agent_id,
        "submission_index": candidate.submission_index,
        "task_id": candidate.task_id,
        "ready_at": _json_number(candidate.ready_at),
        "resolved_at": _json_number(resolved_at),
        "submitted_at": _json_number(resolved_at) if submitted else None,
        "delay_seconds": _json_number(resolved_at - candidate.ready_at),
        "result": result,
        "submitted": submitted,
        "points_delta": _json_number(points_delta),
        "score": _json_number(score),
    }


def simulate(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Run a scenario and return JSON-serializable race results."""

    scenario = _mapping(scenario, "scenario")
    tasks = _parse_tasks(scenario.get("tasks"))
    states, candidates = _parse_agents(scenario.get("agents"), tasks)
    spacing_key = (
        "submission_spacing_seconds"
        if "submission_spacing_seconds" in scenario
        else "submission_spacing"
    )
    spacing = _time_value(
        scenario.get(spacing_key, DEFAULT_SUBMISSION_SPACING),
        spacing_key,
    )

    events: list[tuple[Decimal, str, int, Candidate]] = [
        (
            candidate.ready_at,
            candidate.agent_id,
            candidate.submission_index,
            candidate,
        )
        for candidate in candidates
    ]
    heapq.heapify(events)
    claims: dict[str, Claim] = {}
    claim_timeline: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    while events:
        event_at, _, _, candidate = heapq.heappop(events)
        state = states[candidate.agent_id]
        claim = claims.get(candidate.task_id)
        if claim is not None and claim.claimed_at < event_at:
            result = "stale"
            submitted = False
        else:
            available_at = max(
                candidate.ready_at,
                state.lane_available_at,
                state.cooldown_until.get(candidate.task_id, ZERO),
            )
            if available_at > event_at:
                if claim is not None:
                    result = "stale"
                    submitted = False
                else:
                    heapq.heappush(
                        events,
                        (
                            available_at,
                            candidate.agent_id,
                            candidate.submission_index,
                            candidate,
                        ),
                    )
                    continue
            elif claim is not None:
                result = "already_claimed"
                submitted = True
            else:
                task = tasks[candidate.task_id]
                result = (
                    "correct"
                    if candidate.answer in task.accepted_answers
                    else "incorrect"
                )
                submitted = True

        sequence = len(responses)
        points_delta = ZERO
        extra: dict[str, Any] = {}
        if submitted:
            state.submissions += 1
            state.lane_available_at = event_at + spacing

        if result == "correct":
            task = tasks[candidate.task_id]
            points_delta = Decimal(task.points)
            state.score += points_delta
            state.claims += 1
            claim = Claim(
                task_id=task.id,
                agent_id=candidate.agent_id,
                claimed_at=event_at,
                submission_sequence=sequence,
            )
            claims[task.id] = claim
            claim_timeline.append(
                {
                    "sequence": len(claim_timeline),
                    "submission_sequence": sequence,
                    "task_id": task.id,
                    "category": task.category,
                    "points": task.points,
                    "agent_id": candidate.agent_id,
                    "claimed_at": _json_number(event_at),
                }
            )
        elif result == "incorrect":
            task = tasks[candidate.task_id]
            penalty = Decimal(task.points) * WRONG_ANSWER_PENALTY
            points_delta = -penalty
            state.score += points_delta
            wrong_attempts = state.wrong_attempts.get(task.id, 0) + 1
            state.wrong_attempts[task.id] = wrong_attempts
            cooldown = _cooldown_seconds(wrong_attempts)
            cooldown_until = event_at + cooldown
            state.cooldown_until[task.id] = cooldown_until
            extra = {
                "penalty": _json_number(penalty),
                "wrong_attempt": wrong_attempts,
                "cooldown_seconds": _json_number(cooldown),
                "cooldown_until": _json_number(cooldown_until),
            }
        elif claim is not None:
            extra = {
                "claimed_by": claim.agent_id,
                "claimed_at": _json_number(claim.claimed_at),
            }

        state.candidates += 1
        state.result_counts[result] = state.result_counts.get(result, 0) + 1
        record = _response_record(
            sequence=sequence,
            candidate=candidate,
            resolved_at=event_at,
            result=result,
            submitted=submitted,
            points_delta=points_delta,
            score=state.score,
        )
        record.update(extra)
        responses.append(record)

    ordered_states = sorted(
        states.items(),
        key=lambda item: (-item[1].score, item[0]),
    )
    scoreboard = []
    for rank, (agent_id, state) in enumerate(ordered_states, start=1):
        scoreboard.append(
            {
                "rank": rank,
                "agent_id": agent_id,
                "score": _json_number(state.score),
                "claims": state.claims,
                "candidates": state.candidates,
                "submissions": state.submissions,
                "correct": state.result_counts.get("correct", 0),
                "incorrect": state.result_counts.get("incorrect", 0),
                "already_claimed": state.result_counts.get(
                    "already_claimed", 0
                ),
                "stale": state.result_counts.get("stale", 0),
            }
        )

    return {
        "rules": {
            "submission_spacing_seconds": _json_number(spacing),
            "wrong_answer_penalty": _json_number(WRONG_ANSWER_PENALTY),
            "initial_cooldown_seconds": _json_number(
                INITIAL_COOLDOWN_SECONDS
            ),
            "maximum_cooldown_seconds": _json_number(
                MAX_COOLDOWN_SECONDS
            ),
            "simultaneous_order": "event_time, agent_id, submission_index",
        },
        "submissions": responses,
        "claims": claim_timeline,
        "scoreboard": scoreboard,
    }


def load_scenario(path: pathlib.Path) -> Mapping[str, Any]:
    """Load one scenario without consulting environment variables."""

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=Decimal,
    )
    return _mapping(payload, "scenario")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate a deterministic local Agent Jeopardy race."
    )
    parser.add_argument("scenario", type=pathlib.Path)
    parser.add_argument(
        "output",
        nargs="?",
        type=pathlib.Path,
        help="write JSON here instead of stdout",
    )
    arguments = parser.parse_args(argv)

    try:
        result = simulate(load_scenario(arguments.scenario))
        rendered = json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if arguments.output is None:
            sys.stdout.write(rendered)
        else:
            arguments.output.write_text(rendered, encoding="utf-8")
    except (OSError, json.JSONDecodeError, ScenarioError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
